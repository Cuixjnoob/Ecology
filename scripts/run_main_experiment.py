"""Master experiment runner: Eco-GNRD on all datasets with 10 seeds.

Saves results to 重要实验/results/main/eco_gnrd_alt5_hdyn/{dataset}/
Each species gets its own folder with per-seed metrics.json and summary.md.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_beninca_nbeddyn import (
    LatentDynamicsNet, make_model as make_model_default, get_param_groups,
    freeze, unfreeze, alpha_schedule, HP,
)

SEEDS = [42, 123, 456, 789, 2024, 11111, 22222, 27182, 31415, 65537]


# ===== Dataset configs =====
DATASET_CONFIGS = {
    'huisman': {
        'use_alt': False, 'lam_hdyn': 0.2, 'lr': 0.0008,
        'lam_hf': 0.0, 'epochs': 500,
        'encoder_d': 64, 'encoder_blocks': 2,
        'beta_kl': 0.03, 'margin_null': 0.003, 'margin_shuf': 0.002,
        'lam_necessary': 5.0, 'min_energy': 0.02,
    },
    'beninca': {
        'use_alt': True, 'pa': 5, 'pb': 1,
        'lam_hdyn': 0.5, 'lr': 0.0006,
        'lam_hf': 0.2, 'epochs': 500,
        'encoder_d': 96, 'encoder_blocks': 3,
        'beta_kl': 0.017, 'margin_null': 0.002, 'margin_shuf': 0.001,
        'lam_necessary': 9.5, 'min_energy': 0.14,
    },
    'maizuru': {
        'use_alt': True, 'pa': 5, 'pb': 1,
        'lam_hdyn': 0.5, 'lr': 0.0006,
        'lam_hf': 0.2, 'epochs': 500,
        'encoder_d': 96, 'encoder_blocks': 3,
        'beta_kl': 0.017, 'margin_null': 0.002, 'margin_shuf': 0.001,
        'lam_necessary': 9.5, 'min_energy': 0.14,
    },
}


def make_model(N, cfg, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=cfg.get('use_formula_hints', True),
        use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def train_one(visible, hidden, seed, device, cfg, n_recon_ch=None):
    torch.manual_seed(seed)
    T, N = visible.shape
    epochs = cfg['epochs']
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, cfg, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=cfg['lr'], weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    pa = cfg.get('pa', 5); pb = cfg.get('pb', 1)
    use_alt = cfg.get('use_alt', False)
    DETACH_UNTIL = 100

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float('inf'); best_state = None; best_fh = None
    loss_history = []  # per-epoch loss curve

    for epoch in range(epochs):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup: h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup; h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            max_K = cfg.get('max_rollout_K', 3)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * max_K)))

        if use_alt and epoch >= warmup:
            cyc = (epoch - warmup) % (pa + pb)
            if cyc < pa:
                freeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(False)
                unfreeze(fvis_params)
            else:
                freeze(fvis_params); unfreeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(True)
        else:
            unfreeze(fvis_params); unfreeze(enc_params)
            for p in f_h.parameters(): p.requires_grad_(True)

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); f_h.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]; tr['G'] = out['G'][:, :train_end]
        losses = model.loss(tr, beta_kl=cfg['beta_kl'], free_bits=0.02,
            margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
            lam_necessary=cfg['lam_necessary'], lam_shuffle=cfg['lam_necessary'] * 0.6,
            lam_energy=2.0, min_energy=cfg['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=cfg['lam_hf'], lowpass_sigma=6.0,
            lam_rmse_log=0.1,
            n_recon_channels=n_recon_ch)
        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and cfg['lam_hdyn'] > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            tgt = hm[:, 1:].detach() if epoch < DETACH_UNTIL + warmup else hm[:, 1:]
            loss_ode = F.mse_loss(hp, tgt)
        (losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, h_weight=1.0,
                margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
                lam_energy=2.0, min_energy=cfg['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=cfg['lam_hf'], lowpass_sigma=6.0,
                n_recon_channels=n_recon_ch)
            vr = vl['recon_full'].item()
        # Record loss curve
        loss_history.append({
            'epoch': epoch,
            'train_loss': losses['total'].item(),
            'train_recon': losses['recon_full'].item(),
            'val_recon': vr,
            'h_weight': h_w,
            'ode_loss': loss_ode.item() if isinstance(loss_ode, torch.Tensor) else 0.0,
        })

        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    unfreeze(fvis_params); unfreeze(enc_params)
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()

        # d_ratio: inject hidden_true into learned dynamics
        h_true_norm = (hidden - hidden.mean()) / (hidden.std() + 1e-8)
        h_true_t = torch.tensor(h_true_norm, dtype=torch.float32, device=device).unsqueeze(0)
        vis_safe = torch.clamp(oe['visible'], min=1e-6)
        log_ratio_actual = torch.log(vis_safe[:, 1:] / vis_safe[:, :-1])
        # Recon with encoder h
        recon_enc = ((oe['pred_full'].mean(dim=0) - log_ratio_actual) ** 2).mean().item()
        # Recon with true h substituted
        base = oe['base'][:, :-1]
        G = oe['G'][:, :-1] if 'G' in oe else torch.ones_like(base)
        pred_true = base + h_true_t[:, :-1].unsqueeze(-1) * G
        nc = n_recon_ch if n_recon_ch else pred_true.shape[-1]
        recon_true = ((pred_true[..., :nc] - log_ratio_actual[..., :nc]) ** 2).mean().item()
        d_ratio = recon_true / (recon_enc + 1e-10)

        # Counterfactual margin
        T_null = min(oe['pred_null'].shape[1], log_ratio_actual.shape[1])
        recon_null = ((oe['pred_null'][:, :T_null] - log_ratio_actual[:, :T_null]) ** 2).mean().item()
        margin = recon_null - recon_enc

    L = min(len(h_mean), len(hidden))
    te = train_end
    # All-data Pearson
    X_all = np.column_stack([h_mean[:L], np.ones(L)])
    coef_all, _, _, _ = np.linalg.lstsq(X_all, hidden[:L], rcond=None)
    h_sc_all = X_all @ coef_all
    r_all = float(np.corrcoef(h_sc_all, hidden[:L])[0, 1])
    # Val-only Pearson
    X_tr = np.column_stack([h_mean[:te], np.ones(te)])
    coef_tr, _, _, _ = np.linalg.lstsq(X_tr, hidden[:te], rcond=None)
    X_val = np.column_stack([h_mean[te:L], np.ones(L - te)])
    h_sc_val = X_val @ coef_tr
    r_val = float(np.corrcoef(h_sc_val, hidden[te:L])[0, 1]) if L > te + 2 else 0

    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {
        'pearson_all': r_all, 'pearson_val': r_val,
        'val_recon': best_val,
        'd_ratio': d_ratio, 'margin': margin,
        'recon_enc': recon_enc, 'recon_null': recon_null,
        'config': {k: v for k, v in cfg.items()},
        'h_mean': h_mean, 'h_scaled': h_sc_all,
        'loss_history': loss_history,
    }


def load_dataset(name):
    """Load dataset, return list of (visible, hidden, sp_name, n_recon_ch)."""
    if name == 'huisman':
        d = np.load("runs/huisman1999_chaos/trajectories.npz")
        N_all = d["N_all"]; R_all = d["resources"]
        tasks = []
        for sp_idx in range(6):
            vis = np.concatenate([np.delete(N_all, sp_idx, axis=1), R_all], axis=1)
            vis = (vis + 0.01) / (vis.mean(axis=0, keepdims=True) + 1e-3)
            hid = N_all[:, sp_idx]
            hid = (hid + 0.01) / (hid.mean() + 1e-3)
            tasks.append((vis.astype(np.float32), hid.astype(np.float32), f'sp{sp_idx+1}', 5))
        return tasks
    elif name == 'beninca':
        from scripts.load_beninca import load_beninca
        full, species, _ = load_beninca(include_nutrients=True)
        species = [str(s) for s in species]
        SP = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto", "Picophyto",
              "Filam_diatoms", "Ostracods", "Harpacticoids", "Bacteria"]
        n_sp = len(SP)
        tasks = []
        for h in SP:
            h_idx = species.index(h)
            vis = np.delete(full, h_idx, axis=1).astype(np.float32)
            hid = full[:, h_idx].astype(np.float32)
            tasks.append((vis, hid, h, n_sp - 1))
        return tasks
    elif name == 'maizuru':
        from scripts.load_maizuru import load_maizuru
        full, species, _ = load_maizuru(include_temp=False)
        species = [str(s) for s in species]
        tasks = []
        for h in species:
            h_idx = species.index(h)
            vis = np.delete(full, h_idx, axis=1).astype(np.float32)
            hid = full[:, h_idx].astype(np.float32)
            tasks.append((vis, hid, h, len(species) - 1))
        return tasks
    else:
        raise ValueError(f"Unknown dataset: {name}")


def run_dataset(ds_name, out_root, device, seeds=SEEDS):
    cfg = DATASET_CONFIGS[ds_name]
    tasks = load_dataset(ds_name)
    ds_dir = out_root / ds_name
    ds_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"DATASET: {ds_name} ({len(tasks)} species, {len(seeds)} seeds)")
    print(f"Config: alt={cfg.get('use_alt')}, hdyn={cfg['lam_hdyn']}, lr={cfg['lr']}")
    print(f"{'='*70}")

    all_results = []
    total = len(tasks) * len(seeds)
    ri = 0

    for vis, hid, sp_name, n_rc in tasks:
        sp_dir = ds_dir / sp_name
        sp_dir.mkdir(parents=True, exist_ok=True)
        sp_results = []

        for seed in seeds:
            # Skip if already done
            seed_dir = sp_dir / f"seed_{seed:05d}"
            metrics_file = seed_dir / "metrics.json"
            if metrics_file.exists():
                with open(metrics_file) as f:
                    r = json.load(f)
                sp_results.append(r)
                ri += 1
                print(f"  [{ri}/{total}] {sp_name} seed={seed} CACHED  all={r['pearson_all']:+.3f} val={r['pearson_val']:+.3f}")
                continue

            ri += 1
            t0 = datetime.now()
            result = train_one(vis, hid, seed, device, cfg, n_recon_ch=n_rc)
            dt = (datetime.now() - t0).total_seconds()

            # Save
            seed_dir.mkdir(parents=True, exist_ok=True)
            metrics = {
                'seed': seed, 'dataset': ds_name, 'species': sp_name,
                'method': 'eco_gnrd_alt5_hdyn',
                'pearson_all': result['pearson_all'],
                'pearson_val': result['pearson_val'],
                'val_recon': result['val_recon'],
                'd_ratio': result.get('d_ratio', None),
                'margin': result.get('margin', None),
                'recon_enc': result.get('recon_enc', None),
                'recon_null': result.get('recon_null', None),
                'config': result.get('config', {}),
            }
            with open(metrics_file, 'w') as f:
                json.dump(metrics, f, indent=2)
            # Save trajectory
            np.savez(seed_dir / "trajectory.npz",
                     h_mean=result['h_mean'], h_scaled=result['h_scaled'],
                     hidden_true=hid)
            # Save loss curve as CSV
            import csv
            with open(seed_dir / "loss_curve.csv", 'w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=['epoch', 'train_loss', 'train_recon',
                                                        'val_recon', 'h_weight', 'ode_loss'])
                writer.writeheader()
                for row in result.get('loss_history', []):
                    writer.writerow(row)
            # Summary
            with open(seed_dir / "summary.md", 'w', encoding='utf-8') as f:
                f.write(f"# {ds_name} / {sp_name} / seed={seed}\n\n")
                f.write(f"- P(all): {result['pearson_all']:+.3f}\n")
                f.write(f"- P(val): {result['pearson_val']:+.3f}\n")
                f.write(f"- val_recon: {result['val_recon']:.4f}\n")
                f.write(f"- d_ratio: {result.get('d_ratio', 'N/A')}\n")
                f.write(f"- margin: {result.get('margin', 'N/A')}\n")
                f.write(f"- config: {result.get('config', {})}\n")

            sp_results.append(metrics)
            print(f"  [{ri}/{total}] {sp_name} seed={seed}  all={result['pearson_all']:+.3f} val={result['pearson_val']:+.3f}  ({dt:.1f}s)")

        # Species aggregate
        pa = np.mean([r['pearson_all'] for r in sp_results])
        pv = np.mean([r['pearson_val'] for r in sp_results])
        agg = {'species': sp_name, 'n_seeds': len(sp_results),
               'pearson_all_mean': pa, 'pearson_val_mean': pv,
               'pearson_all_std': np.std([r['pearson_all'] for r in sp_results]),
               'pearson_val_std': np.std([r['pearson_val'] for r in sp_results])}
        with open(sp_dir / "aggregate.json", 'w') as f:
            json.dump(agg, f, indent=2)
        all_results.append(agg)

    # Dataset aggregate
    oa = np.mean([r['pearson_all_mean'] for r in all_results])
    ov = np.mean([r['pearson_val_mean'] for r in all_results])
    ds_agg = {'dataset': ds_name, 'method': 'eco_gnrd_alt5_hdyn',
              'overall_pearson_all_mean': oa, 'overall_pearson_val_mean': ov,
              'per_species': all_results}
    with open(ds_dir / "aggregate.json", 'w') as f:
        json.dump(ds_agg, f, indent=2)
    with open(ds_dir / "summary.md", 'w', encoding='utf-8') as f:
        f.write(f"# {ds_name}: Eco-GNRD alt5+hdyn\n\n")
        f.write("| Species | P(all) | P(val) |\n|---|---|---|\n")
        for r in all_results:
            f.write(f"| {r['species']} | {r['pearson_all_mean']:+.3f} | {r['pearson_val_mean']:+.3f} |\n")
        f.write(f"\n**Overall**: P(all)={oa:+.3f}, P(val)={ov:+.3f}\n")

    print(f"\n{ds_name} DONE: P(all)={oa:+.3f}, P(val)={ov:+.3f}")
    return ds_agg


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['maizuru'],
                        help='Datasets to run')
    parser.add_argument('--seeds', type=int, default=10)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path("重要实验/results/main/eco_gnrd_alt5_hdyn")

    seeds = SEEDS[:args.seeds]
    for ds in args.datasets:
        run_dataset(ds, out_root, device, seeds=seeds)


if __name__ == "__main__":
    main()

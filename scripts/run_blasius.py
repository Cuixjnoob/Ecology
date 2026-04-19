"""Eco-GNRD on Blasius 2020 chemostat data.

3 nodes (algae, rotifers, eggs), 2->1 hidden recovery.
Run on all experiments with T >= 100.
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
    LatentDynamicsNet, get_param_groups,
    freeze, unfreeze, alpha_schedule,
)
from scripts.load_blasius import load_all_blasius, SPECIES

SEEDS = [42, 123, 456, 789, 2024]
EPOCHS = 400


def make_model(N, device):
    """Smaller model for 3-species system."""
    return CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=2,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=12, f_visible_layers=2, f_visible_top_k=2,
        d_species_G=8, G_field_layers=1, G_field_top_k=2,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def train_one(visible, hidden, seed, device):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    f_h = LatentDynamicsNet(N, d_hidden=16).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * EPOCHS)
    ramp = max(1, int(0.2 * EPOCHS))

    def lr_lambda(step):
        if step < 30: return step / 30
        p = (step - 30) / max(1, EPOCHS - 30)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, EPOCHS)
        if epoch < warmup:
            h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, EPOCHS - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))

        # No alternating for small system (like Huisman)
        unfreeze(fvis_params)
        unfreeze(enc_params)
        for p in f_h.parameters():
            p.requires_grad_(True)

        model.train()
        f_h.train()
        opt.zero_grad()
        out = model(x_full, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]
        tr['G'] = out['G'][:, :train_end]
        losses = model.loss(
            tr, beta_kl=0.03, free_bits=0.02,
            margin_null=0.003, margin_shuf=0.002,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=0.02,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0, lam_rmse_log=0.1,
        )
        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0:
            hm = out['mu'][:, :train_end]
            xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = F.mse_loss(hp, hm[:, 1:].detach())

        (losses['total'] + 0.2 * h_w * loss_ode).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]
            vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, h_weight=1.0,
                margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02,
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25))
            vr = vl['recon_full'].item()

        if epoch > warmup + 10 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()

    L = min(len(h_mean), len(hidden))
    te = train_end
    X_tr = np.column_stack([h_mean[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:te], rcond=None)
    X_all = np.column_stack([h_mean[:L], np.ones(L)])
    h_sc = X_all @ coef
    r_all = float(np.corrcoef(h_sc, hidden[:L])[0, 1])
    r_val = float(np.corrcoef(h_sc[te:L], hidden[te:L])[0, 1]) if L > te + 2 else 0

    del model, f_h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r_all, r_val, h_mean, h_sc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path("重要实验/results/main/eco_gnrd_alt5_hdyn/blasius")
    out_root.mkdir(parents=True, exist_ok=True)

    print("Eco-GNRD on Blasius 2020 chemostat data")
    print(f"3 nodes: algae, rotifers, eggs | {len(SEEDS)} seeds\n")

    experiments = load_all_blasius(min_T=100)
    print(f"\n{len(experiments)} experiments loaded\n")

    all_exp_results = []

    for data, species, meta in experiments:
        exp_id = meta['exp_id']
        exp_dir = out_root / f"C{exp_id}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'='*60}")
        print(f"Experiment C{exp_id} (T={meta['T']})")
        print(f"{'='*60}")

        exp_results = []
        for h_name in species:
            h_idx = species.index(h_name)
            visible = np.delete(data, h_idx, axis=1).astype(np.float32)
            hidden = data[:, h_idx].astype(np.float32)

            sp_dir = exp_dir / h_name
            sp_dir.mkdir(parents=True, exist_ok=True)
            sp_vals = []

            for seed in SEEDS:
                seed_dir = sp_dir / f"seed_{seed:05d}"
                mf = seed_dir / "metrics.json"
                if mf.exists():
                    with open(mf) as f:
                        m = json.load(f)
                    sp_vals.append(m['pearson_val'])
                    print(f"  C{exp_id}/{h_name} seed={seed} CACHED val={m['pearson_val']:+.3f}")
                    continue

                t0 = datetime.now()
                r_all, r_val, h_mean, h_sc = train_one(visible, hidden, seed, device)
                dt = (datetime.now() - t0).total_seconds()
                sp_vals.append(r_val)

                seed_dir.mkdir(parents=True, exist_ok=True)
                with open(mf, 'w') as f:
                    json.dump({'seed': seed, 'exp': f'C{exp_id}', 'species': h_name,
                               'pearson_all': r_all, 'pearson_val': r_val}, f, indent=2)
                np.savez(seed_dir / "trajectory.npz", h_mean=h_mean, h_scaled=h_sc)
                print(f"  C{exp_id}/{h_name} seed={seed}  val={r_val:+.3f}  ({dt:.1f}s)")

            mean_val = np.mean(sp_vals)
            exp_results.append({'species': h_name, 'mean_val': mean_val, 'n': len(sp_vals)})

        # Per-experiment summary
        with open(exp_dir / "summary.md", 'w', encoding='utf-8') as f:
            f.write(f"# Blasius C{exp_id}: Eco-GNRD\n\n")
            f.write("| Species | P(val) |\n|---|---|\n")
            for r in exp_results:
                f.write(f"| {r['species']} | {r['mean_val']:+.3f} |\n")
            ov = np.mean([r['mean_val'] for r in exp_results])
            f.write(f"\n**Overall**: P(val)={ov:+.3f}\n")
        all_exp_results.append({'exp': f'C{exp_id}', 'results': exp_results,
                                'overall': np.mean([r['mean_val'] for r in exp_results])})

    # Grand summary
    print(f"\n{'='*60}")
    print(f"{'Experiment':<12} {'algae':>10} {'rotifers':>10} {'eggs':>10} {'Overall':>10}")
    print('-' * 60)
    for er in all_exp_results:
        vals = {r['species']: r['mean_val'] for r in er['results']}
        print(f"C{er['exp'][1:]:<11} {vals.get('algae',0):>+10.3f} {vals.get('rotifers',0):>+10.3f} {vals.get('eggs',0):>+10.3f} {er['overall']:>+10.3f}")
    grand = np.mean([er['overall'] for er in all_exp_results])
    print('-' * 60)
    print(f"{'Grand mean':<12} {'':>10} {'':>10} {'':>10} {grand:>+10.3f}")

    with open(out_root / "summary.md", 'w', encoding='utf-8') as f:
        f.write("# Blasius 2020: Eco-GNRD on all experiments\n\n")
        f.write("| Experiment | algae | rotifers | eggs | Overall |\n|---|---|---|---|---|\n")
        for er in all_exp_results:
            vals = {r['species']: r['mean_val'] for r in er['results']}
            f.write(f"| {er['exp']} | {vals.get('algae',0):+.3f} | {vals.get('rotifers',0):+.3f} | {vals.get('eggs',0):+.3f} | {er['overall']:+.3f} |\n")
        f.write(f"\n**Grand mean**: P(val)={grand:+.3f}\n")

    print(f"\n[OK] Results in {out_root}")


if __name__ == "__main__":
    main()

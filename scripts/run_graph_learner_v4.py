"""GraphLearner v3: dense init + mild L1 (0.001).

v1: random init + L1=0.01 -> too sparse (A=0.04)
v2: dense init + no L1    -> no learning (A=0.95)
v3: dense init + L1=0.001 -> moderate pruning (target A=0.3-0.7)
"""
import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual
from scripts.run_main_experiment import DATASET_CONFIGS, load_dataset, SEEDS
from scripts.cvhi_beninca_nbeddyn import (
    LatentDynamicsNet, get_param_groups, freeze, unfreeze, alpha_schedule,
)

import sys
sys.stdout = open(1, 'w', encoding='utf-8', errors='replace', closefd=False)

SEEDS_TEST = SEEDS[:5]
LAM_SPARSE_GL = 0.001  # applied via log(A) bias on attention scores  # v1=0.01 (too much), v2=0 (too little), v3=0.001


def make_model_gl(N, cfg, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4, takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=cfg.get('use_formula_hints', True),
        use_G_field=True, use_graph_learner=True,
        num_mixture_components=1, G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def train_one(visible, hidden, seed, device, cfg, n_recon_ch=None):
    torch.manual_seed(seed)
    T, N = visible.shape
    epochs = cfg['epochs']
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model_gl(N, cfg, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=cfg['lr'], weight_decay=1e-4)

    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    pa = cfg.get('pa', 5); pb = cfg.get('pb', 1)
    use_alt = cfg.get('use_alt', False)

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float('inf'); best_state = None

    for epoch in range(epochs):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup: h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup; h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))

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
            lam_hf=cfg['lam_hf'], lowpass_sigma=6.0, lam_rmse_log=0.1,
            n_recon_channels=n_recon_ch)

        gl_sp = torch.tensor(0.0, device=device)
        for layer in model.f_visible.layers:
            if hasattr(layer, 'graph_learner'):
                gl_sp = gl_sp + layer.graph_learner.l1_sparsity()

        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and cfg['lam_hdyn'] > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = F.mse_loss(hp, hm[:, 1:].detach())

        (losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode + LAM_SPARSE_GL * gl_sp).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, h_weight=1.0,
                margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
                lam_energy=2.0, min_energy=cfg['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=cfg['lam_hf'], lowpass_sigma=6.0, n_recon_channels=n_recon_ch)
            vr = vl['recon_full'].item()

        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state: model.load_state_dict(best_state)
    model.eval()

    A_matrices = []
    for layer in model.f_visible.layers:
        A = layer.get_interaction_matrix()
        if A is not None: A_matrices.append(A)
    A_avg = np.mean(A_matrices, axis=0) if A_matrices else None

    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()

    L = min(len(h_mean), len(hidden)); te = train_end
    X_tr = np.column_stack([h_mean[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:te], rcond=None)
    X_val = np.column_stack([h_mean[te:L], np.ones(L - te)])
    h_sc = X_val @ coef
    r_val = float(np.corrcoef(h_sc, hidden[te:L])[0, 1]) if L > te + 2 else 0

    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return r_val, A_avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path("runs/graph_learner_v4")
    out_dir.mkdir(parents=True, exist_ok=True)

    K = np.array([[0.26,0.34,0.30,0.24,0.23,0.41],[0.22,0.39,0.34,0.30,0.27,0.16],
        [0.27,0.22,0.39,0.34,0.30,0.07],[0.26,0.24,0.22,0.39,0.34,0.28],
        [0.34,0.30,0.22,0.20,0.39,0.40]])
    C = np.array([[0.04,0.04,0.07,0.04,0.04,0.22],[0.08,0.08,0.08,0.10,0.08,0.14],
        [0.10,0.10,0.10,0.10,0.14,0.22],[0.05,0.03,0.03,0.03,0.03,0.09],
        [0.07,0.09,0.07,0.07,0.07,0.05]])
    alpha = np.zeros((6,6))
    for i in range(6):
        for j in range(6):
            for k in range(5): alpha[i,j] += (1.0/K[k,i])*C[k,j]
    SP = ['sp1','sp2','sp3','sp4','sp5','sp6']

    for ds_name in ['huisman', 'beninca', 'maizuru']:
        cfg = DATASET_CONFIGS[ds_name]
        tasks = load_dataset(ds_name)
        print(f"\n{'='*60}")
        print(f"GraphLearner v3 (dense init + L1={LAM_SPARSE_GL}) on {ds_name}")
        print(f"{'='*60}")

        results = []; total = len(tasks) * len(SEEDS_TEST); ri = 0
        for vis, hid, sp_name, n_rc in tasks:
            sp_vals = []; sp_As = []
            for seed in SEEDS_TEST:
                ri += 1; t0 = datetime.now()
                r_val, A = train_one(vis, hid, seed, device, cfg, n_recon_ch=n_rc)
                dt = (datetime.now() - t0).total_seconds()
                sp_vals.append(r_val)
                if A is not None: sp_As.append(A)
                print(f"  [{ri}/{total}] {sp_name} seed={seed}  val={r_val:+.3f}  ({dt:.1f}s)")

            mean_val = np.mean(sp_vals)
            A_info = ""
            if sp_As:
                A_mean = np.mean(sp_As, axis=0)
                A_info = f" A={A_mean.mean():.3f}"
                if ds_name == 'huisman':
                    h_idx = SP.index(sp_name)
                    vis_idx = [i for i in range(6) if i != h_idx]
                    sr, _ = spearmanr(A_mean[:5,:5].flatten(), alpha[np.ix_(vis_idx,vis_idx)].flatten())
                    A_info += f" Spearman={sr:+.3f}"
                    np.save(out_dir / f"A_{sp_name}.npy", A_mean)
            sys.stdout.flush()

            results.append({'species': sp_name, 'mean_val': mean_val})
            print(f"  -> mean={mean_val:+.3f}{A_info}")

        ov = np.mean([r['mean_val'] for r in results])
        print(f"\n{ds_name} Overall: {ov:+.3f}")

        with open(out_dir / f"{ds_name}_summary.md", 'w', encoding='utf-8') as f:
            f.write(f"# {ds_name}: GraphLearner v3 (L1={LAM_SPARSE_GL})\n\n")
            f.write("| Species | GL_v3 |\n|---|---|\n")
            for r in results:
                f.write(f"| {r['species']} | {r['mean_val']:+.3f} |\n")
            f.write(f"\n**Overall**: {ov:+.3f}\n")

    print("\nDONE")


if __name__ == "__main__":
    main()

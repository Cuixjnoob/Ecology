"""Test: Eco-GNRD with explicit interaction matrix on Huisman.

Compare performance and learned W matrix against true competition structure.
"""
import sys, io

# Save original stdout before imports clobber it
_original_stdout = sys.stdout

import numpy as np
import torch
import json
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.cvhi_residual import CVHI_Residual
from scripts.run_main_experiment import DATASET_CONFIGS, load_dataset, SEEDS
from scripts.run_main_experiment import train_one
from scripts.verify_interactions import compute_effective_competition, SP_NAMES

# Rebuild stdout from fd after imports clobber it
import os
sys.stdout = open(1, 'w', encoding='utf-8', errors='replace', closefd=False)

SEEDS_TEST = [42, 123, 456, 789, 2024]


def make_model_with_W(N, cfg, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        use_interaction_matrix=True,  # NEW
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def train_and_extract(visible, hidden, seed, device, cfg):
    """Train model with interaction matrix, return results + W matrix."""
    torch.manual_seed(seed)
    T, N = visible.shape
    epochs = cfg['epochs']
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model_with_W(N, cfg, device)

    from scripts.cvhi_beninca_nbeddyn import LatentDynamicsNet, get_param_groups, freeze, unfreeze, alpha_schedule
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=cfg['lr'], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup:
            h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))

        unfreeze(fvis_params); unfreeze(enc_params)
        for p in f_h.parameters(): p.requires_grad_(True)

        model.train(); f_h.train(); opt.zero_grad()
        out = model(x_full, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]; tr['G'] = out['G'][:, :train_end]
        losses = model.loss(tr, beta_kl=cfg['beta_kl'], free_bits=0.02,
            margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
            lam_necessary=cfg['lam_necessary'], lam_shuffle=cfg['lam_necessary'] * 0.6,
            lam_energy=2.0, min_energy=cfg['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0, lam_rmse_log=0.1)

        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and cfg['lam_hdyn'] > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = torch.nn.functional.mse_loss(hp, hm[:, 1:].detach())
        (losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, h_weight=1.0,
                margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
                lam_energy=2.0, min_energy=cfg['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25))
            vr = vl['recon_full'].item()

        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Extract W matrix from f_visible layers
    W_matrices = []
    for layer in model.f_visible.layers:
        W = layer.get_interaction_matrix()
        if W is not None:
            W_matrices.append(W)

    # Get h prediction
    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()

    L = min(len(h_mean), len(hidden))
    te = train_end
    X_tr = np.column_stack([h_mean[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:te], rcond=None)
    X_val = np.column_stack([h_mean[te:L], np.ones(L - te)])
    h_sc = X_val @ coef
    r_val = float(np.corrcoef(h_sc, hidden[te:L])[0, 1]) if L > te + 2 else 0

    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()

    # Average W across layers
    W_avg = np.mean(W_matrices, axis=0) if W_matrices else None
    return r_val, W_avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DATASET_CONFIGS['huisman']
    tasks = load_dataset('huisman')

    out_dir = Path("重要实验/results/interaction_matrix_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    true_alpha = compute_effective_competition()  # 6×6

    print("Eco-GNRD with explicit interaction matrix W on Huisman")
    print(f"Seeds: {SEEDS_TEST}\n")

    all_results = []
    all_W = []

    for vis, hid, sp_name, n_rc in tasks:
        h_idx = SP_NAMES.index(sp_name)
        # True competition: hidden species row/col removed
        vis_idx = [i for i in range(6) if i != h_idx]

        print(f"\n--- Hidden={sp_name} ---")
        sp_vals = []
        sp_Ws = []
        for seed in SEEDS_TEST:
            t0 = datetime.now()
            r_val, W = train_and_extract(vis, hid, seed, device, cfg)
            dt = (datetime.now() - t0).total_seconds()
            sp_vals.append(r_val)
            if W is not None:
                sp_Ws.append(W)
            print(f"  seed={seed}  val={r_val:+.3f}  ({dt:.1f}s)")

        mean_val = np.mean(sp_vals)
        all_results.append({'species': sp_name, 'mean_val': mean_val})

        # Average W across seeds, compare with true
        if sp_Ws:
            W_mean = np.mean(sp_Ws, axis=0)
            all_W.append((sp_name, W_mean, h_idx))

            # Compare visible×visible block with true alpha (visible×visible)
            # W is N_visible × N_visible where N_visible = 10 (5 species + 5 resources)
            # True alpha is for the 5 visible species only
            W_sp = W_mean[:5, :5]  # species×species block
            true_sp = true_alpha[np.ix_(vis_idx, vis_idx)]

            # Flatten and compare
            w_flat = W_sp.flatten()
            t_flat = true_sp.flatten()
            sr, sp = spearmanr(np.abs(w_flat), t_flat)
            print(f"  Mean P(val)={mean_val:+.3f}")
            print(f"  W vs True alpha (species block, Spearman): {sr:+.3f} (p={sp:.3f})")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY: Eco-GNRD + Interaction Matrix on Huisman")
    print(f"{'='*60}")
    # Compare with baseline (no interaction matrix)
    baseline_vals = {'sp1': 0.313, 'sp2': 0.640, 'sp3': 0.450,
                     'sp4': 0.591, 'sp5': 0.431, 'sp6': 0.099}
    print(f"{'Species':<10} {'With W':>10} {'Without W':>12} {'Diff':>8}")
    print("-" * 42)
    for r in all_results:
        bv = baseline_vals.get(r['species'], 0)
        diff = r['mean_val'] - bv
        print(f"{r['species']:<10} {r['mean_val']:>+10.3f} {bv:>+12.3f} {diff:>+8.3f}")
    ov_w = np.mean([r['mean_val'] for r in all_results])
    ov_b = np.mean(list(baseline_vals.values()))
    print("-" * 42)
    print(f"{'Overall':<10} {ov_w:>+10.3f} {ov_b:>+12.3f} {ov_w-ov_b:>+8.3f}")

    print(f"\n[OK] Results in {out_dir}")


if __name__ == "__main__":
    main()

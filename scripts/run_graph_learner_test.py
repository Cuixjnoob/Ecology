"""Test: Eco-GNRD with GraphLearner on Huisman.

GraphLearner learns a static adjacency A_ij from species embeddings.
Compare: performance + learned A vs true competition structure.
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
from scripts.verify_interactions import compute_effective_competition, SP_NAMES

# Rebuild stdout
import sys, os
sys.stdout = open(1, 'w', encoding='utf-8', errors='replace', closefd=False)

SEEDS_TEST = [42, 123, 456, 789, 2024]


def make_model_gl(N, cfg, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        use_graph_learner=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def train_and_extract(visible, hidden, seed, device, cfg):
    torch.manual_seed(seed)
    T, N = visible.shape
    epochs = cfg['epochs']
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model_gl(N, cfg, device)
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
        for p in f_h.parameters():
            p.requires_grad_(True)

        model.train(); f_h.train(); opt.zero_grad()
        out = model(x_full, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]
        tr['G'] = out['G'][:, :train_end]
        losses = model.loss(tr, beta_kl=cfg['beta_kl'], free_bits=0.02,
            margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
            lam_necessary=cfg['lam_necessary'], lam_shuffle=cfg['lam_necessary'] * 0.6,
            lam_energy=2.0, min_energy=cfg['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0, lam_rmse_log=0.1)

        # GraphLearner sparsity penalty
        gl_sparsity = torch.tensor(0.0, device=device)
        for layer in model.f_visible.layers:
            if hasattr(layer, 'graph_learner'):
                gl_sparsity = gl_sparsity + layer.graph_learner.l1_sparsity()

        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and cfg['lam_hdyn'] > 0:
            hm = out['mu'][:, :train_end]
            xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = F.mse_loss(hp, hm[:, 1:].detach())

        total = losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode + 0.01 * gl_sparsity
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]
            vo['G'] = out['G'][:, train_end:T]
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

    # Extract learned adjacency
    A_matrices = []
    for layer in model.f_visible.layers:
        A = layer.get_interaction_matrix()
        if A is not None:
            A_matrices.append(A)
    A_avg = np.mean(A_matrices, axis=0) if A_matrices else None

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
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return r_val, A_avg


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cfg = DATASET_CONFIGS['huisman']
    tasks = load_dataset('huisman')

    out_dir = Path("重要实验/results/graph_learner_test")
    out_dir.mkdir(parents=True, exist_ok=True)

    true_alpha = compute_effective_competition()

    print("Eco-GNRD + GraphLearner on Huisman")
    print(f"Seeds: {SEEDS_TEST}")
    print()

    all_results = []

    for vis, hid, sp_name, n_rc in tasks:
        h_idx = SP_NAMES.index(sp_name)
        vis_idx = [i for i in range(6) if i != h_idx]

        print(f"--- Hidden={sp_name} ---")
        sp_vals = []
        sp_As = []
        for seed in SEEDS_TEST:
            t0 = datetime.now()
            r_val, A = train_and_extract(vis, hid, seed, device, cfg)
            dt = (datetime.now() - t0).total_seconds()
            sp_vals.append(r_val)
            if A is not None:
                sp_As.append(A)
            print(f"  seed={seed}  val={r_val:+.3f}  ({dt:.1f}s)")

        mean_val = np.mean(sp_vals)

        # Compare A with true alpha
        if sp_As:
            A_mean = np.mean(sp_As, axis=0)
            A_sp = A_mean[:5, :5]
            true_sp = true_alpha[np.ix_(vis_idx, vis_idx)]
            sr, sp = spearmanr(A_sp.flatten(), true_sp.flatten())
            print(f"  Mean P(val)={mean_val:+.3f}")
            print(f"  A vs True (species block, Spearman): {sr:+.3f} (p={sp:.3f})")
            print(f"  A sparsity: {A_mean.mean():.3f} (0=sparse, 1=dense)")

            # Save A
            np.save(out_dir / f"A_{sp_name}.npy", A_mean)

        all_results.append({'species': sp_name, 'mean_val': mean_val})

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    baseline_vals = {'sp1': 0.313, 'sp2': 0.640, 'sp3': 0.450,
                     'sp4': 0.591, 'sp5': 0.431, 'sp6': 0.099}
    print(f"{'Species':<10} {'GraphLrn':>10} {'Baseline':>10} {'Diff':>8}")
    print("-" * 40)
    for r in all_results:
        bv = baseline_vals.get(r['species'], 0)
        diff = r['mean_val'] - bv
        print(f"{r['species']:<10} {r['mean_val']:>+10.3f} {bv:>+10.3f} {diff:>+8.3f}")
    ov = np.mean([r['mean_val'] for r in all_results])
    ob = np.mean(list(baseline_vals.values()))
    print("-" * 40)
    print(f"{'Overall':<10} {ov:>+10.3f} {ob:>+10.3f} {ov-ob:>+8.3f}")


if __name__ == "__main__":
    main()

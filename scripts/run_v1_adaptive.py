"""V1 with row-wise adaptive sparsity: λ_i based on visible-only concentration.

Step 1: Train baseline, extract attention, compute per-receiver concentration C_i
Step 2: Set λ_i = λ_base * exp(κ * (C_i - mean_C))
Step 3: Train v1 with these per-row λ_i
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

cfg = DATASET_CONFIGS['huisman']
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tasks = load_dataset('huisman')
out_dir = Path('runs/v1_adaptive')
out_dir.mkdir(parents=True, exist_ok=True)

LAM_GLOBAL = 0.1  # overall sparsity strength
LAM_MIN = 0.005   # minimum per-row lambda
KAPPA = 2.0


def train_baseline_and_get_concentration(vis, seed):
    """Train baseline (no GL), extract attention concentration per receiver."""
    torch.manual_seed(seed)
    T, N = vis.shape
    x_full = torch.tensor(vis.astype(np.float32), device=device).unsqueeze(0)
    model = CVHI_Residual(
        num_visible=N, encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4, takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3, prior_std=1.0,
        gnn_backbone='mlp', use_formula_hints=True, use_G_field=True,
        use_graph_learner=False,  # baseline
        num_mixture_components=1, G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=cfg['lr'], weight_decay=1e-4)
    warmup = 100; train_end = int(0.75 * T)
    best_state = None; best_val = float('inf')

    def lr_lambda(step):
        if step < 50: return step / 50
        return 0.5 * (1 + np.cos(np.pi * (step - 50) / max(1, 300 - 50)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for epoch in range(300):  # shorter baseline
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_schedule(epoch, 300)
        h_w = 0.0 if epoch < warmup else min(1.0, (epoch - warmup) / 100)
        K_r = 0 if epoch < warmup else max(1, int(min(1.0, (epoch - warmup) / 200 * 2) * 3))
        model.train(); f_h.train(); opt.zero_grad()
        out = model(x_full, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]; tr['G'] = out['G'][:, :train_end]
        losses = model.loss(tr, beta_kl=cfg['beta_kl'], free_bits=0.02,
            margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
            lam_necessary=cfg['lam_necessary'], lam_shuffle=cfg['lam_necessary'] * 0.6,
            lam_energy=2.0, min_energy=cfg['min_energy'], lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w, rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lam_rmse_log=0.1, n_recon_channels=5)
        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            loss_ode = F.mse_loss(f_h(hm[:, :-1], xv[:, :-1]), hm[:, 1:].detach())
        (losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vr = model.loss(vo, h_weight=1.0, margin_null=cfg['margin_null'],
                margin_shuf=cfg['margin_shuf'], lam_energy=2.0, min_energy=cfg['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25))['recon_full'].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Extract attention and compute concentration
    with torch.no_grad():
        state = x_full[0].unsqueeze(0)
        attns = []
        for layer in model.f_visible.layers:
            _, attn = layer(state)
            attns.append(attn[0].cpu().numpy())  # (T, N, N)
        attn_avg = np.mean([a.mean(axis=0) for a in attns], axis=0)  # (N, N)

    # Concentration: sum of squared attention weights per row (Herfindahl index)
    C = np.sum(attn_avg ** 2, axis=1)  # (N,)
    del model, f_h
    torch.cuda.empty_cache()
    return C


def train_v1_adaptive(vis, hid, seed, lam_row):
    """Train v1 with per-row λ_i."""
    torch.manual_seed(seed)
    T, N = vis.shape
    epochs = cfg['epochs']
    x_full = torch.tensor(vis.astype(np.float32), device=device).unsqueeze(0)
    lam_row_t = torch.tensor(lam_row, dtype=torch.float32, device=device)

    model = CVHI_Residual(
        num_visible=N, encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4, takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3, prior_std=1.0,
        gnn_backbone='mlp', use_formula_hints=True, use_G_field=True,
        use_graph_learner=True,  # v1
        num_mixture_components=1, G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=cfg['lr'], weight_decay=1e-4)
    warmup = 100; train_end = int(0.75 * T)
    best_val = float('inf'); best_state = None

    def lr_lambda(step):
        if step < 50: return step / 50
        return 0.5 * (1 + np.cos(np.pi * (step - 50) / max(1, epochs - 50)))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    for epoch in range(epochs):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        h_w = 0.0 if epoch < warmup else min(1.0, (epoch - warmup) / 100)
        K_r = 0 if epoch < warmup else max(1, int(min(1.0, (epoch - warmup) / 200 * 2) * 3))
        model.train(); f_h.train(); opt.zero_grad()
        out = model(x_full, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]; tr['G'] = out['G'][:, :train_end]
        losses = model.loss(tr, beta_kl=cfg['beta_kl'], free_bits=0.02,
            margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
            lam_necessary=cfg['lam_necessary'], lam_shuffle=cfg['lam_necessary'] * 0.6,
            lam_energy=2.0, min_energy=cfg['min_energy'], lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w, rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lam_rmse_log=0.1, n_recon_channels=5)

        # Row-wise sparsity: Σ_i λ_i * mean_j(A_ij)
        gl_reg = torch.tensor(0.0, device=device)
        for layer in model.f_visible.layers:
            if hasattr(layer, 'graph_learner'):
                A = layer.graph_learner.forward()
                row_means = A.mean(dim=1)  # (N,)
                gl_reg = gl_reg + (lam_row_t * row_means).mean()
        lam_warmup = min(1.0, max(0, (epoch - warmup) / 200.0)) if epoch > warmup else 0.0

        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and cfg['lam_hdyn'] > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            loss_ode = F.mse_loss(f_h(hm[:, :-1], xv[:, :-1]), hm[:, 1:].detach())

        (losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode + lam_warmup * gl_reg).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vr = model.loss(vo, h_weight=1.0, margin_null=cfg['margin_null'],
                margin_shuf=cfg['margin_shuf'], lam_energy=2.0, min_energy=cfg['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25))['recon_full'].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Extract A and h
    A_matrices = []
    for layer in model.f_visible.layers:
        A = layer.get_interaction_matrix()
        if A is not None:
            A_matrices.append(A)
    A_avg = np.mean(A_matrices, axis=0) if A_matrices else None

    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()

    L = min(len(h_mean), len(hid)); te = train_end
    try:
        X_tr = np.column_stack([h_mean[:te], np.ones(te)])
        coef, _, _, _ = np.linalg.lstsq(X_tr, hid[:te], rcond=None)
        X_val = np.column_stack([h_mean[te:L], np.ones(L - te)])
        h_sc = X_val @ coef
        r_val = float(np.corrcoef(h_sc, hid[te:L])[0, 1]) if L > te + 2 else 0
    except:
        r_val = 0.0

    del model, f_h
    torch.cuda.empty_cache()
    return r_val, A_avg


# Main
SP = ['sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6']

for sp_idx in [1, 5]:  # sp2, sp6 first
    vis, hid, sp_name, n_rc = tasks[sp_idx]

    # Step 1: Get concentration from baseline
    print(f'\n--- {sp_name}: computing baseline concentration ---')
    C = train_baseline_and_get_concentration(vis, seed=42)
    C_norm = (C - C.mean()) / (C.std() + 1e-8)
    w = np.exp(KAPPA * C_norm)
    w = w / w.mean()  # normalize so mean(w)=1
    lam_row = np.maximum(LAM_GLOBAL * w, LAM_MIN)
    print(f'  C[:5] = {C[:5].round(4)}')
    print(f'  lam_row[:5] = {lam_row[:5].round(4)}')
    sys.stdout.flush()

    # Step 2: Train v1 with adaptive λ
    for seed in SEEDS[:5]:
        t0 = datetime.now()
        r_val, A = train_v1_adaptive(vis, hid, seed, lam_row)
        dt = (datetime.now() - t0).total_seconds()

        sd = out_dir / 'huisman' / sp_name / f'seed_{seed:05d}'
        sd.mkdir(parents=True, exist_ok=True)
        with open(sd / 'metrics.json', 'w') as f:
            json.dump({'species': sp_name, 'seed': seed, 'pearson_val': r_val,
                       'lam_row': lam_row[:5].tolist()}, f)

        A_info = f' A_mean={A.mean():.3f}' if A is not None else ''
        print(f'  {sp_name} seed={seed} val={r_val:+.3f} ({dt:.1f}s){A_info}')
        sys.stdout.flush()

print('\nDONE')

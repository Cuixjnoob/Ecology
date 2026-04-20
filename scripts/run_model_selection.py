"""Unsupervised model selection: train baseline + v1, select by visible-only scores.

For each hidden species, train both models, save visible metrics, select winner.
"""
import numpy as np
import torch
import torch.nn.functional as F
import json
from pathlib import Path
from datetime import datetime

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
out_dir = Path('runs/model_selection')
out_dir.mkdir(parents=True, exist_ok=True)

SP = ['sp1', 'sp2', 'sp3', 'sp4', 'sp5', 'sp6']


def train_model(vis, hid, seed, use_gl, epochs=500):
    """Train one model. Returns pearson_val + visible-only metrics."""
    torch.manual_seed(seed)
    T, N = vis.shape
    x_full = torch.tensor(vis.astype(np.float32), device=device).unsqueeze(0)

    model = CVHI_Residual(
        num_visible=N, encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4, takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3, prior_std=1.0,
        gnn_backbone='mlp', use_formula_hints=True, use_G_field=True,
        use_graph_learner=use_gl,
        num_mixture_components=1, G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
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

        # GL sparsity (only for v1)
        gl_sp = torch.tensor(0.0, device=device)
        if use_gl:
            for layer in model.f_visible.layers:
                if hasattr(layer, 'graph_learner'):
                    gl_sp = gl_sp + layer.graph_learner.l1_sparsity()
            lam_gl = 0.01 * min(1.0, max(0, (epoch - warmup) / 200.0)) if epoch > warmup else 0.0
        else:
            lam_gl = 0.0

        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and cfg['lam_hdyn'] > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            loss_ode = F.mse_loss(f_h(hm[:, :-1], xv[:, :-1]), hm[:, 1:].detach())

        (losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode + lam_gl * gl_sp).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=cfg['margin_null'],
                margin_shuf=cfg['margin_shuf'], lam_energy=2.0, min_energy=cfg['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25))
            vr = vl['recon_full'].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    # Compute visible-only metrics
    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()

        # val recon (visible loss on val)
        vo = model.slice_out(oe, train_end, T)
        vo['visible'] = oe['visible'][:, train_end:T]; vo['G'] = oe['G'][:, train_end:T]
        vl = model.loss(vo, h_weight=1.0, margin_null=cfg['margin_null'],
            margin_shuf=cfg['margin_shuf'], lam_energy=2.0, min_energy=cfg['min_energy'],
            lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25))
        val_recon = vl['recon_full'].item()

        # Necessity: how much does removing h hurt visible prediction
        vis_safe = torch.clamp(oe['visible'], min=1e-6)
        log_ratio = torch.log(vis_safe[:, 1:] / vis_safe[:, :-1])
        T_use = min(oe['pred_full'].shape[2], log_ratio.shape[1])
        recon_full = ((oe['pred_full'].mean(dim=0)[:, :T_use] - log_ratio[:, :T_use]) ** 2).mean().item()
        T_null = min(oe['pred_null'].shape[1], log_ratio.shape[1])
        recon_null = ((oe['pred_null'][:, :T_null] - log_ratio[:, :T_null]) ** 2).mean().item()
        necessity = recon_null - recon_full  # positive = h is necessary

    # Pearson (for evaluation, not selection)
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
    return {
        'pearson_val': r_val,
        'val_recon': val_recon,
        'necessity': necessity,
        'best_val_loss': best_val,
    }


# Run for all species
print('=== Unsupervised Model Selection: Baseline vs V1 ===\n')

for sp_idx in range(6):
    vis, hid, sp_name, n_rc = tasks[sp_idx]
    print(f'--- {sp_name} ---')

    bl_results = []
    v1_results = []

    for seed in SEEDS[:3]:  # 3 seeds for speed
        # Baseline
        t0 = datetime.now()
        bl = train_model(vis, hid, seed, use_gl=False)
        dt_bl = (datetime.now() - t0).total_seconds()
        bl_results.append(bl)
        print(f'  BL seed={seed}: P={bl["pearson_val"]:+.3f} recon={bl["val_recon"]:.4f} nec={bl["necessity"]:+.4f} ({dt_bl:.0f}s)')

        # V1
        t0 = datetime.now()
        v1 = train_model(vis, hid, seed, use_gl=True)
        dt_v1 = (datetime.now() - t0).total_seconds()
        v1_results.append(v1)
        print(f'  V1 seed={seed}: P={v1["pearson_val"]:+.3f} recon={v1["val_recon"]:.4f} nec={v1["necessity"]:+.4f} ({dt_v1:.0f}s)')
        sys.stdout.flush()

    # Aggregate
    bl_p = np.mean([r['pearson_val'] for r in bl_results])
    v1_p = np.mean([r['pearson_val'] for r in v1_results])
    bl_recon = np.mean([r['val_recon'] for r in bl_results])
    v1_recon = np.mean([r['val_recon'] for r in v1_results])
    bl_nec = np.mean([r['necessity'] for r in bl_results])
    v1_nec = np.mean([r['necessity'] for r in v1_results])
    bl_std = np.std([r['pearson_val'] for r in bl_results])
    v1_std = np.std([r['pearson_val'] for r in v1_results])

    # Unsupervised score: lower recon + higher necessity + lower std
    bl_score = -bl_recon + bl_nec - 0.5 * bl_std
    v1_score = -v1_recon + v1_nec - 0.5 * v1_std

    selected = 'v1' if v1_score > bl_score else 'baseline'
    selected_p = v1_p if selected == 'v1' else bl_p

    print(f'  Summary:')
    print(f'    BL: P={bl_p:+.3f} recon={bl_recon:.4f} nec={bl_nec:+.4f} std={bl_std:.3f} score={bl_score:+.4f}')
    print(f'    V1: P={v1_p:+.3f} recon={v1_recon:.4f} nec={v1_nec:+.4f} std={v1_std:.3f} score={v1_score:+.4f}')
    print(f'    -> SELECT: {selected} (P={selected_p:+.3f})')

    # Save
    sd = out_dir / 'huisman' / sp_name
    sd.mkdir(parents=True, exist_ok=True)
    with open(sd / 'selection.json', 'w') as f:
        json.dump({
            'species': sp_name,
            'selected': selected,
            'bl_pearson': bl_p, 'v1_pearson': v1_p,
            'bl_recon': bl_recon, 'v1_recon': v1_recon,
            'bl_necessity': bl_nec, 'v1_necessity': v1_nec,
            'bl_score': bl_score, 'v1_score': v1_score,
            'selected_pearson': selected_p,
        }, f, indent=2)
    print()
    sys.stdout.flush()

# Final summary
print('='*60)
print('FINAL: Selected model per species')
print('='*60)
all_selected_p = []
for sp in SP:
    sf = out_dir / 'huisman' / sp / 'selection.json'
    if sf.exists():
        d = json.load(open(sf))
        print(f'{sp}: {d["selected"]:>8s}  P={d["selected_pearson"]:+.3f}  (BL={d["bl_pearson"]:+.3f}, V1={d["v1_pearson"]:+.3f})')
        all_selected_p.append(d['selected_pearson'])
if all_selected_p:
    print(f'\nOverall selected: {np.mean(all_selected_p):+.3f}')

print('\nDONE')

"""Gate finetune: initialize gate to calibrated values, then joint train with anchor."""
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
out_dir = Path('runs/gate_finetune')
out_dir.mkdir(parents=True, exist_ok=True)

# Anchor values from gate calibration
TAU_ANCHOR = [1.242, 3.731, 3.837, 3.567, 3.316, 2.227, 2.227, 2.227, 2.227, 2.227]
S_ANCHOR = [0.994, 0.453, 0.481, 0.462, 0.758, 0.887, 0.887, 0.887, 0.887, 0.887]

for sp_idx in [1, 5]:  # sp2, sp6
    vis, hid, sp_name, n_rc = tasks[sp_idx]
    for seed in SEEDS[:5]:
        torch.manual_seed(seed)
        T, N = vis.shape
        epochs = 500
        x_full = torch.tensor(vis.astype(np.float32), device=device).unsqueeze(0)

        model = CVHI_Residual(
            num_visible=N, encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
            encoder_heads=4, takens_lags=(1,2,4,8), encoder_dropout=0.1,
            d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=12, G_field_layers=1, G_field_top_k=3, prior_std=1.0,
            gnn_backbone='mlp', use_formula_hints=True, use_G_field=True,
            use_graph_learner=True, num_mixture_components=1,
            G_anchor_first=True, G_anchor_sign=+1,
        ).to(device)
        f_h = LatentDynamicsNet(N, d_hidden=32).to(device)

        tau_anchor = torch.tensor(TAU_ANCHOR, device=device)
        s_anchor = torch.tensor(S_ANCHOR, device=device)

        # Initialize gate to calibrated values
        with torch.no_grad():
            for layer in model.f_visible.layers:
                if hasattr(layer, '_gl_tau_raw'):
                    target = tau_anchor.clone()
                    raw = torch.log(torch.exp(target - layer._gl_tau_min) - 1 + 1e-6)
                    layer._gl_tau_raw.copy_(raw)
                if hasattr(layer, '_gl_scale_raw'):
                    target_s = s_anchor.clone().clamp(layer._gl_s_min + 0.01, 0.999)
                    norm = (target_s - layer._gl_s_min) / (1 - layer._gl_s_min)
                    raw = torch.log(norm / (1 - norm + 1e-6) + 1e-6)
                    layer._gl_scale_raw.copy_(raw)

        # Separate params
        gate_params = []
        backbone_params = []
        for name, p in model.named_parameters():
            if '_gl_tau' in name or '_gl_scale' in name or 'graph_learner' in name:
                gate_params.append(p)
            else:
                backbone_params.append(p)

        opt = torch.optim.AdamW([
            {'params': backbone_params, 'lr': cfg['lr'] * 0.15},
            {'params': gate_params, 'lr': 0.005},
            {'params': list(f_h.parameters()), 'lr': cfg['lr'] * 0.15},
        ], weight_decay=1e-4)

        warmup = 100
        train_end = int(0.75 * T)
        best_val = float('inf')
        best_state = None

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
            tr['visible'] = out['visible'][:, :train_end]
            tr['G'] = out['G'][:, :train_end]
            losses = model.loss(
                tr, beta_kl=cfg['beta_kl'], free_bits=0.02,
                margin_null=cfg['margin_null'], margin_shuf=cfg['margin_shuf'],
                lam_necessary=cfg['lam_necessary'], lam_shuffle=cfg['lam_necessary'] * 0.6,
                lam_energy=2.0, min_energy=cfg['min_energy'],
                lam_smooth=0.02, lam_sparse=0.02,
                h_weight=h_w, lam_rollout=0.5 * h_w,
                rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lam_rmse_log=0.1, n_recon_channels=5,
            )

            # Sparsity reg with warmup
            gl_reg = sum(
                l.graph_learner.l1_sparsity()
                for l in model.f_visible.layers if hasattr(l, 'graph_learner')
            )
            lam_gl = 0.01 * min(1.0, max(0, (epoch - warmup) / 200.0)) if epoch > warmup else 0.0

            # Anchor loss (decays over 50 epochs)
            anchor_w = max(0, 1.0 - epoch / 50.0) * 0.1
            anchor_loss = torch.tensor(0.0, device=device)
            for layer in model.f_visible.layers:
                if hasattr(layer, '_gl_tau_raw'):
                    tau_now = layer._gl_tau_min + F.softplus(layer._gl_tau_raw)
                    anchor_loss = anchor_loss + ((tau_now - tau_anchor) ** 2).mean()
                if hasattr(layer, '_gl_scale_raw'):
                    s_now = layer._gl_s_min + (1 - layer._gl_s_min) * torch.sigmoid(layer._gl_scale_raw)
                    anchor_loss = anchor_loss + ((s_now - s_anchor) ** 2).mean()

            # ODE
            loss_ode = torch.tensor(0.0, device=device)
            if h_w > 0 and cfg['lam_hdyn'] > 0:
                hm = out['mu'][:, :train_end]
                xv = out['visible'][:, :train_end]
                loss_ode = F.mse_loss(f_h(hm[:, :-1], xv[:, :-1]), hm[:, 1:].detach())

            total = losses['total'] + cfg['lam_hdyn'] * h_w * loss_ode + lam_gl * gl_reg + anchor_w * anchor_loss
            total.backward()
            torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(f_h.parameters()), 1.0)
            opt.step()
            sched.step()

            with torch.no_grad():
                vo = model.slice_out(out, train_end, T)
                vo['visible'] = out['visible'][:, train_end:T]
                vo['G'] = out['G'][:, train_end:T]
                vr = model.loss(
                    vo, h_weight=1.0, margin_null=cfg['margin_null'],
                    margin_shuf=cfg['margin_shuf'], lam_energy=2.0,
                    min_energy=cfg['min_energy'], lam_rollout=0.5,
                    rollout_weights=(1.0, 0.5, 0.25),
                )['recon_full'].item()
            if epoch > warmup + 15 and vr < best_val:
                best_val = vr
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if best_state:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            oe = model(x_full, n_samples=30, rollout_K=3)
            h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
            l = model.f_visible.layers[0]
            tau_f = (l._gl_tau_min + F.softplus(l._gl_tau_raw)).cpu().numpy()
            s_f = (l._gl_s_min + (1 - l._gl_s_min) * torch.sigmoid(l._gl_scale_raw)).cpu().numpy()

        L = min(len(h_mean), len(hid))
        te = train_end
        try:
            X_tr = np.column_stack([h_mean[:te], np.ones(te)])
            coef, _, _, _ = np.linalg.lstsq(X_tr, hid[:te], rcond=None)
            X_val = np.column_stack([h_mean[te:L], np.ones(L - te)])
            h_sc = X_val @ coef
            r_val = float(np.corrcoef(h_sc, hid[te:L])[0, 1]) if L > te + 2 else 0
        except:
            r_val = 0.0

        sd = out_dir / 'huisman' / sp_name / f'seed_{seed:05d}'
        sd.mkdir(parents=True, exist_ok=True)
        with open(sd / 'metrics.json', 'w') as f:
            json.dump({
                'species': sp_name, 'seed': seed, 'pearson_val': r_val,
                'tau_sp': tau_f[:5].tolist(), 's_sp': s_f[:5].tolist(),
            }, f)
        print(f'{sp_name} seed={seed} val={r_val:+.3f} tau0={tau_f[0]:.2f} s0={s_f[0]:.3f}')
        sys.stdout.flush()
        del model, f_h
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("DONE")

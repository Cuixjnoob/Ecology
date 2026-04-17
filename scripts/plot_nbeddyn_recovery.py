"""Retrain Ostracods + Bacteria with NbedDyn and plot recovery trajectories."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate
from scripts.cvhi_beninca_nbeddyn import (
    LatentDynamicsNet, make_model, get_param_groups,
    freeze, unfreeze, alpha_schedule, find_burst_mask,
    HP, PA, PB, LAM_H_ODE
)

full, species, days = load_beninca(include_nutrients=True)
species = [str(s) for s in species]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500


def run_one(h_name, seed):
    h_idx = species.index(h_name)
    visible = np.delete(full, h_idx, axis=1).astype(np.float32)
    hidden_raw = full[:, h_idx].astype(np.float32)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(seed)
    model = make_model(N, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=HP['lr'], weight_decay=1e-4)
    warmup = int(0.2 * EPOCHS)
    ramp = max(1, int(0.2 * EPOCHS))

    def lr_lambda(step):
        if step < 50:
            return step / 50
        p = (step - 50) / max(1, EPOCHS - 50)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float('inf')
    best_state = None
    best_fh = None

    for epoch in range(EPOCHS):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_schedule(epoch, EPOCHS)
        if epoch < warmup:
            h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, EPOCHS - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))

        if epoch >= warmup:
            cycle_pos = (epoch - warmup) % (PA + PB)
            if cycle_pos < PA:
                freeze(enc_params)
                for p in f_h.parameters():
                    p.requires_grad_(False)
                unfreeze(fvis_params)
            else:
                freeze(fvis_params)
                unfreeze(enc_params)
                for p in f_h.parameters():
                    p.requires_grad_(True)
        else:
            unfreeze(fvis_params)
            unfreeze(enc_params)
            for p in f_h.parameters():
                p.requires_grad_(True)

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train()
        f_h.train()
        opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out['visible'] = out['visible'][:, :train_end]
        tr_out['G'] = out['G'][:, :train_end]
        losses = model.loss(
            tr_out, beta_kl=HP['lam_kl'], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP['lam_cf'], lam_shuffle=HP['lam_cf'] * 0.6,
            lam_energy=2.0, min_energy=HP['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.2, lowpass_sigma=6.0, lam_rmse_log=0.1,
        )
        loss_h_ode = torch.tensor(0.0, device=device)
        if h_w > 0:
            h_mu = out['mu'][:, :train_end]
            x_vis = out['visible'][:, :train_end]
            h_pred = f_h(h_mu[:, :-1], x_vis[:, :-1])
            loss_h_ode = F.mse_loss(h_pred, h_mu[:, 1:].detach())
        total = losses['total'] + LAM_H_ODE * h_w * loss_h_ode
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_out['visible'] = out['visible'][:, train_end:T]
            val_out['G'] = out['G'][:, train_end:T]
            vl = model.loss(
                val_out, h_weight=1.0, margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=HP['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.2, lowpass_sigma=6.0,
            )
            vr = vl['recon_full'].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    unfreeze(fvis_params)
    unfreeze(enc_params)
    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval['h_samples'].mean(dim=0)[0].cpu().numpy()
    pear, h_scaled = evaluate(h_mean, hidden_raw)
    del model, f_h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return pear, h_scaled.flatten(), h_mean, hidden_raw, train_end


# --- Run ---
print('Running Ostracods seed=42...')
p_ost, rec_ost, h_ost, true_ost, te = run_one('Ostracods', 42)
print(f'  Pearson = {p_ost:.3f}')

print('Running Bacteria seed=42...')
p_bac, rec_bac, h_bac, true_bac, _ = run_one('Bacteria', 42)
print(f'  Pearson = {p_bac:.3f}')

# --- Plot ---
fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                         gridspec_kw={'height_ratios': [2, 1, 2, 1]})


def norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)


L = min(len(days), len(rec_ost), len(true_ost))
t = days[:L]

# Panel 1: Ostracods recovery
ax = axes[0]
ax.plot(t, norm(true_ost[:L]), 'k-', lw=1.2, alpha=0.8, label='True Ostracods')
ax.plot(t, norm(rec_ost[:L]), 'r-', lw=1.0, alpha=0.7,
        label=f'Recovered (r={p_ost:.3f})')
ax.axvline(days[te], color='gray', ls='--', alpha=0.5, label='Train/Val split')
burst_t = find_burst_mask(true_ost, pct=10)
for i in range(L):
    if burst_t[i]:
        ax.axvspan(t[i] - 2, t[i] + 2, color='blue', alpha=0.08)
ax.set_ylabel('Normalized')
ax.set_title(f'Ostracods: NbedDyn + Alt 5:1 (seed=42, Pearson={p_ost:.3f})')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Ostracods h(t)
ax = axes[1]
ax.plot(t[:len(h_ost)], h_ost[:L], 'g-', lw=0.8)
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.axvline(days[te], color='gray', ls='--', alpha=0.5)
ax.set_ylabel('h(t)')
ax.set_title('Ostracods h(t)')
ax.grid(True, alpha=0.3)

# Panel 3: Bacteria recovery
L2 = min(len(days), len(rec_bac), len(true_bac))
ax = axes[2]
ax.plot(t[:L2], norm(true_bac[:L2]), 'k-', lw=1.2, alpha=0.8,
        label='True Bacteria')
ax.plot(t[:L2], norm(rec_bac[:L2]), 'r-', lw=1.0, alpha=0.7,
        label=f'Recovered (r={p_bac:.3f})')
ax.axvline(days[te], color='gray', ls='--', alpha=0.5, label='Train/Val split')
burst_b = find_burst_mask(true_bac, pct=10)
for i in range(L2):
    if burst_b[i]:
        ax.axvspan(t[i] - 2, t[i] + 2, color='blue', alpha=0.08)
ax.set_ylabel('Normalized')
ax.set_title(f'Bacteria: NbedDyn + Alt 5:1 (seed=42, Pearson={p_bac:.3f})')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: Bacteria h(t)
ax = axes[3]
ax.plot(t[:len(h_bac)], h_bac[:L2], 'g-', lw=0.8)
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.axvline(days[te], color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Day')
ax.set_ylabel('h(t)')
ax.set_title('Bacteria h(t)')
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'runs/20260416_213439_beninca_nbeddyn/ost_bac_recovery.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()

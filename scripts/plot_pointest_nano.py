"""Plot Nanophyto recovery: PointEst (best seed=456)."""
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
    LatentDynamicsNet, find_burst_mask,
)
from scripts.cvhi_beninca_pointest import (
    make_model as make_model_pe, get_param_groups, freeze, unfreeze,
    alpha_schedule, HP, PA, PB, LAM_H_ODE,
)
from scripts.cvhi_beninca_nbeddyn import make_model as make_model_vae

full, species, days = load_beninca(include_nutrients=True)
species = [str(s) for s in species]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500

h_name = 'Nanophyto'
h_idx = species.index(h_name)
visible = np.delete(full, h_idx, axis=1).astype(np.float32)
hidden_raw = full[:, h_idx].astype(np.float32)
T, N = visible.shape
train_end = int(0.75 * T)


def train_run(model, seed, use_point_est):
    torch.manual_seed(seed)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=HP['lr'], weight_decay=1e-4)
    warmup = int(0.2 * EPOCHS)
    ramp = max(1, int(0.2 * EPOCHS))
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, EPOCHS - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_val = float('inf'); best_state = None; best_fh = None

    for epoch in range(EPOCHS):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_schedule(epoch, EPOCHS)
        if epoch < warmup: h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, EPOCHS - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
        if epoch >= warmup:
            cyc = (epoch - warmup) % (PA + PB)
            if cyc < PA:
                freeze(enc_params); [p.requires_grad_(False) for p in f_h.parameters()]; unfreeze(fvis_params)
            else:
                freeze(fvis_params); unfreeze(enc_params); [p.requires_grad_(True) for p in f_h.parameters()]
        else:
            unfreeze(fvis_params); unfreeze(enc_params); [p.requires_grad_(True) for p in f_h.parameters()]
        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else: x_train = x_full
        model.train(); f_h.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]; tr['G'] = out['G'][:, :train_end]
        losses = model.loss(tr, beta_kl=0.0 if use_point_est else HP.get('lam_kl', 0.017),
            free_bits=0.02, margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP['lam_cf'], lam_shuffle=HP['lam_cf']*0.6,
            lam_energy=2.0, min_energy=HP['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5*h_w,
            rollout_weights=(1.0,0.5,0.25), lam_hf=0.2, lowpass_sigma=6.0, lam_rmse_log=0.1)
        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = F.mse_loss(hp, hm[:, 1:].detach())
        (losses['total'] + LAM_H_ODE * h_w * loss_ode).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, beta_kl=0.0, h_weight=1.0, margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=HP['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0,0.5,0.25), lam_hf=0.2, lowpass_sigma=6.0)
            vr = vl['recon_full'].item()
        if epoch > warmup+15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    unfreeze(fvis_params); unfreeze(enc_params)
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oe = model(x_full, n_samples=1 if use_point_est else 30, rollout_K=3)
        if use_point_est:
            hm = oe['h_samples'][0, 0].cpu().numpy()
        else:
            hm = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    pear, h_scaled = evaluate(hm, hidden_raw)
    del f_h
    return pear, h_scaled.flatten(), hm


# Run PointEst seed=456
print('PointEst seed=456...')
model_pe = make_model_pe(N, device)
p_pe, rec_pe, h_pe = train_run(model_pe, 456, True)
del model_pe; torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f'  Pearson = {p_pe:.3f}')

# Run NbedDyn+VAE seed=456 for comparison
print('NbedDyn+VAE seed=456...')
model_vae = make_model_vae(N, device)
p_vae, rec_vae, h_vae = train_run(model_vae, 456, False)
del model_vae; torch.cuda.empty_cache() if torch.cuda.is_available() else None
print(f'  Pearson = {p_vae:.3f}')

# Plot
def norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)

L = min(len(days), len(rec_pe), len(hidden_raw))
t = days[:L]
true_n = norm(hidden_raw[:L])
burst = find_burst_mask(hidden_raw, pct=10)

fig, axes = plt.subplots(3, 1, figsize=(16, 11), gridspec_kw={'height_ratios': [2, 2, 1]})

# Panel 1: PointEst
ax = axes[0]
ax.plot(t, true_n, 'k-', lw=1.2, alpha=0.8, label='True Nanophyto')
ax.plot(t, norm(rec_pe[:L]), 'r-', lw=1.0, alpha=0.7, label=f'PointEst (r={p_pe:.3f})')
ax.axvline(days[train_end], color='gray', ls='--', alpha=0.5)
for i in range(L):
    if burst[i]: ax.axvspan(t[i]-2, t[i]+2, color='blue', alpha=0.08)
ax.set_ylabel('Normalized')
ax.set_title(f'Nanophyto: Point Estimate + NbedDyn (seed=456, Pearson={p_pe:.3f})')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: NbedDyn+VAE
ax = axes[1]
ax.plot(t, true_n, 'k-', lw=1.2, alpha=0.8, label='True Nanophyto')
ax.plot(t, norm(rec_vae[:L]), color='darkorange', lw=1.0, alpha=0.7, label=f'NbedDyn+VAE (r={p_vae:.3f})')
ax.axvline(days[train_end], color='gray', ls='--', alpha=0.5)
for i in range(L):
    if burst[i]: ax.axvspan(t[i]-2, t[i]+2, color='blue', alpha=0.08)
ax.set_ylabel('Normalized')
ax.set_title(f'Nanophyto: NbedDyn + VAE (seed=456, Pearson={p_vae:.3f})')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: h(t) comparison
ax = axes[2]
ax.plot(t[:len(h_pe)], h_pe[:L], 'r-', lw=0.8, alpha=0.7, label='h(t) PointEst')
ax.plot(t[:len(h_vae)], h_vae[:L], color='darkorange', lw=0.8, alpha=0.7, label='h(t) VAE')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.axvline(days[train_end], color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Day')
ax.set_ylabel('h(t)')
ax.set_title('Latent h(t) comparison')
ax.legend(loc='upper right', fontsize=9)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'runs/20260416_220027_beninca_pointest/nanophyto_comparison.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()

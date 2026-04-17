"""Plot Picophyto oracle recovery from per-species h."""
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
from scripts.cvhi_beninca_nbeddyn import find_burst_mask
from scripts.cvhi_beninca_per_species_h import (
    make_model_and_patch, custom_forward, get_param_groups,
    freeze, unfreeze, alpha_schedule, PerSpeciesLatentDyn,
)
from scripts.cvhi_beninca_psh_agg import custom_loss_with_agg

full, species, days = load_beninca(include_nutrients=True)
species = [str(s) for s in species]
device = 'cpu'  # force CPU to avoid stale CUDA state
EPOCHS = 500
HP = dict(encoder_d=96, encoder_blocks=3, encoder_dropout=0.1, takens_lags=(1,2,4,8),
          lr=0.0006033475528697158, lam_cf=9.517725868477207, min_energy=0.14353013693386804)
PA, PB = 5, 1; LAM_H_ODE = 0.5

h_name = 'Picophyto'; seed = 123
h_idx = species.index(h_name)
visible = np.delete(full, h_idx, axis=1).astype(np.float32)
hidden_raw = full[:, h_idx].astype(np.float32)
T, N = visible.shape
train_end = int(0.75 * T)

torch.manual_seed(seed)
x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
model = make_model_and_patch(N, device)
model.point_estimate = True
f_h = PerSpeciesLatentDyn(N, d_hidden=32).to(device)
import torch.nn as nn
agg_w = nn.Parameter(torch.zeros(N, device=device))
agg_v = nn.Parameter(torch.zeros(N, device=device))
fvis_params, enc_params = get_param_groups(model)
all_params = list(model.parameters()) + list(f_h.parameters()) + [agg_w, agg_v]
opt = torch.optim.AdamW(all_params, lr=HP['lr'], weight_decay=1e-4)
warmup = int(0.2*EPOCHS); ramp = max(1, int(0.2*EPOCHS))
def lr_lambda(step):
    if step < 50: return step/50
    p = (step-50)/max(1,EPOCHS-50)
    return 0.5*(1+np.cos(np.pi*p))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
best_val = float('inf'); best_state = None; best_fh = None; best_aw = None; best_av = None

print(f'Training {h_name} seed={seed}...')
for epoch in range(EPOCHS):
    if hasattr(model,'G_anchor_alpha'): model.G_anchor_alpha = alpha_schedule(epoch,EPOCHS)
    if epoch < warmup: h_w = 0.0
    else: h_w = min(1.0, (epoch-warmup)/ramp)
    if epoch >= warmup:
        cyc = (epoch-warmup)%(PA+PB)
        if cyc < PA:
            freeze(enc_params); [p.requires_grad_(False) for p in f_h.parameters()]
            agg_w.requires_grad_(False); agg_v.requires_grad_(False); unfreeze(fvis_params)
        else:
            freeze(fvis_params); unfreeze(enc_params); [p.requires_grad_(True) for p in f_h.parameters()]
            agg_w.requires_grad_(True); agg_v.requires_grad_(True)
    else:
        unfreeze(fvis_params); unfreeze(enc_params); [p.requires_grad_(True) for p in f_h.parameters()]
        agg_w.requires_grad_(True); agg_v.requires_grad_(True)
    if epoch > warmup:
        mask = (torch.rand(1,T,1,device=device)>0.05).float()
        x_train = x_full*mask+(1-mask)*x_full.mean(dim=1,keepdim=True)
    else: x_train = x_full
    model.train(); f_h.train(); opt.zero_grad()
    out = custom_forward(model, x_train, n_samples=2)
    T_pred = out['pred_full'].shape[2]
    out_tr = {k: (v[:,:,:train_end-1] if v.dim()==4 else v[:,:train_end-1] if k in ('pred_null','actual') else v[:,:train_end])
              for k,v in out.items() if k != 'base'}
    out_tr['base'] = out['base'][:, :train_end]
    losses = custom_loss_with_agg(out_tr, agg_w, agg_v, h_weight=h_w)
    loss_ode = torch.tensor(0.0, device=device)
    if h_w > 0:
        hm=out['mu'][:,:train_end]; xv=out['visible'][:,:train_end]
        hp=f_h(hm[:,:-1],xv[:,:-1]); loss_ode=F.mse_loss(hp,hm[:,1:].detach())
    (losses['total']+LAM_H_ODE*h_w*loss_ode).backward()
    torch.nn.utils.clip_grad_norm_(all_params,1.0); opt.step(); sched.step()
    with torch.no_grad():
        out_val = {k: (v[:,:,train_end-1:] if v.dim()==4 else v[:,train_end-1:] if k in ('pred_null','actual') else v[:,train_end:])
                   for k,v in out.items() if k != 'base'}
        out_val['base'] = out['base'][:, train_end:]
        vl = custom_loss_with_agg(out_val, agg_w, agg_v, h_weight=1.0)
        vr = vl['recon_full'].item()
    if epoch > warmup+15 and vr < best_val:
        best_val = vr
        best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
        best_fh = {k:v.detach().cpu().clone() for k,v in f_h.state_dict().items()}
        best_aw = agg_w.detach().cpu().clone(); best_av = agg_v.detach().cpu().clone()

unfreeze(fvis_params); unfreeze(enc_params)
if best_state: model.load_state_dict(best_state)
model.eval()
with torch.no_grad():
    oe = custom_forward(model, x_full, n_samples=1)
    h_all = oe['mu'][0].cpu().numpy()  # (T, N)

L = min(len(h_all), len(hidden_raw))
h_all = h_all[:L]; hidden = hidden_raw[:L]

# Oracle lstsq
X = np.concatenate([h_all, np.ones((L,1))], axis=1)
coef, _, _, _ = np.linalg.lstsq(X, hidden, rcond=None)
h_oracle = X @ coef
p_oracle = float(np.corrcoef(h_oracle, hidden)[0,1])
print(f'Oracle Pearson = {p_oracle:.3f}')

# Learned agg
if best_aw is not None:
    w_np = torch.softmax(best_aw, dim=0).numpy()
    h_learned = (h_all * w_np).sum(axis=1)
    p_learned, _ = evaluate(h_learned, hidden)
else:
    p_learned = 0.0; h_learned = np.zeros(L)

# Scalar h baseline (NbedDyn)
p_scalar = 0.162  # ref from previous experiment

# Plot
def norm(x): return (x - x.mean()) / (x.std() + 1e-8)
t = days[:L]
burst = find_burst_mask(hidden, pct=10)

fig, axes = plt.subplots(3, 1, figsize=(16, 12), gridspec_kw={'height_ratios': [2, 2, 1.5]})

# Panel 1: Oracle recovery
ax = axes[0]
ax.plot(t, norm(hidden), 'k-', lw=1.2, alpha=0.8, label='True Picophyto')
ax.plot(t, norm(h_oracle), 'b-', lw=1.0, alpha=0.7, label=f'Oracle lstsq (r={p_oracle:.3f})')
ax.axvline(days[train_end], color='gray', ls='--', alpha=0.5)
for i in range(L):
    if burst[i]: ax.axvspan(t[i]-2, t[i]+2, color='blue', alpha=0.08)
ax.set_ylabel('Normalized')
ax.set_title(f'Picophyto: Oracle linear combination of 12 h_i (Pearson={p_oracle:.3f})')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 2: Learned vs Oracle
ax = axes[1]
ax.plot(t, norm(hidden), 'k-', lw=1.2, alpha=0.8, label='True Picophyto')
ax.plot(t, norm(h_oracle), 'b-', lw=0.8, alpha=0.5, label=f'Oracle (r={p_oracle:.3f})')
ax.plot(t, norm(h_learned), 'r-', lw=0.8, alpha=0.7, label=f'Learned agg (r={p_learned:.3f})')
ax.axvline(days[train_end], color='gray', ls='--', alpha=0.5)
ax.set_ylabel('Normalized')
ax.set_title('Comparison: Oracle vs Learned aggregation')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Per-species h_i contributions (top 5 by oracle weight)
ax = axes[2]
vis_names = [s for s in species if s != h_name]
top_idx = np.argsort(np.abs(coef[:-1]))[::-1][:5]
for rank, j in enumerate(top_idx):
    ax.plot(t, h_all[:, j], lw=0.7, alpha=0.6,
            label=f'h_{vis_names[j]} (w={coef[j]:.3f})')
ax.axhline(0, color='gray', ls=':', alpha=0.5)
ax.axvline(days[train_end], color='gray', ls='--', alpha=0.5)
ax.set_xlabel('Day')
ax.set_ylabel('h_i(t)')
ax.set_title('Top 5 h_i channels by oracle weight')
ax.legend(fontsize=8, ncol=2)
ax.grid(True, alpha=0.3)

plt.tight_layout()
out_path = 'runs/20260416_232943_beninca_psh_agg/picophyto_oracle.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'Saved: {out_path}')
plt.close()

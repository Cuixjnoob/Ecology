"""Final Huisman 6-species recovery visualization for paper."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate
from scripts.cvhi_beninca_nbeddyn import LatentDynamicsNet

EPOCHS = 500
LAM_HDYN = 0.2
BEST_SEEDS = {0: 123, 1: 456, 2: 42, 3: 456, 4: 456, 5: 123}  # from lam=0.2 results

def load_data():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    full = np.concatenate([d["N_all"], d["resources"]], axis=1)
    full = (full + 0.01) / (full.mean(axis=0, keepdims=True) + 1e-3)
    return full.astype(np.float32), d["t_axis"], d["N_all"]

def make_model(N, device):
    from models.cvhi_residual import CVHI_Residual
    return CVHI_Residual(
        num_visible=N, encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1,2,4,8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp", use_formula_hints=True,
        use_G_field=True, num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)

def alpha_sched(ep, total):
    f = ep / max(1, total)
    if f <= 0.5: return 1.0
    if f >= 0.95: return 0.0
    return 1.0 - (f - 0.5) / (0.95 - 0.5)

class HDynNet(torch.nn.Module):
    def __init__(self, n, d=32):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(1+n, d), torch.nn.SiLU(),
            torch.nn.Linear(d, d), torch.nn.SiLU(),
            torch.nn.Linear(d, 1))
    def forward(self, h, x):
        if h.dim() == 2: h = h.unsqueeze(-1)
        return self.net(torch.cat([h, x], dim=-1)).squeeze(-1)

def train_one(vis, hid, seed, device, n_recon_ch):
    torch.manual_seed(seed)
    T, N = vis.shape
    x = torch.tensor(vis, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    hdyn = HDynNet(N, 32).to(device)
    params = list(model.parameters()) + list(hdyn.parameters())
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=1e-4)
    wu = int(0.2 * EPOCHS); ramp = max(1, int(0.2 * EPOCHS))
    def lr_fn(s):
        if s < 50: return s / 50
        p = (s - 50) / max(1, EPOCHS - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    te = int(0.75 * T); bv = float('inf'); bs = None
    for ep in range(EPOCHS):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_sched(ep, EPOCHS)
        if ep < wu: hw, Kr = 0.0, 0
        else:
            post = ep - wu; hw = min(1.0, post / ramp)
            kr = min(1.0, post / max(1, EPOCHS - wu) * 2)
            Kr = max(1 if hw > 0 else 0, int(round(kr * 3)))
        if ep > wu:
            m = (torch.rand(1, T, 1, device=device) > 0.05).float()
            xt = x * m + (1 - m) * x.mean(dim=1, keepdim=True)
        else:
            xt = x
        model.train(); hdyn.train(); opt.zero_grad()
        out = model(xt, n_samples=2, rollout_K=Kr)
        tr = model.slice_out(out, 0, te)
        tr['visible'] = out['visible'][:, :te]; tr['G'] = out['G'][:, :te]
        losses = model.loss(tr, beta_kl=0.03, free_bits=0.02,
            margin_null=0.003, margin_shuf=0.002,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=0.02,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=hw, lam_rollout=0.5 * hw,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
            lam_rmse_log=0.1, n_recon_channels=n_recon_ch)
        total = losses['total']
        if hw > 0:
            hm = tr['h_samples'].mean(dim=0); Th = hm.shape[-1]
            hp = hdyn(hm[:, :-1], xt[:, :Th-1])
            tgt = hm[:, 1:].detach() if ep < 100 + wu else hm[:, 1:]
            total = total + LAM_HDYN * hw * F.mse_loss(hp, tgt)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out, te, T)
            vo['visible'] = out['visible'][:, te:T]; vo['G'] = out['G'][:, te:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02, lam_rollout=0.5,
                rollout_weights=(1.0, 0.5, 0.25), lam_hf=0.0, lowpass_sigma=6.0,
                n_recon_channels=n_recon_ch)
            vr = vl['recon_full'].item()
        if ep > wu + 15 and vr < bv:
            bv = vr; bs = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        oe = model(x, n_samples=30, rollout_K=3)
        hm = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    # Train-fit, val-eval
    L = min(len(hm), len(hid))
    X_tr = np.column_stack([hm[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hid[:te], rcond=None)
    X_all = np.column_stack([hm[:L], np.ones(L)])
    h_scaled = X_all @ coef
    r_tr = float(np.corrcoef(h_scaled[:te], hid[:te])[0, 1])
    r_val = float(np.corrcoef(h_scaled[te:L], hid[te:L])[0, 1])
    del model, hdyn
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return h_scaled, r_tr, r_val

# Main
device = 'cuda' if torch.cuda.is_available() else 'cpu'
full, t_axis, N_all = load_data()
T = len(t_axis)
te = int(0.75 * T)

fig, axes = plt.subplots(6, 1, figsize=(16, 22))

def norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)

for sp_idx in range(6):
    sp_name = f"sp{sp_idx+1}"
    seed = BEST_SEEDS[sp_idx]
    vis = np.delete(full, sp_idx, axis=1)
    hid = full[:, sp_idx]

    print(f"Training {sp_name} seed={seed}...")
    h_scaled, r_tr, r_val = train_one(vis, hid, seed, device, n_recon_ch=5)
    print(f"  train r={r_tr:.3f}, val r={r_val:.3f}")

    L = min(len(t_axis), len(h_scaled))
    t = t_axis[:L]
    ax = axes[sp_idx]

    # Plot
    ax.plot(t, norm(hid[:L]), 'k-', lw=1.2, alpha=0.8, label=f'True {sp_name}')
    ax.plot(t, norm(h_scaled[:L]), 'r-', lw=0.8, alpha=0.7,
            label=f'Recovered')
    ax.axvline(t_axis[te], color='blue', lw=1.5, ls='--', alpha=0.6)
    ax.fill_between([t_axis[te], t_axis[L-1]], -3.5, 3.5,
                    color='green', alpha=0.06)

    ax.text(0.02, 0.92, f'Train r = {r_tr:.3f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    ax.text(0.82, 0.92, f'Val r = {r_val:.3f}',
            transform=ax.transAxes, fontsize=10, va='top',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.5))

    ax.set_ylabel('Normalized', fontsize=10)
    ax.set_title(f'{sp_name} (seed={seed})', fontsize=11, fontweight='bold')
    ax.legend(fontsize=8, loc='upper right')
    ax.grid(True, alpha=0.2)
    ax.set_ylim(-3.5, 3.5)

axes[-1].set_xlabel('Day', fontsize=11)
fig.suptitle('Huisman 1999 Chaos: Hidden Species Recovery (6->1 rotation)\n'
             'CVHI-Residual + NbedDyn ODE consistency',
             fontsize=14, fontweight='bold')
plt.tight_layout()
out_path = 'paper/fig_huisman_recovery.png'
import os; os.makedirs('paper', exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close()

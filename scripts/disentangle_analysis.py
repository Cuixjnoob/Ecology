"""Disentanglement analysis: separate hidden-species signal from shared residual.

Idea: Train NbedDyn with each of 9 species hidden (same seed).
The SHARED component across rotations = model misspecification / env noise.
The SPECIFIC component (h_i - mean) = species-specific signal.

If Pearson improves after subtracting shared component, it confirms that
h was contaminated by non-species factors, and the specific component
better represents the actual hidden species.
"""
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
    freeze, unfreeze, alpha_schedule, HP, PA, PB, LAM_H_ODE
)

full, species, days = load_beninca(include_nutrients=True)
species = [str(s) for s in species]
device = 'cuda' if torch.cuda.is_available() else 'cpu'
EPOCHS = 500
SEED = 42

SPECIES_ORDER = ['Cyclopoids', 'Calanoids', 'Rotifers', 'Nanophyto',
                 'Picophyto', 'Filam_diatoms', 'Ostracods',
                 'Harpacticoids', 'Bacteria']


def run_one(h_name):
    h_idx = species.index(h_name)
    visible = np.delete(full, h_idx, axis=1).astype(np.float32)
    hidden_raw = full[:, h_idx].astype(np.float32)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    torch.manual_seed(SEED)
    model = make_model(N, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=HP['lr'], weight_decay=1e-4)
    warmup = int(0.2 * EPOCHS)
    ramp = max(1, int(0.2 * EPOCHS))

    def lr_lambda(step):
        if step < 50: return step / 50
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
            cyc = (epoch - warmup) % (PA + PB)
            if cyc < PA:
                freeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(False)
                unfreeze(fvis_params)
            else:
                freeze(fvis_params)
                unfreeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(True)
        else:
            unfreeze(fvis_params)
            unfreeze(enc_params)
            for p in f_h.parameters(): p.requires_grad_(True)

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); f_h.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]
        tr['G'] = out['G'][:, :train_end]
        losses = model.loss(
            tr, beta_kl=HP['lam_kl'], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP['lam_cf'], lam_shuffle=HP['lam_cf'] * 0.6,
            lam_energy=2.0, min_energy=HP['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.2, lowpass_sigma=6.0, lam_rmse_log=0.1)
        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0:
            hm = out['mu'][:, :train_end]
            xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = F.mse_loss(hp, hm[:, 1:].detach())
        (losses['total'] + LAM_H_ODE * h_w * loss_ode).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]
            vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=HP['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.2, lowpass_sigma=6.0)
            vr = vl['recon_full'].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    unfreeze(fvis_params); unfreeze(enc_params)
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        h_mean = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    pear, h_scaled = evaluate(h_mean, hidden_raw)
    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return h_scaled.flatten(), hidden_raw, pear


# ===== Run all 9 rotations =====
all_h = {}
all_hidden = {}
all_pear = {}
for sp in SPECIES_ORDER:
    print(f'Running {sp}...')
    h_sc, hid, p = run_one(sp)
    all_h[sp] = h_sc
    all_hidden[sp] = hid
    all_pear[sp] = p
    print(f'  Pearson = {p:.3f}')


# ===== Disentanglement analysis =====
L = min(len(v) for v in all_h.values())
H = np.stack([all_h[sp][:L] for sp in SPECIES_ORDER])  # (9, L)
h_mean_across = H.mean(axis=0)  # (L,) shared component


def pearson(a, b):
    a_c = a - a.mean()
    b_c = b - b.mean()
    return float(np.sum(a_c * b_c) / (np.sqrt(np.sum(a_c**2) * np.sum(b_c**2)) + 1e-8))


print('\n' + '=' * 80)
print('DISENTANGLEMENT ANALYSIS')
print('=' * 80)
header = f"{'Species':<16} {'Original':>10} {'Specific':>10} {'Shared':>10} {'Delta':>10}"
print(header)
print('-' * 80)

pear_orig = []
pear_spec = []
pear_shared = []

for i, sp in enumerate(SPECIES_ORDER):
    h_orig = H[i]
    h_specific = h_orig - h_mean_across
    hidden = all_hidden[sp][:L]

    p_orig = pearson(h_orig, hidden)
    p_spec = pearson(h_specific, hidden)
    p_shared = pearson(h_mean_across, hidden)
    delta = p_spec - p_orig

    pear_orig.append(p_orig)
    pear_spec.append(p_spec)
    pear_shared.append(p_shared)

    print(f'{sp:<16} {p_orig:>+10.3f} {p_spec:>+10.3f} {p_shared:>+10.3f} {delta:>+10.3f}')

print('-' * 80)
avg_orig = np.mean(pear_orig)
avg_spec = np.mean(pear_spec)
avg_shared = np.mean(pear_shared)
print(f"{'Overall':<16} {avg_orig:>+10.3f} {avg_spec:>+10.3f} {avg_shared:>+10.3f} {avg_spec - avg_orig:>+10.3f}")

# Variance decomposition
total_var = np.var(H, axis=1).mean()
shared_var = np.var(h_mean_across)
specific_var = np.var(H - h_mean_across, axis=1).mean()
print(f'\nVariance decomposition:')
print(f'  Total h variance:    {total_var:.4f}')
print(f'  Shared component:    {shared_var:.4f} ({shared_var / total_var * 100:.1f}%)')
print(f'  Specific component:  {specific_var:.4f} ({specific_var / total_var * 100:.1f}%)')


# ===== Plot =====
fig, axes = plt.subplots(4, 1, figsize=(16, 16),
                         gridspec_kw={'height_ratios': [2, 2, 2, 1.5]})
t = days[:L]

# Panel 1: All 9 h trajectories + shared mean
ax = axes[0]
colors = plt.cm.tab10(np.linspace(0, 1, 9))
for i, sp in enumerate(SPECIES_ORDER):
    ax.plot(t, H[i], color=colors[i], lw=0.6, alpha=0.6, label=sp)
ax.plot(t, h_mean_across, 'k-', lw=2.5, label='Shared mean')
ax.set_ylabel('h(t)')
ax.set_title('All 9 rotation h(t) overlaid + shared mean (black)')
ax.legend(fontsize=7, ncol=5, loc='upper right')
ax.grid(True, alpha=0.3)

# Panel 2: Ostracods decomposition
def norm(x):
    return (x - x.mean()) / (x.std() + 1e-8)

bi = SPECIES_ORDER.index('Ostracods')
ax = axes[1]
ax.plot(t, norm(all_hidden['Ostracods'][:L]), 'k-', lw=1.2, alpha=0.8, label='True Ostracods')
ax.plot(t, norm(H[bi]), 'r-', lw=0.8, alpha=0.6,
        label=f'h original (r={pear_orig[bi]:.3f})')
ax.plot(t, norm(H[bi] - h_mean_across), 'b-', lw=0.8, alpha=0.6,
        label=f'h specific (r={pear_spec[bi]:.3f})')
ax.set_ylabel('Normalized')
ax.set_title('Ostracods: original h vs species-specific h (shared subtracted)')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 3: Bacteria decomposition
bi2 = SPECIES_ORDER.index('Bacteria')
ax = axes[2]
ax.plot(t, norm(all_hidden['Bacteria'][:L]), 'k-', lw=1.2, alpha=0.8, label='True Bacteria')
ax.plot(t, norm(H[bi2]), 'r-', lw=0.8, alpha=0.6,
        label=f'h original (r={pear_orig[bi2]:.3f})')
ax.plot(t, norm(H[bi2] - h_mean_across), 'b-', lw=0.8, alpha=0.6,
        label=f'h specific (r={pear_spec[bi2]:.3f})')
ax.set_ylabel('Normalized')
ax.set_title('Bacteria: original h vs species-specific h')
ax.legend(fontsize=9)
ax.grid(True, alpha=0.3)

# Panel 4: Bar chart
ax = axes[3]
x_pos = np.arange(len(SPECIES_ORDER))
w = 0.35
ax.bar(x_pos - w / 2, pear_orig, w, label='Original h', color='red', alpha=0.7)
ax.bar(x_pos + w / 2, pear_spec, w, label='Specific h (shared removed)', color='blue', alpha=0.7)
ax.set_xticks(x_pos)
ax.set_xticklabels(SPECIES_ORDER, rotation=45, ha='right', fontsize=8)
ax.set_ylabel('Pearson with hidden species')
ax.set_title('Original h vs Species-specific h (shared component removed)')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='gray', ls='-', alpha=0.3)

plt.tight_layout()
out_path = 'runs/20260416_213439_beninca_nbeddyn/disentangle_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f'\nSaved: {out_path}')
plt.close()

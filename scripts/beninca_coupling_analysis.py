"""Beninca: analyze what determines hidden species recoverability.

Use GNN's learned G field as proxy for coupling structure.
Correlate with val-only Pearson.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
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

# Val-only Pearson from 5-seed experiment
pearson_val = {
    'Cyclopoids': -0.200, 'Calanoids': 0.037, 'Rotifers': 0.495,
    'Nanophyto': -0.075, 'Picophyto': 0.240, 'Filam_diatoms': 0.071,
    'Ostracods': 0.390, 'Harpacticoids': 0.358, 'Bacteria': -0.260,
}

print('Extracting GNN G-field for each rotation...')
results = {}

for h_name in SPECIES_ORDER:
    h_idx = species.index(h_name)
    visible = np.delete(full, h_idx, axis=1).astype(np.float32)
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
        if step < 50:
            return step / 50
        p = (step - 50) / max(1, EPOCHS - 50)
        return 0.5 * (1 + np.cos(np.pi * p))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float('inf')
    best_state = None

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
        opt.step()
        sched.step()
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

    if best_state:
        model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        G = model.compute_G(x_full)  # (1, T, N)
        G_mean = G[0].mean(dim=0).cpu().numpy()

    G_abs = np.abs(G_mean)
    total_G = G_abs.sum()
    G_frac = G_abs / (total_G + 1e-8)
    specificity = G_frac.std() / (G_frac.mean() + 1e-8)
    entropy = -np.sum(G_frac * np.log(G_frac + 1e-10))

    results[h_name] = {
        'total_G': total_G,
        'specificity': specificity,
        'entropy': entropy,
        'val_pearson': pearson_val[h_name],
    }

    print(f"  {h_name:<16}: total_|G|={total_G:.3f}  spec={specificity:.3f}  "
          f"ent={entropy:.3f}  val_P={pearson_val[h_name]:+.3f}")

    del model, f_h
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Correlations
print("\n" + "=" * 70)
print("BENINCA: Coupling metrics vs Recovery")
print("=" * 70)

total_Gs = [results[sp]['total_G'] for sp in SPECIES_ORDER]
specs = [results[sp]['specificity'] for sp in SPECIES_ORDER]
ents = [results[sp]['entropy'] for sp in SPECIES_ORDER]
pearsons = [results[sp]['val_pearson'] for sp in SPECIES_ORDER]

r1, p1 = stats.pearsonr(total_Gs, pearsons)
r2, p2 = stats.pearsonr(specs, pearsons)
r3, p3 = stats.pearsonr(ents, pearsons)

print(f"Corr(total_|G|, Pearson)   = {r1:+.3f} (p={p1:.3f})")
print(f"Corr(specificity, Pearson) = {r2:+.3f} (p={p2:.3f})")
print(f"Corr(entropy, Pearson)     = {r3:+.3f} (p={p3:.3f})")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
data_pairs = [
    (axes[0], total_Gs, 'Total |G| (coupling strength)', r1, p1),
    (axes[1], specs, 'G specificity (CV)', r2, p2),
    (axes[2], ents, 'G entropy (diffuseness)', r3, p3),
]

for ax, vals, label, r, p in data_pairs:
    ax.scatter(vals, pearsons, s=80, c='darkred', zorder=5)
    for i, sp in enumerate(SPECIES_ORDER):
        ax.annotate(sp[:4], (vals[i], pearsons[i]),
                    textcoords='offset points', xytext=(5, 5), fontsize=8)
    z = np.polyfit(vals, pearsons, 1)
    x_line = np.linspace(min(vals), max(vals), 50)
    ax.plot(x_line, np.polyval(z, x_line), 'r--', alpha=0.5)
    ax.set_xlabel(label)
    ax.set_ylabel('Val Pearson')
    ax.set_title(f'r = {r:+.3f} (p = {p:.3f})')
    ax.axhline(0.078, color='gray', ls=':', alpha=0.5, label='random baseline')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=7)

plt.suptitle('Beninca: What determines hidden species recoverability?\n'
             '(G field from learned GNN)', fontsize=13)
plt.tight_layout()
out_path = 'runs/20260417_120615_beninca_valonly/recoverability_analysis.png'
plt.savefig(out_path, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_path}")
plt.close()

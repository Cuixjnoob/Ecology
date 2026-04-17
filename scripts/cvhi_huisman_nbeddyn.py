"""Validate NbedDyn + alt 5:1 on Huisman 1999 synthetic chaos data.

Ground truth is perfectly known. If method works here, the low Pearson
on Beninca is due to real-world complexity, not method failure.

Saves trajectories for visualization.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate
from scripts.cvhi_beninca_nbeddyn import (
    LatentDynamicsNet, find_burst_mask, burst_precision_recall,
)

# Generate data
from scripts.generate_huisman1999 import generate

SEEDS = [42, 123, 456]
EPOCHS = 500
HP = dict(
    encoder_d=64, encoder_blocks=2, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0008,
    lam_kl=0.02,
    lam_cf=8.0,
    min_energy=0.1,
)
PA, PB = 5, 1
LAM_H_ODE = 0.5


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=HP["encoder_d"], encoder_blocks=HP["encoder_blocks"],
        encoder_heads=4,
        takens_lags=HP["takens_lags"], encoder_dropout=HP["encoder_dropout"],
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def get_param_groups(model):
    enc_ids = set()
    for name, p in model.named_parameters():
        if 'encoder' in name or 'readout' in name:
            enc_ids.add(id(p))
    fvis = [p for p in model.parameters() if id(p) not in enc_ids]
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    return fvis, enc


def freeze(params):
    for p in params: p.requires_grad_(False)
def unfreeze(params):
    for p in params: p.requires_grad_(True)

def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden, seed, device, n_recon_ch=None, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=HP["lr"], weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_fh = None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup: h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup; h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
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
        losses = model.loss(tr, beta_kl=HP['lam_kl'], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP['lam_cf'], lam_shuffle=HP['lam_cf'] * 0.6,
            lam_energy=2.0, min_energy=HP['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25), lam_hf=0.2, lowpass_sigma=6.0,
            lam_rmse_log=0.1, n_recon_channels=n_recon_ch)
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
            vl = model.loss(vo, h_weight=1.0, margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=HP['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25), lam_hf=0.2, lowpass_sigma=6.0,
                n_recon_channels=n_recon_ch)
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
    pear, h_scaled = evaluate(h_mean, hidden)
    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return pear, h_scaled.flatten(), h_mean


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_nbeddyn_validate")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Generating Huisman 1999 chaos data...")
    N_all, R_all, t_axis = generate(t_transient=1000.0, t_record=2000.0, dt=2.0)
    visible_species = N_all[:, :5]
    hidden_species = N_all[:, 5]
    # Combine visible species + resources as input (resources = input only, like nutrients)
    visible = np.concatenate([visible_species, R_all], axis=1).astype(np.float32)
    hidden = hidden_species.astype(np.float32)
    T, N = visible.shape
    n_species = 5  # only species in recon loss, not resources
    train_end = int(0.75 * T)

    # Normalize per channel (same as Beninca)
    visible = visible + 0.01
    col_means = visible.mean(axis=0, keepdims=True)
    col_means = np.maximum(col_means, 1e-3)
    visible = visible / col_means

    print(f"Visible: {visible.shape} (5 species + 5 resources)")
    print(f"Hidden: species 6, shape {hidden.shape}")
    print(f"T={T}, dt=2 day, train/val split at {train_end}\n")

    # Run 3 seeds
    results = []
    best_pear = -1; best_h_scaled = None; best_h_mean = None; best_seed = None

    for seed in SEEDS:
        print(f"  seed={seed}...", end=" ", flush=True)
        pear, h_scaled, h_mean = train_one(visible, hidden, seed, device, n_recon_ch=n_species)
        print(f"Pearson = {pear:.3f}")
        results.append({"seed": seed, "pearson": pear})
        if pear > best_pear:
            best_pear = pear; best_h_scaled = h_scaled; best_h_mean = h_mean; best_seed = seed

    avg_pear = np.mean([r["pearson"] for r in results])
    print(f"\nOverall: {avg_pear:.3f} (best seed={best_seed}: {best_pear:.3f})")
    print(f"Ref: Beninca NbedDyn = 0.162")

    # --- Plot ---
    L = min(len(t_axis), len(best_h_scaled), len(hidden))
    t = t_axis[:L]

    def norm(x):
        return (x - x.mean()) / (x.std() + 1e-8)

    fig, axes = plt.subplots(4, 1, figsize=(16, 14),
                             gridspec_kw={'height_ratios': [2, 2, 1, 1]})

    # Panel 1: All 6 species
    ax = axes[0]
    for i in range(6):
        lw = 2.0 if i == 5 else 0.8
        alpha = 1.0 if i == 5 else 0.5
        label = f'Species {i+1}' + (' (HIDDEN)' if i == 5 else '')
        ax.plot(t, N_all[:L, i], lw=lw, alpha=alpha, label=label)
    ax.axvline(t_axis[train_end], color='gray', ls='--', alpha=0.5)
    ax.set_ylabel('Abundance')
    ax.set_title('Huisman 1999 Chaos: 6 species competing for 5 resources')
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)

    # Panel 2: True hidden vs recovered
    ax = axes[1]
    ax.plot(t, norm(hidden[:L]), 'k-', lw=1.5, alpha=0.8, label='True hidden (sp6)')
    ax.plot(t, norm(best_h_scaled[:L]), 'r-', lw=1.0, alpha=0.7,
            label=f'Recovered (r={best_pear:.3f})')
    ax.axvline(t_axis[train_end], color='gray', ls='--', alpha=0.5, label='Train/Val split')
    ax.set_ylabel('Normalized')
    ax.set_title(f'Hidden Species Recovery: NbedDyn + Alt 5:1 (seed={best_seed}, Pearson={best_pear:.3f})')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel 3: h(t) latent variable
    ax = axes[2]
    ax.plot(t[:len(best_h_mean)], best_h_mean[:L], 'g-', lw=0.8)
    ax.axhline(0, color='gray', ls=':', alpha=0.5)
    ax.axvline(t_axis[train_end], color='gray', ls='--', alpha=0.5)
    ax.set_ylabel('h(t)')
    ax.set_title('Inferred latent h(t)')
    ax.grid(True, alpha=0.3)

    # Panel 4: Scatter plot (train vs val)
    ax = axes[3]
    h_sc_train = best_h_scaled[:train_end]
    h_sc_val = best_h_scaled[train_end:L]
    hid_train = hidden[:train_end]
    hid_val = hidden[train_end:L]
    ax.scatter(hid_train, h_sc_train, s=3, alpha=0.3, c='blue', label='Train')
    ax.scatter(hid_val, h_sc_val, s=5, alpha=0.5, c='red', label='Val')
    r_train = float(np.corrcoef(h_sc_train, hid_train)[0, 1])
    r_val = float(np.corrcoef(h_sc_val, hid_val)[0, 1])
    ax.set_xlabel('True hidden species')
    ax.set_ylabel('Recovered h (scaled)')
    ax.set_title(f'Scatter: Train r={r_train:.3f}, Val r={r_val:.3f}')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig_path = out_dir / 'huisman_recovery.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f'Saved: {fig_path}')
    plt.close()

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Huisman 1999 Validation: NbedDyn + Alt 5:1\n\n")
        f.write(f"Setup: 5 visible species + 5 resources, 1 hidden species (sp6)\n")
        f.write(f"T={T}, dt=2 day, chaotic dynamics\n\n")
        for r in results:
            f.write(f"- seed={r['seed']}: Pearson={r['pearson']:.3f}\n")
        f.write(f"\n**Average: {avg_pear:.3f}**, Best: {best_pear:.3f} (seed={best_seed})\n")
        f.write(f"Train r={r_train:.3f}, Val r={r_val:.3f}\n")
        f.write(f"\nRef: Beninca NbedDyn = 0.162\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

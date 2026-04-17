"""Full Huisman 6->1 rotation: hdyn_only (best from ablation), 500 epochs.

All 6 species take turns as hidden. Resources as input-only.
Save trajectories for visualization.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate

SEEDS = [42, 123, 456]
EPOCHS = 500
ALL_SP = list(range(6))
ALL_NAMES = [f"sp{i+1}" for i in range(6)]
LAM_HDYN = 0.2


class HDynNet(nn.Module):
    def __init__(self, n_vis, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1+n_vis, d), nn.SiLU(),
                                  nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, h, x):
        if h.dim() == 2: h = h.unsqueeze(-1)
        return self.net(torch.cat([h, x], dim=-1)).squeeze(-1)


def load_huisman():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    full = np.concatenate([d["N_all"], d["resources"]], axis=1)
    full = (full + 0.01) / (full.mean(axis=0, keepdims=True) + 1e-3)
    return full.astype(np.float32), d["t_axis"], d["N_all"]


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N, encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1,2,4,8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp", use_formula_hints=True,
        use_G_field=True, num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def alpha_sched(ep, total, s=0.5, e=0.95):
    f = ep/max(1,total)
    if f <= s: return 1.0
    if f >= e: return 0.0
    return 1.0-(f-s)/(e-s)


def train(visible, hidden, seed, device, n_recon_ch, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    hdyn = HDynNet(N, 32).to(device)
    params = list(model.parameters()) + list(hdyn.parameters())
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=1e-4)
    wu = int(0.2*epochs); ramp = max(1, int(0.2*epochs))
    DETACH_UNTIL = 100
    def lr_fn(s):
        if s < 50: return s/50
        p = (s-50)/max(1,epochs-50)
        return 0.5*(1+np.cos(np.pi*p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    te = int(0.75*T); bv = float('inf'); bs = None

    for ep in range(epochs):
        if hasattr(model,'G_anchor_alpha'): model.G_anchor_alpha = alpha_sched(ep,epochs)
        if ep < wu: hw, Kr = 0.0, 0
        else:
            post = ep-wu; hw = min(1.0, post/ramp)
            kr = min(1.0, post/max(1,epochs-wu)*2)
            Kr = max(1 if hw>0 else 0, int(round(kr*3)))
        if ep > wu:
            m = (torch.rand(1,T,1,device=device)>0.05).float()
            xt = x*m+(1-m)*x.mean(dim=1,keepdim=True)
        else: xt = x
        model.train(); hdyn.train(); opt.zero_grad()
        out = model(xt, n_samples=2, rollout_K=Kr)
        tr = model.slice_out(out, 0, te)
        tr['visible'] = out['visible'][:,:te]; tr['G'] = out['G'][:,:te]
        losses = model.loss(tr, beta_kl=0.03, free_bits=0.02,
            margin_null=0.003, margin_shuf=0.002,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=0.02,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=hw, lam_rollout=0.5*hw,
            rollout_weights=(1.0,0.5,0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
            lam_rmse_log=0.1, n_recon_channels=n_recon_ch)
        total = losses['total']
        if hw > 0:
            hm = tr['h_samples'].mean(dim=0)
            Th = hm.shape[-1]
            hp = hdyn(hm[:,:-1], xt[:,:Th-1])
            tgt = hm[:,1:].detach() if ep < DETACH_UNTIL+wu else hm[:,1:]
            total = total + LAM_HDYN * hw * F.mse_loss(hp, tgt)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out, te, T)
            vo['visible'] = out['visible'][:,te:T]; vo['G'] = out['G'][:,te:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02, lam_rollout=0.5,
                rollout_weights=(1.0,0.5,0.25), lam_hf=0.0, lowpass_sigma=6.0,
                n_recon_channels=n_recon_ch)
            vr = vl['recon_full'].item()
        if ep > wu+15 and vr < bv:
            bv = vr; bs = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        oe = model(x, n_samples=30, rollout_K=3)
        hm = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    p, hs = evaluate(hm, hidden)
    del model, hdyn
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return p, hs.flatten(), hm


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_full")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, t_axis, N_all = load_huisman()
    n_recon_ch = 5  # 5 visible species, resources as input-only

    print(f"Huisman Full 6->1 Rotation: hdyn_only, lam={LAM_HDYN}, {EPOCHS}ep")
    print(f"6 species x 3 seeds = {6*3} runs\n")

    results = {n: [] for n in ALL_NAMES}
    best_per_sp = {}  # store best trajectory per species
    total = len(ALL_SP) * len(SEEDS)
    ri = 0

    for sp_idx, sp_name in zip(ALL_SP, ALL_NAMES):
        visible = np.delete(full, sp_idx, axis=1)
        hidden = full[:, sp_idx]
        print(f"--- hidden={sp_name} ---")
        best_p = -1
        for seed in SEEDS:
            ri += 1
            t0 = datetime.now()
            p, hs, hm = train(visible, hidden, seed, device, n_recon_ch)
            dt = (datetime.now() - t0).total_seconds()
            print(f"  [{ri}/{total}] seed={seed}  P={p:+.3f}  ({dt:.1f}s)")
            results[sp_name].append({"seed": seed, "pearson": p})
            if p > best_p:
                best_p = p
                best_per_sp[sp_name] = {"p": p, "seed": seed, "h_scaled": hs, "h_mean": hm}

    # Summary
    print(f"\n{'='*70}")
    print("HUISMAN FULL 6->1 ROTATION")
    print('='*70)
    all_means = []
    for n in ALL_NAMES:
        ps = [r["pearson"] for r in results[n]]
        m = np.mean(ps)
        all_means.append(m)
        print(f"  {n}: mean={m:+.3f}  seeds={[f'{p:+.3f}' for p in ps]}")
    overall = np.mean(all_means)
    print(f"\n  Overall: {overall:+.3f}")
    print(f"  Ref: ablation hdyn_only(300ep, 3sp) = 0.525")

    # Save summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Huisman Full 6->1: hdyn_only, lam={LAM_HDYN}, {EPOCHS}ep\n\n")
        f.write("| Species | Mean P | Seeds |\n|---|---|---|\n")
        for n in ALL_NAMES:
            ps = [r["pearson"] for r in results[n]]
            f.write(f"| {n} | {np.mean(ps):+.3f} | {', '.join(f'{p:+.3f}' for p in ps)} |\n")
        f.write(f"\n**Overall: {overall:+.3f}**\n")

    # Save raw results
    raw = {n: [{"seed": r["seed"], "pearson": float(r["pearson"])} for r in results[n]] for n in ALL_NAMES}
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)

    # Plot: best recovery for each species
    T = len(t_axis)
    train_end = int(0.75 * T)
    fig, axes = plt.subplots(6, 1, figsize=(16, 20), constrained_layout=True)
    for i, (sp_name, ax) in enumerate(zip(ALL_NAMES, axes)):
        if sp_name not in best_per_sp:
            continue
        info = best_per_sp[sp_name]
        hidden = full[:, i]
        L = min(len(info["h_scaled"]), len(hidden), len(t_axis))
        t = t_axis[:L]
        def norm(x): return (x - x.mean()) / (x.std() + 1e-8)
        ax.plot(t, norm(hidden[:L]), 'k-', lw=1.2, alpha=0.8, label=f'True {sp_name}')
        ax.plot(t, norm(info["h_scaled"][:L]), 'r-', lw=0.8, alpha=0.7,
                label=f'Recovered (r={info["p"]:.3f})')
        ax.axvline(t_axis[train_end], color='gray', ls='--', alpha=0.5)
        ax.set_ylabel('Norm')
        ax.set_title(f'{sp_name}: Pearson={info["p"]:.3f} (seed={info["seed"]})')
        ax.legend(fontsize=8, loc='upper right')
        ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel('Day')
    fig.suptitle(f'Huisman Full 6->1: hdyn_only, Overall={overall:.3f}', fontsize=14)
    fig.savefig(out_dir / 'all_species_recovery.png', dpi=150)
    plt.close()
    print(f"Saved: {out_dir / 'all_species_recovery.png'}")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

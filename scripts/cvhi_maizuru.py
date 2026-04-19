"""Eco-GNRD on Maizuru Bay fish community (5 species + 2 temp).

Open system test. Temp as input-only (not in recon loss).
hdyn_only (no alternating - following Huisman best practice for new data).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate
from scripts.cvhi_beninca_nbeddyn import LatentDynamicsNet
from scripts.load_maizuru import load_maizuru

SEEDS = [42, 123, 456]
EPOCHS = 500
LAM_HDYN = 0.1


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N, encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.15,
        d_species_f=16, f_visible_layers=2, f_visible_top_k=3,
        d_species_G=10, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp", use_formula_hints=True,
        use_G_field=True, num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def alpha_sched(ep, total):
    f = ep / max(1, total)
    if f <= 0.5: return 1.0
    if f >= 0.95: return 0.0
    return 1.0 - (f - 0.5) / (0.95 - 0.5)


class HDynNet(nn.Module):
    def __init__(self, n, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1+n, d), nn.SiLU(),
                                  nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, h, x):
        if h.dim() == 2: h = h.unsqueeze(-1)
        return self.net(torch.cat([h, x], dim=-1)).squeeze(-1)


def train_one(visible, hidden, seed, device, n_recon_ch, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    hdyn = HDynNet(N, 32).to(device)
    params = list(model.parameters()) + list(hdyn.parameters())
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=1e-4)
    wu = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    def lr_fn(s):
        if s < 50: return s / 50
        p = (s - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    te = int(0.75 * T); bv = float('inf'); bs = None

    for ep in range(epochs):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_sched(ep, epochs)
        if ep < wu: hw, Kr = 0.0, 0
        else:
            post = ep - wu; hw = min(1.0, post / ramp)
            kr = min(1.0, post / max(1, epochs - wu) * 2)
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
    # Pearson: train-fit, val-eval
    L = min(len(hm), len(hidden))
    X_tr = np.column_stack([hm[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:te], rcond=None)
    X_all = np.column_stack([hm[:L], np.ones(L)])
    h_sc = X_all @ coef
    r_all = float(np.corrcoef(h_sc, hidden[:L])[0, 1])
    r_val = float(np.corrcoef(h_sc[te:L], hidden[te:L])[0, 1]) if L > te + 2 else 0
    del model, hdyn
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return r_all, r_val


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_maizuru")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, days = load_maizuru()
    species = [str(s) for s in species]
    sp_only = [s for s in species if s not in ['surf_temp', 'bot_temp']]
    n_species = len(sp_only)

    print(f"Maizuru: {n_species} species, temp as input-only")
    print(f"n_recon_channels = {n_species} (temp excluded from loss)")
    print(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n")

    results = {sp: [] for sp in sp_only}
    total = len(sp_only) * len(SEEDS)
    ri = 0

    for h_name in sp_only:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        n_rc = n_species - 1  # recon on species only
        print(f"--- hidden={h_name} ---")
        for seed in SEEDS:
            ri += 1
            t0 = datetime.now()
            r_all, r_val = train_one(visible, hidden, seed, device, n_rc)
            dt = (datetime.now() - t0).total_seconds()
            print(f"  [{ri}/{total}] seed={seed}  all={r_all:+.3f}  val={r_val:+.3f}  ({dt:.1f}s)")
            results[h_name].append({"seed": seed, "r_all": r_all, "r_val": r_val})

    print(f"\n{'='*70}")
    print(f"{'Species':<35} {'P(all)':>10} {'P(val)':>10}")
    print('-' * 70)
    all_a = []; all_v = []
    for sp in sp_only:
        rs = results[sp]
        ma = np.mean([r["r_all"] for r in rs])
        mv = np.mean([r["r_val"] for r in rs])
        all_a.append(ma); all_v.append(mv)
        print(f"{sp:<35} {ma:>+10.3f} {mv:>+10.3f}")
    print('-' * 70)
    print(f"{'Overall':<35} {np.mean(all_a):>+10.3f} {np.mean(all_v):>+10.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Maizuru Bay: Eco-GNRD\n\n")
        f.write(f"{n_species} species, temp input-only, {EPOCHS}ep, hdyn lam={LAM_HDYN}\n\n")
        f.write("| Species | P(all) | P(val) |\n|---|---|---|\n")
        for sp in sp_only:
            rs = results[sp]
            ma = np.mean([r["r_all"] for r in rs])
            mv = np.mean([r["r_val"] for r in rs])
            f.write(f"| {sp} | {ma:+.3f} | {mv:+.3f} |\n")
        f.write(f"\n**Overall**: P(all)={np.mean(all_a):+.3f}, P(val)={np.mean(all_v):+.3f}\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

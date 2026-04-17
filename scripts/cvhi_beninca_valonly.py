"""Beninca: NbedDyn best config, 5 seeds, report val-only Pearson.

For fair comparison with supervised ceiling.
Also reports all-data Pearson for reference.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_beninca_nbeddyn import (
    LatentDynamicsNet, make_model, get_param_groups,
    freeze, unfreeze, alpha_schedule, HP, PA, PB, LAM_H_ODE,
)

SEEDS = [42, 123, 456, 789, 1024]
EPOCHS = 500
SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]


def pearson_split(h, hidden, train_end):
    """Fit lstsq on train, compute Pearson on val and all."""
    L = min(len(h), len(hidden))
    h = h[:L]; hidden = hidden[:L]
    # All-data
    X_all = np.column_stack([h, np.ones(L)])
    coef_all, _, _, _ = np.linalg.lstsq(X_all, hidden, rcond=None)
    pred_all = X_all @ coef_all
    r_all = float(np.corrcoef(pred_all, hidden)[0, 1])
    # Train-fit, val-eval
    X_tr = np.column_stack([h[:train_end], np.ones(train_end)])
    coef_tr, _, _, _ = np.linalg.lstsq(X_tr, hidden[:train_end], rcond=None)
    X_val = np.column_stack([h[train_end:], np.ones(L - train_end)])
    pred_val = X_val @ coef_tr
    hid_val = hidden[train_end:]
    r_val = float(np.corrcoef(pred_val, hid_val)[0, 1]) if len(hid_val) > 2 else 0
    return r_all, r_val


def train_one(visible, hidden, seed, device, epochs=EPOCHS):
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
            rollout_weights=(1.0, 0.5, 0.25), lam_hf=0.2, lowpass_sigma=6.0, lam_rmse_log=0.1)
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
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25), lam_hf=0.2, lowpass_sigma=6.0)
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
    r_all, r_val = pearson_split(h_mean, hidden, train_end)
    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return r_all, r_val


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_valonly")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]

    print(f"Beninca NbedDyn: 5 seeds, val-only Pearson")
    print(f"Seeds: {SEEDS}\n")

    total = len(SPECIES_ORDER) * len(SEEDS)
    results = {sp: [] for sp in SPECIES_ORDER}
    ri = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        print(f"--- hidden={h_name} ---")
        for seed in SEEDS:
            ri += 1
            t0 = datetime.now()
            r_all, r_val = train_one(visible, hidden, seed, device)
            dt = (datetime.now() - t0).total_seconds()
            print(f"  [{ri}/{total}] seed={seed}  all={r_all:+.3f}  val={r_val:+.3f}  ({dt:.1f}s)")
            results[h_name].append({"seed": seed, "r_all": r_all, "r_val": r_val})

    print(f"\n{'='*80}")
    print(f"{'Species':<16} {'P(all)':>10} {'P(val)':>10}")
    print('-' * 80)
    all_a = []; all_v = []
    for sp in SPECIES_ORDER:
        rs = results[sp]
        ma = np.mean([r["r_all"] for r in rs])
        mv = np.mean([r["r_val"] for r in rs])
        sa = np.std([r["r_all"] for r in rs])
        sv = np.std([r["r_val"] for r in rs])
        all_a.append(ma); all_v.append(mv)
        print(f"{sp:<16} {ma:>+7.3f}+-{sa:.3f} {mv:>+7.3f}+-{sv:.3f}")
    print('-' * 80)
    oa = np.mean(all_a); ov = np.mean(all_v)
    print(f"{'Overall':<16} {oa:>+10.3f} {ov:>+10.3f}")
    print(f"\nSupervised ceiling (val): 0.103")
    print(f"CVHI val-only:           {ov:.3f}")
    if ov > 0:
        print(f"CVHI / ceiling:          {ov/0.103*100:.0f}%")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca NbedDyn: 5 seeds, val-only Pearson\n\n")
        f.write("| Species | P(all) | P(val) |\n|---|---|---|\n")
        for sp in SPECIES_ORDER:
            rs = results[sp]
            ma = np.mean([r["r_all"] for r in rs])
            mv = np.mean([r["r_val"] for r in rs])
            f.write(f"| {sp} | {ma:+.3f} | {mv:+.3f} |\n")
        f.write(f"\n**Overall**: P(all)={oa:+.3f}, P(val)={ov:+.3f}\n")
        f.write(f"\nSupervised ceiling (val): 0.103\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

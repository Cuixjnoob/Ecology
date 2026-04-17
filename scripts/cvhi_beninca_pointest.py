"""Point estimate h (no VAE) + NbedDyn ODE consistency + alt 5:1.

Drop VAE: encoder outputs h(t) directly, no sampling, no KL.
Keep: counterfactual null/shuffle, energy, smoothness, ODE consistency.
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate
from scripts.cvhi_beninca_nbeddyn import (
    LatentDynamicsNet, find_burst_mask, burst_precision_recall,
)


SEEDS = [42, 123, 456]
EPOCHS = 500

HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_cf=9.517725868477207,
    min_energy=0.14353013693386804,
)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

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
        point_estimate=True,      # <-- no VAE, direct h output
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


def train_one(visible, hidden, seed, device, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)

    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())

    opt = torch.optim.AdamW(all_params, lr=HP["lr"], weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))

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
        if epoch < warmup:
            h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))

        # Alternating
        if epoch >= warmup:
            cycle_pos = (epoch - warmup) % (PA + PB)
            if cycle_pos < PA:
                freeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(False)
                unfreeze(fvis_params)
            else:
                freeze(fvis_params); unfreeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(True)
        else:
            unfreeze(fvis_params); unfreeze(enc_params)
            for p in f_h.parameters(): p.requires_grad_(True)

        # Input augmentation
        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); f_h.train()
        opt.zero_grad()

        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, :train_end]
        tr_out["G"] = out["G"][:, :train_end]

        losses = model.loss(
            tr_out,
            beta_kl=0.0,           # <-- no KL (point estimate)
            free_bits=0.0,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP["lam_cf"], lam_shuffle=HP["lam_cf"] * 0.6,
            lam_energy=2.0, min_energy=HP["min_energy"],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.2, lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )

        # NbedDyn ODE consistency
        loss_h_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and LAM_H_ODE > 0:
            h_mu = out["mu"][:, :train_end]
            x_vis = out["visible"][:, :train_end]
            h_pred = f_h(h_mu[:, :-1], x_vis[:, :-1])
            loss_h_ode = F.mse_loss(h_pred, h_mu[:, 1:].detach())

        total = losses["total"] + LAM_H_ODE * h_w * loss_h_ode
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step(); sched.step()

        # Validation
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_out["visible"] = out["visible"][:, train_end:T]
            val_out["G"] = out["G"][:, train_end:T]
            val_losses = model.loss(
                val_out, beta_kl=0.0, h_weight=1.0,
                margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=HP["min_energy"],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.2, lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()

        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    # Eval
    unfreeze(fvis_params); unfreeze(enc_params)
    for p in f_h.parameters(): p.requires_grad_(True)
    if best_state: model.load_state_dict(best_state)
    if best_fh: f_h.load_state_dict(best_fh)
    model.eval(); f_h.eval()

    with torch.no_grad():
        out_eval = model(x_full, n_samples=1, rollout_K=3)
        # point estimate: h_samples shape (1, B, T), just take [0,0]
        h_mean = out_eval["h_samples"][0, 0].cpu().numpy()

    pear, h_scaled = evaluate(h_mean, hidden)
    burst_m = burst_precision_recall(h_scaled.flatten(), hidden, pct=10)

    with torch.no_grad():
        vis = out_eval["visible"]
        safe_v = torch.clamp(vis, min=1e-6)
        log_ratio = torch.log(safe_v[:, 1:] / safe_v[:, :-1]).unsqueeze(0)
        recon_full_val = ((out_eval["pred_full"] - log_ratio) ** 2).mean().item()
        recon_null_val = ((out_eval["pred_null"] - log_ratio[0]) ** 2).mean().item()
    margin = recon_null_val - recon_full_val

    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"pearson": pear, "val_recon": best_val,
            "margin": margin, **burst_m}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_pointest")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, _ = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]

    print(f"Point estimate (no VAE) + NbedDyn ODE (lam={LAM_H_ODE}) + alt {PA}:{PB}")
    print(f"No KL, no sampling. h = encoder output directly.\n")

    total_runs = len(SPECIES_ORDER) * len(SEEDS)
    results = {sp: [] for sp in SPECIES_ORDER}
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        print(f"--- hidden={h_name} ---")
        for seed in SEEDS:
            run_i += 1
            t0 = datetime.now()
            try:
                r = train_one(visible, hidden_raw, seed, device)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  [{run_i}/{total_runs}] seed={seed}  "
                      f"P={r['pearson']:+.3f}  "
                      f"burst_F={r['burst_f_score']:.3f}  "
                      f"margin={r['margin']:.4f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "val_recon": float("nan"),
                     "margin": float("nan"),
                     "burst_precision": 0, "burst_recall": 0, "burst_f_score": 0}
            r["seed"] = seed
            results[h_name].append(r)

    # Summary
    print(f"\n{'='*100}")
    print(f"RESULTS: Point Estimate + NbedDyn + Alt 5:1")
    print('='*100)
    print(f"{'Species':<16} {'Pearson':>10} {'Burst_F':>10} {'Margin':>10}")
    print('-'*100)

    all_pear = []; all_bf = []
    for sp in SPECIES_ORDER:
        rs = results[sp]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        BF = np.array([r["burst_f_score"] for r in rs])
        MG = np.array([r["margin"] for r in rs if not np.isnan(r.get("margin", float("nan")))])
        mp = float(P.mean()) if len(P) else float("nan")
        mbf = float(BF.mean()) if len(BF) else 0
        mmg = float(MG.mean()) if len(MG) else float("nan")
        if not np.isnan(mp): all_pear.append(mp)
        all_bf.append(mbf)
        print(f"{sp:<16} {mp:>+10.3f} {mbf:>10.3f} {mmg:>10.4f}")

    print('-'*100)
    op = np.mean(all_pear) if all_pear else float("nan")
    obf = np.mean(all_bf)
    print(f"{'Overall':<16} {op:>+10.4f} {obf:>10.3f}")
    print(f"\nRef: NbedDyn(VAE)=+0.1620, alt_5_1=+0.1595")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Point Estimate + NbedDyn + Alt 5:1\n\n")
        f.write(f"No VAE, no KL, no sampling. h = encoder output.\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}, Alt: {PA}:{PB}, lam_h_ode={LAM_H_ODE}\n\n")
        f.write("| Species | Pearson | Burst_F | Margin |\n|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            rs = results[sp]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            BF = np.array([r["burst_f_score"] for r in rs])
            MG = np.array([r["margin"] for r in rs if not np.isnan(r.get("margin", float("nan")))])
            f.write(f"| {sp} | {P.mean():+.3f} | {BF.mean():.3f} | {MG.mean():.4f} |\n")
        f.write(f"\n**Overall**: Pearson={op:+.4f}, Burst_F={obf:.3f}\n")
        f.write(f"\nRef: NbedDyn(VAE)=+0.1620, alt_5_1=+0.1595\n")

    raw = {}
    for sp in SPECIES_ORDER:
        raw[sp] = [{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                     for k, v in r.items()} for r in results[sp]]
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

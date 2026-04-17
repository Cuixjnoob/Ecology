"""Encoder boost: extend Takens lags to give encoder more temporal context.

Supervised oracle shows:
  Ridge (linear): +0.14  ≈ our CVHI
  RF:             +0.60
  GBR:            +0.71

Current encoder lags = (1,2,4,8), max 32 days = 1 oscillation.
Extended lags = (1,2,4,8,16,32), max 128 days = 4 oscillations.

Configs:
  A. baseline: lags=(1,2,4,8), d=96, blocks=3
  B. ext_lags: lags=(1,2,4,8,16,32), d=96, blocks=3
  C. ext_big:  lags=(1,2,4,8,16,32), d=128, blocks=4
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate


SEEDS = [42, 123, 456]
EPOCHS = 500

BASE_HP = dict(
    lr=0.0006033475528697158,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

# (name, takens_lags, encoder_d, encoder_blocks)
CONFIGS = [
    ("baseline",  (1,2,4,8),         96,  3),
    ("ext_lags",  (1,2,4,8,16,32),   96,  3),
    ("ext_big",   (1,2,4,8,16,32),   128, 4),
]


def make_model(N, device, lags, enc_d, enc_blocks):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=enc_d, encoder_blocks=enc_blocks,
        encoder_heads=4,
        takens_lags=lags, encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden, seed, device, lags, enc_d, enc_blocks, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device, lags, enc_d, enc_blocks)
    opt = torch.optim.AdamW(model.parameters(), lr=BASE_HP["lr"], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w
        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full
        model.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]
        losses = model.loss(
            tr_out, beta_kl=BASE_HP["lam_kl"], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=BASE_HP["lam_cf"], lam_shuffle=BASE_HP["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=BASE_HP["min_energy"],
            lam_smooth=BASE_HP["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=BASE_HP["lam_hf"], lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_out["visible"] = out["visible"][:, train_end:T]
            val_out["G"] = out["G"][:, train_end:T]
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=BASE_HP["min_energy"],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=BASE_HP["lam_hf"], lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pearson": pear, "val_recon": best_val}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_encoder_boost")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== Encoder boost experiment ===")
    for name, lags, d, b in CONFIGS:
        feat_dim = 2 + 2 * len(lags)
        print(f"  {name}: lags={lags}, d={d}, blocks={b}, feat_dim={feat_dim}")
    total_runs = len(SPECIES_ORDER) * len(CONFIGS) * len(SEEDS)
    print(f"Total: {total_runs} runs\n")

    results = {c[0]: {sp: [] for sp in SPECIES_ORDER} for c in CONFIGS}
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        print(f"\n--- hidden={h_name} ---")
        for cfg_name, lags, enc_d, enc_blocks in CONFIGS:
            for seed in SEEDS:
                run_i += 1
                t0 = datetime.now()
                try:
                    r = train_one(visible, hidden_raw, seed, device,
                                  lags, enc_d, enc_blocks)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  [{run_i}/{total_runs}] {cfg_name:<12} seed={seed}  "
                          f"P={r['pearson']:+.3f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "val_recon": float("nan")}
                r["seed"] = seed; r["config"] = cfg_name
                results[cfg_name][h_name].append(r)

    # Summary
    print(f"\n{'='*80}")
    print("ENCODER BOOST RESULTS")
    print('='*80)
    header = f"{'Species':<16}"
    for c in CONFIGS:
        header += f"{c[0]:<16}"
    print(header)
    print('-'*80)

    config_means = {c[0]: [] for c in CONFIGS}
    for sp in SPECIES_ORDER:
        line = f"{sp:<16}"
        for c in CONFIGS:
            rs = results[c[0]][sp]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            m = float(P.mean()) if len(P) else float("nan")
            if not np.isnan(m):
                config_means[c[0]].append(m)
            line += f"{m:<+16.3f}"
        print(line)

    print('-'*80)
    avg_line = f"{'Overall':<16}"
    for c in CONFIGS:
        avg = np.mean(config_means[c[0]]) if config_means[c[0]] else float("nan")
        avg_line += f"{avg:<+16.4f}"
    print(avg_line)

    best_cfg = max(CONFIGS, key=lambda c: np.mean(config_means[c[0]]) if config_means[c[0]] else -999)
    best_avg = np.mean(config_means[best_cfg[0]])
    print(f"\nBest: {best_cfg[0]} = {best_avg:+.4f} (S1b ref: +0.132)")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Encoder boost experiment\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("| Config | lags | d | blocks | feat_dim |\n|---|---|---|---|---|\n")
        for name, lags, d, b in CONFIGS:
            f.write(f"| {name} | {lags} | {d} | {b} | {2+2*len(lags)} |\n")
        f.write(f"\n| Species |" + "|".join(f" {c[0]} " for c in CONFIGS) + "|\n")
        f.write("|---|" + "---|" * len(CONFIGS) + "\n")
        for sp in SPECIES_ORDER:
            row = f"| {sp}"
            for c in CONFIGS:
                rs = results[c[0]][sp]
                P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
                m = float(P.mean()) if len(P) else float("nan")
                row += f" | {m:+.3f}"
            f.write(row + " |\n")
        f.write(f"\n**Overall**: " + ", ".join(
            f"{c[0]}={np.mean(config_means[c[0]]):+.4f}" for c in CONFIGS) + "\n")
        f.write(f"\nBest: **{best_cfg[0]}** = {best_avg:+.4f} (S1b ref: +0.132)\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

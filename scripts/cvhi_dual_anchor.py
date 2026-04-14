"""Path A''': 双向 anchor + val_recon 选方向.

每 seed 训练两份:
  model_pos: G_anchor_sign = +1  (pin G[0] ≥ 0)
  model_neg: G_anchor_sign = -1  (pin G[0] ≤ 0)
评测时选 val_recon 更低的那份 (严格无监督, val_recon 不用 hidden).
这样彻底消除 "convention 方向" 的 luck factor.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib, hidden_true_substitution,
)
from scripts.cvhi_residual_mendota import load_lake_mendota


SEEDS_20 = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
             7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def load_mendota_xtroph():
    matrix, _, _, _, _ = load_lake_mendota()
    hid = matrix[:, 1]
    vis = np.delete(matrix, 1, axis=1)
    return vis.astype(np.float32), hid.astype(np.float32)


def make_portal(N, sign, device, anneal=True):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=sign,
    ).to(device)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden_eval, device, seed, sign, epochs=300, use_anneal=True):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_portal(N, sign, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None
    m_null, m_shuf, min_e = 0.002, 0.001, 0.05

    for epoch in range(epochs):
        model.G_anchor_alpha = alpha_schedule(epoch, epochs) if use_anneal else 1.0
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / (epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w
        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        losses = model.loss(tr_out, beta_kl=0.03, free_bits=0.02,
                            margin_null=m_null, margin_shuf=m_shuf,
                            lam_necessary=5.0, lam_shuffle=3.0,
                            lam_energy=2.0, min_energy=min_e,
                            lam_smooth=0.02, lam_sparse=0.02,
                            h_weight=h_w, lam_rollout=lam_r,
                            rollout_weights=(1.0, 0.5, 0.25),
                            lam_hf=0.0, lowpass_sigma=6.0)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=5.0, lam_shuffle=3.0,
                lam_energy=2.0, min_energy=min_e,
                lam_rollout=lam_r, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0)
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden_eval)
    diag = hidden_true_substitution(model, visible, hidden_eval, device)
    d_ratio = diag["recon_true_scaled"] / diag["recon_encoder"]
    return {"pearson": pear, "val_recon": best_val, "d_ratio": d_ratio}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=20)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--dataset", type=str, default="Portal", choices=["Portal", "Mendota"])
    ap.add_argument("--use_anneal", type=int, default=1)
    args = ap.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_dual_anchor_{args.dataset}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if args.dataset == "Portal":
        vis, hid = load_portal("OT")
    else:
        vis, hid = load_mendota_xtroph()
    seeds = SEEDS_20[:args.n_seeds]

    # 训练两份 × n_seeds
    rows = []
    for i, s in enumerate(seeds):
        t0 = datetime.now()
        r_pos = train_one(vis, hid, device, s, sign=+1, epochs=args.epochs, use_anneal=bool(args.use_anneal))
        t_pos = (datetime.now() - t0).total_seconds()
        t1 = datetime.now()
        r_neg = train_one(vis, hid, device, s, sign=-1, epochs=args.epochs, use_anneal=bool(args.use_anneal))
        t_neg = (datetime.now() - t1).total_seconds()
        # val-select: lower val_recon 胜
        if r_pos["val_recon"] <= r_neg["val_recon"]:
            winner = "+1"
            chosen = r_pos; rejected = r_neg
        else:
            winner = "-1"
            chosen = r_neg; rejected = r_pos
        print(f"  [{i+1}/{len(seeds)}] seed={s}  "
               f"+1: P={r_pos['pearson']:+.3f}/val={r_pos['val_recon']:.4f}  "
               f"-1: P={r_neg['pearson']:+.3f}/val={r_neg['val_recon']:.4f}  "
               f"→ chose {winner}, P_chosen={chosen['pearson']:+.3f}  "
               f"(pos {t_pos:.1f}s + neg {t_neg:.1f}s)")
        rows.append(dict(seed=s, sign_chosen=winner,
                         P_pos=r_pos["pearson"], val_pos=r_pos["val_recon"], d_pos=r_pos["d_ratio"],
                         P_neg=r_neg["pearson"], val_neg=r_neg["val_recon"], d_neg=r_neg["d_ratio"],
                         P_chosen=chosen["pearson"], val_chosen=chosen["val_recon"], d_chosen=chosen["d_ratio"]))

    # 汇总
    P_chosen = np.array([r["P_chosen"] for r in rows])
    P_pos_only = np.array([r["P_pos"] for r in rows])
    P_neg_only = np.array([r["P_neg"] for r in rows])
    P_oracle = np.array([max(r["P_pos"], r["P_neg"]) for r in rows])  # oracle: pick by Pearson
    n_chose_pos = sum(1 for r in rows if r["sign_chosen"] == "+1")

    print(f"\n{'='*70}\nSUMMARY ({args.dataset}, {args.n_seeds} seeds, anneal={args.use_anneal})\n{'='*70}")
    print(f"pos_only       mean P = {P_pos_only.mean():+.4f} ± {P_pos_only.std(ddof=1):.4f}")
    print(f"neg_only       mean P = {P_neg_only.mean():+.4f} ± {P_neg_only.std(ddof=1):.4f}")
    print(f"chosen (dual)  mean P = {P_chosen.mean():+.4f} ± {P_chosen.std(ddof=1):.4f}  "
           f"(chose +1 in {n_chose_pos}/{len(seeds)})")
    print(f"oracle (cheat) mean P = {P_oracle.mean():+.4f} ± {P_oracle.std(ddof=1):.4f}")

    from scipy import stats
    t, p = stats.ttest_rel(P_chosen, P_pos_only)
    print(f"\npaired t: chosen vs pos_only:  Δ={P_chosen.mean()-P_pos_only.mean():+.4f}, t={t:+.3f}, p={p:.4g}")
    t, p = stats.ttest_rel(P_chosen, P_oracle)
    print(f"paired t: chosen vs oracle:    Δ={P_chosen.mean()-P_oracle.mean():+.4f}, t={t:+.3f}, p={p:.4g}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Dual-Anchor + val-select 实验\n\n")
        f.write(f"- dataset: {args.dataset}\n- seeds: {len(seeds)}  epochs: {args.epochs}\n\n")
        f.write("## 主表\n\n| method | mean P | std | notes |\n|---|---|---|---|\n")
        f.write(f"| pos_only (单向 +1) | {P_pos_only.mean():+.4f} | {P_pos_only.std(ddof=1):.4f} | 上次 SOTA |\n")
        f.write(f"| neg_only (单向 -1) | {P_neg_only.mean():+.4f} | {P_neg_only.std(ddof=1):.4f} | 对照 |\n")
        f.write(f"| **chosen (dual+val_recon)** | **{P_chosen.mean():+.4f}** | {P_chosen.std(ddof=1):.4f} | 无监督选 |\n")
        f.write(f"| oracle (pick by Pearson) | {P_oracle.mean():+.4f} | {P_oracle.std(ddof=1):.4f} | 作弊上限 |\n\n")
        f.write(f"Chose +1 in {n_chose_pos}/{len(seeds)} seeds.\n\n")
        f.write("## Per-seed\n\n| seed | P_+1 | val_+1 | P_-1 | val_-1 | chose | P_chosen |\n|---|---|---|---|---|---|---|\n")
        for r in rows:
            f.write(f"| {r['seed']} | {r['P_pos']:+.3f} | {r['val_pos']:.4f} | "
                     f"{r['P_neg']:+.3f} | {r['val_neg']:.4f} | {r['sign_chosen']} | {r['P_chosen']:+.3f} |\n")

    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

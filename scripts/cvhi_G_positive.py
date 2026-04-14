"""路 A: G(x) ≥ 0 约束实验, 破 ±h 对称.

对比 configs:
  baseline   : G 无约束 (原版, 允许正负)
  G_positive : G = softplus(raw_G) ≥ 0 硬约束
  (optional) K=2  MoG  — 后续

5 / 20 seeds × Portal × 2 configs.
同时记录 pairwise corr 分布 → 看 sign ambiguity 是否被消除.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib, hidden_true_substitution,
)
from scripts.cvhi_residual_synthetic_comparison import load_lv, load_holling


def load_mendota_xtroph():
    """Mendota cross-trophic: hidden = Microcystis aeruginosa."""
    from scripts.cvhi_residual_mendota import load_lake_mendota
    matrix, _, _, _, _ = load_lake_mendota()
    hid = matrix[:, 1]   # Microcystis = idx 1
    vis = np.delete(matrix, 1, axis=1)
    return vis.astype(np.float32), hid.astype(np.float32)


SEEDS_20 = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
             7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]

CONFIGS = {
    "baseline":         dict(G_positive=False, G_anchor_first=False, use_coupling_weight=False, use_coupling_attn=False),
    "G_positive":       dict(G_positive=True,  G_anchor_first=False, use_coupling_weight=False, use_coupling_attn=False),
    "G_anchor_first":   dict(G_positive=False, G_anchor_first=True,  use_coupling_weight=False, use_coupling_attn=False),
    "anchor+coupling":  dict(G_positive=False, G_anchor_first=True,  use_coupling_weight=True,  use_coupling_attn=False),
    "coupling_only":    dict(G_positive=False, G_anchor_first=False, use_coupling_weight=True,  use_coupling_attn=False),
    # Option 3 B+C: attention bias
    "anchor+coup_attn": dict(G_positive=False, G_anchor_first=True,  use_coupling_weight=False, use_coupling_attn=True),
    "coup_attn_only":   dict(G_positive=False, G_anchor_first=False, use_coupling_weight=False, use_coupling_attn=True),
    # Part B: Residual-driven attention
    "anchor+res_attn":  dict(G_positive=False, G_anchor_first=True,  use_coupling_weight=False, use_coupling_attn=False, use_residual_attn=True),
    "res_attn_only":    dict(G_positive=False, G_anchor_first=False, use_coupling_weight=False, use_coupling_attn=False, use_residual_attn=True),
}


def make_model(N, cfg, is_portal, device):
    common = dict(
        num_visible=N, prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_positive=cfg["G_positive"],
        G_anchor_first=cfg["G_anchor_first"],
        use_coupling_weight=cfg.get("use_coupling_weight", False),
        use_coupling_attn=cfg.get("use_coupling_attn", False),
        use_residual_attn=cfg.get("use_residual_attn", False),
        coupling_top_k=3,
    )
    if is_portal:
        return CVHI_Residual(
            encoder_d=48, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
            d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=12, G_field_layers=1, G_field_top_k=3,
            **common,
        ).to(device)
    return CVHI_Residual(
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=24, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=16, G_field_layers=1, G_field_top_k=3,
        **common,
    ).to(device)


def train_one(visible, hidden_eval, device, seed, cfg, epochs=300, is_portal=False):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, cfg, is_portal, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None
    if is_portal:
        m_null, m_shuf, min_e = 0.002, 0.001, 0.05
    else:
        m_null, m_shuf, min_e = 0.003, 0.002, 0.02
    for epoch in range(epochs):
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
        losses = model.loss(
            tr_out, beta_kl=0.03, free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_e,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
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
                lam_hf=0.0, lowpass_sigma=6.0,
            )
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
    return {"seed": seed, "pearson": pear, "val_recon": best_val,
            "d_ratio": d_ratio, "h_mean": h_mean}


def aggregate(pears):
    arr = np.asarray(pears); n = len(arr)
    mean = float(arr.mean()); std = float(arr.std(ddof=1))
    from scipy import stats
    se = std / np.sqrt(n); tc = stats.t.ppf(0.975, n - 1)
    return dict(n=n, mean=mean, std=std, ci_lo=mean - tc * se, ci_hi=mean + tc * se,
                max=float(arr.max()), min=float(arr.min()))


def pairwise_corr_stats(H):
    Hc = H - H.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(Hc, axis=1, keepdims=True)
    pc = (Hc @ Hc.T) / (norms @ norms.T + 1e-12)
    iu = np.triu_indices(len(H), k=1)
    vals = pc[iu]
    return dict(mean=float(vals.mean()), std=float(vals.std()),
                median=float(np.median(vals)),
                frac_pos05=float((vals > 0.5).mean()),
                frac_neg03=float((vals < -0.3).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--datasets", nargs="+", default=["LV", "Holling", "Portal"])
    ap.add_argument("--configs", nargs="+", default=list(CONFIGS.keys()))
    args = ap.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_G_positive")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS_20[:args.n_seeds]

    loaders = {
        "LV":      (load_lv, False),
        "Holling": (load_holling, False),
        "Portal":  (lambda: load_portal("OT"), True),
        "Mendota": (load_mendota_xtroph, True),   # 大 T + 真实数据, 当作 Portal 组配置
    }

    results = {c: {d: [] for d in args.datasets} for c in args.configs}
    total = len(args.configs) * len(args.datasets) * len(seeds); i = 0
    for cfg_name in args.configs:
        cfg = CONFIGS[cfg_name]
        print(f"\n{'='*70}\n{cfg_name}  {cfg}\n{'='*70}")
        for ds in args.datasets:
            loader, is_p = loaders[ds]
            vis, hid = loader()
            for s in seeds:
                i += 1; t0 = datetime.now()
                r = train_one(vis, hid, device, s, cfg, epochs=args.epochs, is_portal=is_p)
                dt = (datetime.now() - t0).total_seconds()
                results[cfg_name][ds].append(r)
                print(f"  [{i}/{total}] {cfg_name}/{ds}/seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")

    # 汇总 + 对比
    summary = {}
    for c in args.configs:
        summary[c] = {}
        for ds in args.datasets:
            rs = results[c][ds]
            H = np.array([r["h_mean"] for r in rs])
            summary[c][ds] = {
                "pearson": aggregate([r["pearson"] for r in rs]),
                "d_ratio": aggregate([r["d_ratio"] for r in rs]),
                "pairwise": pairwise_corr_stats(H),
            }
    print(f"\n{'='*100}")
    print(f"SUMMARY: G ≥ 0 constraint effect")
    print(f"{'='*100}")
    print(f"{'config':<12s}{'dataset':<10s}{'mean P':<12s}{'std':<8s}{'95% CI':<22s}"
           f"{'max':<8s}{'pair_μ':<10s}{'pair σ':<8s}{'%>0.5':<8s}{'%<-0.3':<8s}")
    for c in args.configs:
        for ds in args.datasets:
            s = summary[c][ds]
            p = s["pearson"]; pw = s["pairwise"]
            print(f"{c:<12s}{ds:<10s}{p['mean']:<+12.4f}{p['std']:<8.4f}"
                   f"[{p['ci_lo']:+.3f}, {p['ci_hi']:+.3f}]    "
                   f"{p['max']:<+8.3f}{pw['mean']:<+10.3f}{pw['std']:<8.3f}"
                   f"{pw['frac_pos05']*100:<8.0f}{pw['frac_neg03']*100:<8.0f}")
        print()

    print(f"{'Δ vs baseline':=^60}")
    for c in args.configs:
        if c == "baseline": continue
        for ds in args.datasets:
            b = summary["baseline"][ds]["pearson"]["mean"]
            t = summary[c][ds]["pearson"]["mean"]
            bpw = summary["baseline"][ds]["pairwise"]["mean"]
            tpw = summary[c][ds]["pairwise"]["mean"]
            print(f"  {c:<12s}{ds:<10s}: ΔP = {t - b:+.4f}  Δpair_mean_corr = {tpw - bpw:+.3f}")

    # save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Path A: G ≥ 0 约束实验\n\n- seeds: {len(seeds)}  epochs: {args.epochs}\n\n")
        f.write("## Per-config Pearson + pairwise corr stats\n\n")
        f.write("| cfg | ds | mean P | std | max | pair μ | pair σ | %>0.5 | %<-0.3 |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for c in args.configs:
            for ds in args.datasets:
                s = summary[c][ds]; p = s["pearson"]; pw = s["pairwise"]
                f.write(f"| {c} | {ds} | {p['mean']:+.4f} | {p['std']:.4f} | "
                         f"{p['max']:+.3f} | {pw['mean']:+.3f} | {pw['std']:.3f} | "
                         f"{pw['frac_pos05']*100:.0f}% | {pw['frac_neg03']*100:.0f}% |\n")
        f.write("\n## Δ vs baseline\n\n| cfg | ds | ΔP | Δpair_mean_corr |\n|---|---|---|---|\n")
        for c in args.configs:
            if c == "baseline": continue
            for ds in args.datasets:
                b = summary["baseline"][ds]["pearson"]["mean"]
                t = summary[c][ds]["pearson"]["mean"]
                bpw = summary["baseline"][ds]["pairwise"]["mean"]
                tpw = summary[c][ds]["pairwise"]["mean"]
                f.write(f"| {c} | {ds} | {t-b:+.4f} | {tpw-bpw:+.3f} |\n")

    raw = {c: {ds: [{k: (float(v) if isinstance(v, (int, float, np.floating)) else
                          (v.tolist() if isinstance(v, np.ndarray) else v))
                      for k, v in r.items()}
                     for r in rs] for ds, rs in d.items()}
            for c, d in results.items()}
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({"summary": summary, "raw": raw}, f, indent=2, default=float)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

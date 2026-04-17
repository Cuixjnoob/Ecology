"""深度研究: encoder_h 的 Holling-saturation 补偿项.

Experiments:
  E1: 减去补偿后 Pearson 上限
  E2: 识别 x_1 的生态身份 + hidden 身份
  E4: 补偿的时间结构 (residual vs x_1_sat overlay)

Holling seed=123, d_ratio=22.8, corr(residual, x_1/(1+x_1)) = -0.84
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_synthetic_comparison import load_lv, load_holling
from scripts.train_utils_fast import train_one_fast


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def extract_h(model, visible, device):
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        enc_out = model.encoder(x)
        if len(enc_out) == 2:
            mu_k, _ = enc_out
            return mu_k[..., 0].cpu().numpy()[0]
        else:
            mu_K, _, logits_K = enc_out
            mu_K = mu_K[..., 0]; logits_K = logits_K[..., 0]
            pi = F.softmax(logits_K, dim=-1)
            return (pi * mu_K).sum(-1).cpu().numpy()[0]


def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return float((a * b).sum() / (np.sqrt((a*a).sum() * (b*b).sum()) + 1e-12))


def e1_subtraction_ceiling(eh, th, vis, name):
    """Subtract best β·feature from eh, see if Pearson(h_clean, th) improves."""
    # Scale eh to true_h std
    eh_c = eh - eh.mean()
    th_c = th - th.mean()
    eh_sc = eh_c * (th_c.std() / (eh_c.std() + 1e-8))

    # Candidate compensation features
    N = vis.shape[1]
    feats = {}
    for j in range(N):
        x_j = vis[:, j]
        feats[f"x_{j}"] = x_j - x_j.mean()
        feats[f"x_{j}/(1+x_{j})"] = x_j / (1 + x_j)
        feats[f"x_{j}^2"] = (x_j ** 2) - (x_j ** 2).mean()
    # Multi-variable OLS subtraction: try best single feature subtraction
    baseline_P = pearson(eh_sc, th_c)
    results = []
    for fk, fv in feats.items():
        fv = fv - fv.mean()
        if fv.std() < 1e-8:
            continue
        # Best β such that eh_sc - β·fv has max |Pearson| with th_c
        # Closed form: dPearson/dβ = 0 → β* for max corr
        # Use simple grid search (fast enough)
        betas = np.linspace(-2.0, 2.0, 81)
        best_P = baseline_P
        best_b = 0.0
        for b in betas:
            h_clean = eh_sc - b * fv
            P = pearson(h_clean, th_c)
            if abs(P) > abs(best_P):
                best_P = P
                best_b = float(b)
        results.append((fk, best_P, best_b))

    results.sort(key=lambda r: -abs(r[1]))
    print(f"\n{name}: E1 subtraction ceiling")
    print(f"  baseline Pearson(eh_sc, th): {baseline_P:+.4f}")
    for fk, P, b in results[:5]:
        print(f"  subtract {b:+.3f}·{fk:25s} → Pearson = {P:+.4f}  (Δ={P - baseline_P:+.4f})")

    return {"baseline": baseline_P, "best_single": results[0],
            "all_results": results}


def e4_time_structure(eh, th, vis, name, out_dir):
    """Plot residual(t) and compensation feature(t) overlay."""
    eh_c = eh - eh.mean(); th_c = th - th.mean()
    eh_sc = eh_c * (th_c.std() / (eh_c.std() + 1e-8))
    a, b = np.polyfit(th_c, eh_sc, 1)
    residual = eh_sc - (a * th_c + b)

    # Compute all saturation features, find top
    N = vis.shape[1]
    best_k = None; best_c = 0
    for j in range(N):
        xj_sat = vis[:, j] / (1 + vis[:, j])
        xj_sat = xj_sat - xj_sat.mean()
        c = pearson(residual, xj_sat)
        if abs(c) > abs(best_c):
            best_c = c; best_k = j

    xstar = vis[:, best_k] / (1 + vis[:, best_k])
    xstar_c = xstar - xstar.mean()
    # Scale xstar to residual's scale
    xstar_scaled = xstar_c * (residual.std() / (xstar_c.std() + 1e-8))
    # Sign correction
    if best_c < 0:
        xstar_scaled = -xstar_scaled

    T = len(residual)
    t = np.arange(T)

    fig, axes = plt.subplots(3, 1, figsize=(13, 9), constrained_layout=True,
                              sharex=True)
    # Panel 1: true h, encoder_h (scaled), compensation feature
    ax = axes[0]
    ax.plot(t, th[:T], color="black", lw=2, label="true h(t)")
    ax.plot(t, eh_sc, color="#1976d2", lw=1.2, alpha=0.8, label="encoder h (scaled)")
    ax.set_ylabel("h"); ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    ax.set_title(f"{name}: h over time")

    # Panel 2: residual vs flipped-sign compensation feature
    ax = axes[1]
    sign_lbl = "-" if best_c < 0 else "+"
    ax.plot(t, residual, color="#c62828", lw=1.3, label="residual = eh_scaled − linear(th)")
    ax.plot(t, xstar_scaled, color="#2e7d32", lw=1.3, linestyle="--",
            label=f"{sign_lbl}x_{best_k}/(1+x_{best_k}) (scaled)")
    ax.set_ylabel("signal")
    ax.legend(loc="upper right"); ax.grid(alpha=0.3)
    ax.set_title(f"residual vs Holling-saturation feature (corr={best_c:+.3f})")

    # Panel 3: x_best_k state over time + |residual|
    ax = axes[2]
    ax.plot(t, vis[:T, best_k], color="#2e7d32", lw=1.3, label=f"x_{best_k}(t) [state]")
    axb = ax.twinx()
    axb.plot(t, np.abs(residual), color="#c62828", lw=0.8, alpha=0.6,
             label="|residual|")
    ax.set_xlabel("time")
    ax.set_ylabel(f"x_{best_k}", color="#2e7d32")
    axb.set_ylabel("|residual|", color="#c62828")
    ax.legend(loc="upper left"); axb.legend(loc="upper right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"E4: compensation time structure — {name}", fontweight="bold")
    fig.savefig(out_dir / f"E4_{name}_time.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

    # Is |residual| larger at high-x or low-x?
    high_x_mask = vis[:, best_k] > np.median(vis[:, best_k])
    res_hi = np.abs(residual[high_x_mask]).mean()
    res_lo = np.abs(residual[~high_x_mask]).mean()

    return {
        "best_k": int(best_k),
        "corr_residual_satk": float(best_c),
        "mean_abs_residual_at_high_x": float(res_hi),
        "mean_abs_residual_at_low_x": float(res_lo),
        "hi_lo_ratio": float(res_hi / (res_lo + 1e-8)),
    }


def e2_identify_species(vis, hid, name):
    """Describe species statistics to understand x_1's role."""
    N = vis.shape[1]
    stats = []
    for j in range(N):
        x_j = vis[:, j]
        stats.append({
            "species_idx": j,
            "mean": float(x_j.mean()),
            "std": float(x_j.std()),
            "max": float(x_j.max()),
            "range_ratio": float(x_j.max() / (x_j.min() + 1e-8)),
            "sat_nonlinearity": float(np.std(x_j / (1 + x_j)) / (x_j.std() / x_j.mean() + 1e-8)),
            "corr_w_hidden": pearson(x_j, hid[:len(x_j)]),
            "corr_w_hidden_sat": pearson(x_j / (1 + x_j), hid[:len(x_j)]),
        })
    print(f"\n{name}: E2 species identity")
    print(f"  hidden: mean={hid.mean():.3f}  std={hid.std():.3f}  max={hid.max():.3f}")
    print(f"  {'idx':<5}{'mean':<9}{'std':<9}{'max':<9}"
          f"{'range_r':<10}{'corr(h)':<11}{'corr(sat,h)':<12}")
    for s in stats:
        print(f"  {s['species_idx']:<5}{s['mean']:<9.3f}{s['std']:<9.3f}{s['max']:<9.3f}"
              f"{s['range_ratio']:<10.1f}{s['corr_w_hidden']:<+11.3f}"
              f"{s['corr_w_hidden_sat']:<+12.3f}")
    return stats


def run_one(name, visible, hidden, seed, device, out_dir):
    print(f"\n{'='*72}\n{name}  seed={seed}  (T={visible.shape[0]}, N={visible.shape[1]})\n{'='*72}")
    torch.manual_seed(seed)
    model = make_model(visible.shape[1], device)
    r = train_one_fast(
        model, visible, hidden, device=device,
        epochs=300, lr=0.0008,
        beta_kl=0.03, free_bits=0.02,
        margin_null=0.003, margin_shuf=0.002,
        lam_necessary=5.0, lam_shuffle=3.0,
        lam_energy=2.0, min_energy=0.02,
        lam_smooth=0.02, lam_sparse=0.02,
        lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
        lam_hf=0.0,
        lam_rmse_log=0.1, input_dropout_prob=0.05,
        use_compile=True, use_ema=False, use_snapshot_ensemble=False,
    )
    print(f"Trained: Pearson={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}")

    eh = extract_h(model, visible, device)
    th = hidden[:len(eh)]
    vis = visible[:len(eh)]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    e2 = e2_identify_species(vis, th, name)
    e1 = e1_subtraction_ceiling(eh, th, vis, name)
    e4 = e4_time_structure(eh, th, vis, name, out_dir)

    return {
        "train": {"pearson": r["pearson"], "d_ratio": r["d_ratio"]},
        "e2": e2, "e1": e1, "e4": e4,
        "encoder_h": eh.tolist(), "true_h": th.tolist(),
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_deepdive_compensation")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    lv_vis, lv_hid = load_lv()
    ho_vis, ho_hid = load_holling()

    results["LV"] = run_one("LV", lv_vis, lv_hid, seed=42, device=device, out_dir=out_dir)
    results["Holling"] = run_one("Holling", ho_vis, ho_hid, seed=123, device=device, out_dir=out_dir)

    # Also run a different Holling seed to check robustness
    results["Holling_seed42"] = run_one("Holling_seed42", ho_vis, ho_hid, seed=42,
                                         device=device, out_dir=out_dir)

    # Write report
    with open(out_dir / "analysis.md", "w", encoding="utf-8") as f:
        f.write("# Deep-dive: encoder_h 补偿项分析\n\n")
        f.write("## E1: 减去补偿后 Pearson 能否提升?\n\n")
        for k, r in results.items():
            e1 = r["e1"]
            f.write(f"### {k} (train Pearson {r['train']['pearson']:+.3f}, "
                    f"d_ratio {r['train']['d_ratio']:.2f})\n\n")
            f.write(f"baseline Pearson(eh_scaled, th) = **{e1['baseline']:+.4f}**\n\n")
            f.write("| 减去的特征 | β* | Pearson 后 | Δ |\n|---|---|---|---|\n")
            for fk, P, b in e1["all_results"][:6]:
                f.write(f"| {b:+.3f}·{fk} | {b:+.3f} | {P:+.4f} | {P - e1['baseline']:+.4f} |\n")
            f.write("\n")
        f.write("## E2: 物种身份\n\n")
        for k, r in results.items():
            f.write(f"### {k}\n\n")
            f.write("| idx | mean | std | range_r | corr(h) | corr(sat, h) |\n")
            f.write("|---|---|---|---|---|---|\n")
            for s in r["e2"]:
                f.write(f"| {s['species_idx']} | {s['mean']:.3f} | {s['std']:.3f} | "
                        f"{s['range_ratio']:.1f} | {s['corr_w_hidden']:+.3f} | "
                        f"{s['corr_w_hidden_sat']:+.3f} |\n")
            f.write("\n")
        f.write("## E4: 补偿的时间结构\n\n")
        for k, r in results.items():
            e4 = r["e4"]
            f.write(f"### {k}\n\n")
            f.write(f"- 最强补偿来源: `x_{e4['best_k']}/(1+x_{e4['best_k']})` "
                    f"corr={e4['corr_residual_satk']:+.3f}\n")
            f.write(f"- |residual| 在高 x_{e4['best_k']} 时均值: {e4['mean_abs_residual_at_high_x']:.4f}\n")
            f.write(f"- |residual| 在低 x_{e4['best_k']} 时均值: {e4['mean_abs_residual_at_low_x']:.4f}\n")
            f.write(f"- 高/低 ratio: **{e4['hi_lo_ratio']:.2f}** (>1 意味着补偿在高密度时强)\n\n")
        f.write("\n见 `E4_*_time.png`.\n")

    # JSON dump (strip heavy arrays)
    dump = {k: {kk: (vv if kk not in ("encoder_h", "true_h") else "(truncated)")
                for kk, vv in r.items()} for k, r in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

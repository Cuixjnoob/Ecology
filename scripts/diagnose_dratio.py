"""D1 + D2 诊断: encoder_h 为什么 Holling d_ratio=16 而 LV d_ratio=1.7?

D1: scatter(encoder_h, true_h) - 看是否非线性
D2: residual = encoder_h - linear(true_h); corr(residual, x_j^k for k=1,2) and
    corr(residual, Holling-saturation features) - 找补偿信号

复用 stage1b config + G_anchor_first, 单 seed best.
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


def train_and_extract(name, visible, hidden, seed, device, epochs=300):
    """Train one seed and return encoder_h + G + f_visible evaluations."""
    print(f"\n=== {name} (seed={seed}) ===")
    torch.manual_seed(seed)
    model = make_model(visible.shape[1], device)
    r = train_one_fast(
        model, visible, hidden, device=device,
        epochs=epochs, lr=0.0008,
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
    print(f"Pearson={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}")

    # Extract encoder_h (unsampled mean)
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        enc_out = model.encoder(x)
        if len(enc_out) == 2:
            mu_k, _ = enc_out
            encoder_h = mu_k[..., 0]
        else:
            mu_K, _, logits_K = enc_out
            mu_K = mu_K[..., 0]; logits_K = logits_K[..., 0]
            pi = F.softmax(logits_K, dim=-1)
            encoder_h = (pi * mu_K).sum(-1)
        base = model.compute_f_visible(x).cpu().numpy()[0]
        G = model.compute_G(x).cpu().numpy()[0]
        encoder_h_np = encoder_h.cpu().numpy()[0]
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pearson": r["pearson"],
        "d_ratio": r["d_ratio"],
        "encoder_h": encoder_h_np,    # (T,)
        "true_h": hidden,              # (T,)
        "visible": visible,            # (T, N)
        "base": base,                  # (T, N)
        "G": G,                        # (T, N)
    }


def d1_scatter(results, out_dir):
    """Scatter encoder_h vs true_h, with linear fit."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    for ax, (name, r) in zip(axes, results.items()):
        eh = r["encoder_h"]
        th = r["true_h"][:len(eh)]
        # Scale eh to match true_h std for visual
        eh_c = eh - eh.mean()
        th_c = th - th.mean()
        eh_scaled = eh_c * (th_c.std() / (eh_c.std() + 1e-8)) + th.mean()
        # Linear fit
        a, b = np.polyfit(eh_scaled, th, 1)
        pred = a * eh_scaled + b
        # Quadratic fit
        p2 = np.polyfit(eh_scaled, th, 2)
        xs = np.linspace(eh_scaled.min(), eh_scaled.max(), 200)
        pred2 = np.polyval(p2, xs)
        # R² of each
        ss_res_lin = np.sum((th - pred) ** 2)
        ss_res_q = np.sum((th - np.polyval(p2, eh_scaled)) ** 2)
        ss_tot = np.sum((th - th.mean()) ** 2)
        r2_lin = 1 - ss_res_lin / ss_tot
        r2_q = 1 - ss_res_q / ss_tot
        ax.scatter(eh_scaled, th, s=8, alpha=0.4, color="#555")
        ax.plot(xs, a * xs + b, color="#1976d2", lw=2,
                label=f"linear  R²={r2_lin:.3f}")
        ax.plot(xs, pred2, color="#d32f2f", lw=2, linestyle="--",
                label=f"quadratic  R²={r2_q:.3f}")
        ax.set_xlabel("encoder_h (scaled to match true_h std)")
        ax.set_ylabel("true_h")
        ax.set_title(f"{name}: P={r['pearson']:+.3f}  d_ratio={r['d_ratio']:.1f}\n"
                     f"R²(q)-R²(lin) = {r2_q - r2_lin:.4f}")
        ax.legend(); ax.grid(alpha=0.3)
    fig.suptitle("D1: encoder_h vs true_h", fontsize=14, fontweight="bold")
    fig.savefig(out_dir / "D1_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(fig)


def d2_residual_analysis(results, out_dir):
    """residual = encoder_h - linear(true_h). Correlate with visible features."""
    report = []
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), constrained_layout=True)

    for col_idx, (name, r) in enumerate(results.items()):
        eh = r["encoder_h"]
        th = r["true_h"][:len(eh)]
        vis = r["visible"][:len(eh)]   # (T, N)

        # Scale encoder_h to true_h std
        eh_c = eh - eh.mean()
        th_c = th - th.mean()
        eh_scaled = eh_c * (th_c.std() / (eh_c.std() + 1e-8))
        # Linear fit: eh_scaled = a * th_c + b
        a, b = np.polyfit(th_c, eh_scaled, 1)
        pred_lin = a * th_c + b
        residual = eh_scaled - pred_lin

        # Correlation with visible features
        N = vis.shape[1]
        feat_types = {}
        for j in range(N):
            x_j = vis[:, j]
            feat_types[f"x_{j}"] = x_j
            feat_types[f"x_{j}^2"] = x_j ** 2
            # Holling saturation: x / (1 + x)
            feat_types[f"x_{j}/(1+x_{j})"] = x_j / (1 + x_j)
            # Bilinear with others (take avg)
            feat_types[f"x_{j}·x_mean"] = x_j * vis.mean(axis=1)

        corrs = {}
        for k, v in feat_types.items():
            if v.std() > 1e-8:
                c = np.corrcoef(residual, v)[0, 1]
                corrs[k] = c

        # Also compare: correlation of encoder_h with these features (baseline)
        corrs_eh = {}
        for k, v in feat_types.items():
            if v.std() > 1e-8:
                c = np.corrcoef(eh_scaled, v)[0, 1]
                corrs_eh[k] = c

        # residual time series plot
        ax = axes[0, col_idx]
        ax.plot(residual, color="#c62828", lw=1)
        ax.set_title(f"{name} (d_r={r['d_ratio']:.1f}): residual = encoder_h_scaled − linear(true_h)")
        ax.set_xlabel("time"); ax.set_ylabel("residual")
        ax.grid(alpha=0.3)
        ax.text(0.02, 0.95, f"std(residual) = {residual.std():.4f}\n"
                f"std(encoder_h_scaled) = {eh_scaled.std():.4f}\n"
                f"ratio = {residual.std() / eh_scaled.std():.3f}",
                transform=ax.transAxes, va="top", fontsize=10,
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

        # Top feature correlations
        ax = axes[1, col_idx]
        sorted_feats = sorted(corrs.items(), key=lambda kv: -abs(kv[1]))[:10]
        names = [k for k, _ in sorted_feats]
        vals = [v for _, v in sorted_feats]
        colors = ["#1976d2" if abs(v) > 0.15 else "#999" for v in vals]
        ax.barh(range(len(names)), vals, color=colors)
        ax.set_yticks(range(len(names)))
        ax.set_yticklabels(names, fontsize=9)
        ax.set_xlabel("corr(residual, feature)")
        ax.set_title(f"{name}: top-10 |corr(residual, feature)|")
        ax.axvline(0, color="k", lw=0.5)
        ax.grid(alpha=0.3, axis="x")

        # Report
        report.append(f"\n### {name} (d_ratio={r['d_ratio']:.2f}, Pearson={r['pearson']:+.3f})")
        report.append(f"- residual std / encoder_h_scaled std = **{residual.std() / eh_scaled.std():.3f}** "
                      f"(0 = 残差为零; 1 = 全是残差)")
        report.append(f"- Top-5 |corr(residual, feature)|:")
        for k, v in sorted_feats[:5]:
            report.append(f"    - {k}: corr = {v:+.4f}")
        # How do Holling-saturation features rank?
        sat_corrs = {k: v for k, v in corrs.items() if "/(1+" in k}
        if sat_corrs:
            max_sat = max(sat_corrs.items(), key=lambda kv: abs(kv[1]))
            report.append(f"- Max |corr| with Holling-saturation feature: "
                          f"{max_sat[0]} = {max_sat[1]:+.4f}")

    fig.suptitle("D2: residual 结构分析 (encoder_h 扣除 linear(true_h) 后剩什么?)",
                 fontsize=13, fontweight="bold")
    fig.savefig(out_dir / "D2_residual_analysis.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    return report


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_diagnose_dratio")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Best seeds from previous run:
    # LV: seed=42 P=0.807 d_r=1.27 (best d_r)
    # Holling: seed=123 P=0.898 d_r=22.8 (best Pearson, high d_r)
    lv_vis, lv_hid = load_lv()
    ho_vis, ho_hid = load_holling()

    results = {}
    results["LV"] = train_and_extract("LV", lv_vis, lv_hid, seed=42, device=device)
    results["Holling"] = train_and_extract("Holling", ho_vis, ho_hid, seed=123, device=device)

    d1_scatter(results, out_dir)
    report = d2_residual_analysis(results, out_dir)

    # Write summary
    with open(out_dir / "analysis.md", "w", encoding="utf-8") as f:
        f.write("# d_ratio 诊断 — D1 + D2 结果\n\n")
        f.write(f"Seeds: LV=42, Holling=123\n\n")
        f.write("## 假说\n\n")
        f.write("encoder_h = linear(true_h) + 补偿项, 补偿项吃掉 G 的非线性饱和误差. \n")
        f.write("Holling 的 x/(1+αx) 饱和 → G 在高密度时压低 → encoder 必须补偿 → d_ratio 爆炸.\n\n")
        f.write("## D1 scatter 判定\n\n")
        f.write("见 D1_scatter.png. 如 quadratic R² 显著 > linear R² → 非线性关系证实.\n\n")
        f.write("## D2 residual 结构\n\n")
        f.write("".join([s + "\n" for s in report]))
        f.write("\n\n见 D2_residual_analysis.png.\n\n")
        f.write("## 关键观察\n\n")
        for name, r in results.items():
            f.write(f"- **{name}**: Pearson {r['pearson']:+.3f}, d_ratio {r['d_ratio']:.2f}\n")

    # JSON dump for re-analysis
    dump = {k: {"pearson": float(r["pearson"]), "d_ratio": float(r["d_ratio"]),
                "encoder_h": r["encoder_h"].tolist(),
                "true_h": r["true_h"][:len(r["encoder_h"])].tolist()}
            for k, r in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

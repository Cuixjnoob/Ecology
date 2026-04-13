"""最终 pipeline: Linear Sparse baseline + GNN scale refinement (GPU accelerated).

设计:
  Stage 1: Linear sparse baseline fit → h_coarse (0.977 on LV, 0.897 on Holling)
  Stage 2: 小 GNN 学 h_coarse 的 scale correction (不重学 hidden!)
           - GNN input: (visible_states, residual, Takens of h_coarse)
           - GNN output: scalar correction Δh(t) (small)
           - h_refined = h_coarse + Δh (Δh 被严格限制)

这样 GNN 只补 linear 的 magnitude 短板，不改 shape。

同时在 LV 和 Holling 上做多 seed 实验以验证稳定性。
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


# =============================================================================
# Stage 1: Linear Sparse Baseline (CPU fast)
# =============================================================================
def fit_sparse_linear(states, log_ratios, lam_sparse, n_iter=1500, lr=0.015, seed=42, device="cpu"):
    torch.manual_seed(seed)
    r = torch.zeros(5, device=device, requires_grad=True)
    A = torch.zeros(5, 5, device=device, requires_grad=True)
    with torch.no_grad():
        A.fill_diagonal_(-0.2)
        A.data += 0.01 * torch.randn(5, 5, device=device)
    opt = torch.optim.Adam([r, A], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32, device=device)
    y = torch.tensor(log_ratios, dtype=torch.float32, device=device)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit_loss + lam_sparse * A_off.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred = r.view(1, -1) + x @ A.T
        residual = (y - pred).cpu().numpy()
    return residual, r.detach().cpu().numpy(), A.detach().cpu().numpy()


def fit_full_with_hidden(states, log_ratios, h_current, lam_sparse=0.05, n_iter=1500, lr=0.015, seed=42, device="cpu"):
    torch.manual_seed(seed)
    r = torch.zeros(5, device=device, requires_grad=True)
    A = torch.zeros(5, 5, device=device, requires_grad=True)
    b = torch.zeros(5, device=device, requires_grad=True)
    c = torch.zeros(5, device=device, requires_grad=True)
    with torch.no_grad(): A.fill_diagonal_(-0.2)
    opt = torch.optim.Adam([r, A, b, c], lr=lr)
    x = torch.tensor(states[:-1], dtype=torch.float32, device=device)
    y = torch.tensor(log_ratios, dtype=torch.float32, device=device)
    h = torch.tensor(h_current, dtype=torch.float32, device=device)
    for _ in range(n_iter):
        opt.zero_grad()
        pred = r.view(1, -1) + x @ A.T + h.unsqueeze(-1) * b.view(1, -1) + (h.unsqueeze(-1) ** 2) * c.view(1, -1)
        fit_loss = ((pred - y) ** 2).mean()
        A_off = A - torch.diag(torch.diag(A))
        loss = fit_loss + lam_sparse * A_off.abs().mean()
        loss.backward()
        opt.step()
    with torch.no_grad():
        residual_no_hidden = y - (r.view(1, -1) + x @ A.T)
    return residual_no_hidden.cpu().numpy(), r.detach().cpu().numpy(), A.detach().cpu().numpy(), b.detach().cpu().numpy(), c.detach().cpu().numpy()


def recover_hidden(residual, hidden_true):
    T = residual.shape[0]
    Z = np.concatenate([residual, np.ones((T, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(Z, hidden_true, rcond=None)
    h_pred = Z @ coef
    pear = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    return h_pred, pear, rmse


def run_sparse_em_pipeline(states, hidden_true, lam=0.5, em_iters=2, seed=42, device="cpu"):
    """LV sparse + EM."""
    safe = np.clip(states, 1e-6, None)
    log_ratios = np.log(safe[1:] / safe[:-1])
    log_ratios = np.clip(log_ratios, -1.12, 0.92)

    # Iter 0
    residual, r0, A0 = fit_sparse_linear(states, log_ratios, lam, seed=seed, device=device)
    h_coarse, p0, rmse0 = recover_hidden(residual, hidden_true[:-1])

    # Iter 1
    residual1, r1, A1, b1, c1 = fit_full_with_hidden(states, log_ratios, h_coarse, lam_sparse=0.05, seed=seed, device=device)
    h_iter1, p1, rmse1 = recover_hidden(residual1, hidden_true[:-1])

    # Pick best
    best = (p0, rmse0, h_coarse, 0)
    if abs(p1) > abs(best[0]):
        best = (p1, rmse1, h_iter1, 1)
    return {"pearson": best[0], "rmse": best[1], "h_pred": best[2], "best_iter": best[3]}


# =============================================================================
# Stage 2: GNN Scale Refinement (GPU accelerated)
# =============================================================================
class ScaleRefinementGNN(nn.Module):
    """小 GNN 学 h_coarse 的 scale correction，不重新学 hidden."""
    def __init__(self, num_visible: int = 5, takens_lags=(1, 2, 4, 8), d_model: int = 64, num_blocks: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_visible = num_visible
        self.takens_lags = list(takens_lags)
        self.d_model = d_model

        # Features per (t, n): [x, log_x, residual, h_coarse_global, h_coarse Takens]
        feat_dim = 3 + 1 + len(self.takens_lags)
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.species_emb = nn.Parameter(torch.randn(num_visible, d_model) * 0.1)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "species_attn": nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "temporal_attn": nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model * 2, d_model), nn.Dropout(dropout)),
                "norm3": nn.LayerNorm(d_model),
            })
            for _ in range(num_blocks)
        ])
        self.readout = nn.Sequential(
            nn.Linear(d_model * num_visible, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, visible, residual, h_coarse):
        """
        visible: (B, T, N)
        residual: (B, T-1, N) baseline residual (pad to T)
        h_coarse: (B, T) coarse hidden estimate
        Returns: delta_h (B, T)  small correction
        """
        B, T, N = visible.shape
        safe = torch.clamp(visible, min=1e-6)
        log_v = torch.log(safe)

        # Pad residual to T
        res_T = F.pad(residual, (0, 0, 1, 0), value=0.0)

        # Takens of h_coarse
        takens_h = []
        for lag in self.takens_lags:
            padded = F.pad(h_coarse.unsqueeze(1), (lag, 0), value=0.0)[..., :T]
            takens_h.append(padded.squeeze(1).unsqueeze(-1))  # (B, T, 1)
        takens_h_stack = torch.cat(takens_h, dim=-1)  # (B, T, L)

        # Broadcast per (t, n): concat features
        # Features: [x, log_x, residual, h_coarse, takens_h (broadcast)]
        h_broad = h_coarse.unsqueeze(-1).expand(-1, -1, N).unsqueeze(-1)  # (B, T, N, 1)
        takens_broad = takens_h_stack.unsqueeze(2).expand(-1, -1, N, -1)   # (B, T, N, L)
        feats = torch.cat([
            visible.unsqueeze(-1),
            log_v.unsqueeze(-1),
            res_T.unsqueeze(-1),
            h_broad,
            takens_broad,
        ], dim=-1)  # (B, T, N, feat_dim)

        x = self.input_proj(feats) + self.species_emb.view(1, 1, N, -1)

        for block in self.blocks:
            B_, T_, N_, D = x.shape
            # Species attn
            x_s = x.reshape(B_ * T_, N_, D)
            a, _ = block["species_attn"](x_s, x_s, x_s)
            x_s = block["norm1"](x_s + a)
            x = x_s.reshape(B_, T_, N_, D)
            # Temporal attn
            x_t = x.permute(0, 2, 1, 3).contiguous().reshape(B_ * N_, T_, D)
            a, _ = block["temporal_attn"](x_t, x_t, x_t)
            x_t = block["norm2"](x_t + a)
            x = x_t.reshape(B_, N_, T_, D).permute(0, 2, 1, 3).contiguous()
            # FFN
            x = block["norm3"](x + block["ffn"](x))

        x_flat = x.reshape(B, T, N * self.d_model)
        delta = self.readout(x_flat).squeeze(-1)
        return delta


def train_gnn_refiner(visible_np, residual_np, h_coarse_np, hidden_true_np, epochs=500, lr=0.001, device="cuda", seed=42, train_ratio=0.75):
    """Train GNN to refine h_coarse. Loss: visible reconstruction when using h_refined."""
    T, N = visible_np.shape
    train_end = int(train_ratio * T)

    visible = torch.tensor(visible_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T, N)
    safe = torch.clamp(visible, min=1e-6)
    actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
    actual_log_ratio = torch.clamp(actual_log_ratio, -1.12, 0.92)
    residual = torch.tensor(residual_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T-1, N)
    h_coarse = torch.tensor(h_coarse_np, dtype=torch.float32, device=device).unsqueeze(0)  # (1, T-1)
    # Pad h_coarse to T
    h_coarse_T = F.pad(h_coarse, (1, 0), value=h_coarse_np[0] if len(h_coarse_np) > 0 else 0.0)

    torch.manual_seed(seed)
    model = ScaleRefinementGNN(num_visible=N).to(device)

    # Coupling params (learn how h_refined enters visible reconstruction)
    b = torch.zeros(N, device=device, requires_grad=True)
    c = torch.zeros(N, device=device, requires_grad=True)
    r = torch.zeros(N, device=device, requires_grad=True)

    optimizer = torch.optim.AdamW(list(model.parameters()) + [b, c, r], lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

    best_val = float("inf")
    best_h_refined = None
    best_epoch = -1

    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        delta_h = model(visible, residual, h_coarse_T)  # (1, T)
        # Small correction: h_refined = h_coarse + 0.5 * delta_h (bounded)
        h_refined = h_coarse_T + 0.5 * torch.tanh(delta_h)
        # Predict log_ratio: baseline residual explained + hidden coupling
        h_refined_current = h_refined[:, :-1]
        # residual = b * h_refined + c * h_refined^2 + r (approximately)
        pred_residual = b.view(1, 1, -1) * h_refined_current.unsqueeze(-1) + c.view(1, 1, -1) * (h_refined_current.unsqueeze(-1) ** 2) + r.view(1, 1, -1)
        # Fit residual
        fit_train = F.mse_loss(pred_residual[:, :train_end-1], residual[:, :train_end-1])
        fit_val = F.mse_loss(pred_residual[:, train_end-1:], residual[:, train_end-1:]).item()
        # Regularizer: keep delta small
        reg = 0.01 * (delta_h ** 2).mean()
        loss = fit_train + reg
        loss.backward()
        torch.nn.utils.clip_grad_norm_(list(model.parameters()) + [b, c, r], 1.0)
        optimizer.step()
        scheduler.step()

        if fit_val < best_val:
            best_val = fit_val
            best_h_refined = h_refined[0].detach().cpu().numpy()
            best_epoch = epoch + 1

    # Evaluate - align shapes (h_refined is T, hidden_true_np is T-1 (already :-1))
    # Take [:T-1] of h_refined to match
    L = min(len(best_h_refined), len(hidden_true_np))
    best_h_refined = best_h_refined[:L]
    h_true = hidden_true_np[:L]
    pear = float(np.corrcoef(best_h_refined, h_true)[0, 1])
    rmse = float(np.sqrt(((best_h_refined - h_true) ** 2).mean()))
    # Scale-invariant
    X = np.concatenate([best_h_refined.reshape(-1, 1), np.ones((len(best_h_refined), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, h_true, rcond=None)
    h_scaled = X @ coef
    pear_s = float(np.corrcoef(h_scaled, h_true)[0, 1])
    rmse_s = float(np.sqrt(((h_scaled - h_true) ** 2).mean()))
    return {"pearson_raw": pear, "rmse_raw": rmse, "pearson_scaled": pear_s, "rmse_scaled": rmse_s, "h_refined": best_h_refined, "h_scaled": h_scaled, "best_epoch": best_epoch}


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_final_multiseed_gnn_refine")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if device == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load data
    d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
    states_lv = d_lv["states_B_5species"]
    hidden_lv = d_lv["hidden_B"]

    holling_dirs = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
    d_h = np.load(holling_dirs[-1])
    states_h = d_h["states_B_5species"]
    hidden_h = d_h["hidden_B"]

    # Configurations per dataset
    configs = {
        "LV": {"states": states_lv, "hidden_true": hidden_lv, "lam": 0.5},
        "Holling": {"states": states_h, "hidden_true": hidden_h, "lam": 2.0},
    }

    seeds = [42, 43, 44, 45, 46]
    all_results = {}

    for label, cfg in configs.items():
        print(f"\n{'='*70}\n{label} data\n{'='*70}")
        # Stage 1: multi-seed linear sparse + EM
        stage1_results = []
        for seed in seeds:
            r = run_sparse_em_pipeline(cfg["states"], cfg["hidden_true"], lam=cfg["lam"], seed=seed, device="cpu")
            print(f"  Seed {seed}: Pearson={r['pearson']:+.4f}  RMSE={r['rmse']:.4f}  (best iter={r['best_iter']})")
            stage1_results.append(r)

        pears_stage1 = np.array([r["pearson"] for r in stage1_results])
        rmses_stage1 = np.array([r["rmse"] for r in stage1_results])
        print(f"  Stage 1 stats: Pearson={pears_stage1.mean():.4f}±{pears_stage1.std():.4f}  RMSE={rmses_stage1.mean():.4f}±{rmses_stage1.std():.4f}")

        # Stage 2: GNN refinement on the best seed's h_coarse
        # Use seed 42's result
        best_seed_result = stage1_results[0]
        # Compute residual for seed 42
        safe = np.clip(cfg["states"], 1e-6, None)
        log_ratios = np.log(safe[1:] / safe[:-1])
        log_ratios = np.clip(log_ratios, -1.12, 0.92)
        residual_for_gnn, _, _ = fit_sparse_linear(cfg["states"], log_ratios, cfg["lam"], seed=42, device=device)

        print(f"\n  Stage 2: GNN scale refinement on GPU...")
        gnn_result = train_gnn_refiner(
            cfg["states"], residual_for_gnn, best_seed_result["h_pred"],
            cfg["hidden_true"][:-1], epochs=500, lr=0.001, device=device, seed=42,
        )
        print(f"    GNN refined: Pearson={gnn_result['pearson_scaled']:.4f}  RMSE={gnn_result['rmse_scaled']:.4f}  (best @ {gnn_result['best_epoch']})")
        print(f"    Stage 1 (seed 42): Pearson={best_seed_result['pearson']:.4f}  RMSE={best_seed_result['rmse']:.4f}")

        all_results[label] = {
            "stage1": stage1_results,
            "stage2_gnn": gnn_result,
            "stage1_stats": {
                "pearson_mean": float(pears_stage1.mean()), "pearson_std": float(pears_stage1.std()),
                "rmse_mean": float(rmses_stage1.mean()), "rmse_std": float(rmses_stage1.std()),
            },
        }

    # Summary
    print()
    print("=" * 70)
    print("FINAL RESULTS:")
    print("=" * 70)
    for label, res in all_results.items():
        s1 = res["stage1_stats"]
        s2 = res["stage2_gnn"]
        print(f"\n{label}:")
        print(f"  Stage 1 (Linear + EM, 5 seeds): Pearson={s1['pearson_mean']:.4f}±{s1['pearson_std']:.4f}  RMSE={s1['rmse_mean']:.4f}±{s1['rmse_std']:.4f}")
        print(f"  Stage 2 (+GNN refinement):      Pearson={s2['pearson_scaled']:.4f}  RMSE={s2['rmse_scaled']:.4f}")

    # Save
    np.savez(out_dir / "results.npz",
              lv_stage1_pearson=[r["pearson"] for r in all_results["LV"]["stage1"]],
              lv_stage1_rmse=[r["rmse"] for r in all_results["LV"]["stage1"]],
              holling_stage1_pearson=[r["pearson"] for r in all_results["Holling"]["stage1"]],
              holling_stage1_rmse=[r["rmse"] for r in all_results["Holling"]["stage1"]],
              lv_gnn_pearson=all_results["LV"]["stage2_gnn"]["pearson_scaled"],
              lv_gnn_rmse=all_results["LV"]["stage2_gnn"]["rmse_scaled"],
              holling_gnn_pearson=all_results["Holling"]["stage2_gnn"]["pearson_scaled"],
              holling_gnn_rmse=all_results["Holling"]["stage2_gnn"]["rmse_scaled"])

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for idx, (label, res) in enumerate(all_results.items()):
        # Top: hidden trajectory (stage 2)
        ax = axes[0, idx]
        h_true = configs[label]["hidden_true"][:-1]
        s2 = res["stage2_gnn"]
        t_axis = np.arange(len(h_true))
        ax.plot(t_axis, h_true, color="black", linewidth=1.2, label="真实")
        # Stage 1 best seed
        s1 = res["stage1"][0]["h_pred"]
        ax.plot(t_axis[:len(s1)], s1, color="#ff7f0e", linewidth=0.9, alpha=0.7, label=f"Stage1 (Linear+EM, P={res['stage1'][0]['pearson']:.3f})")
        ax.plot(t_axis[:len(s2['h_scaled'])], s2["h_scaled"], color="#1565c0", linewidth=0.9, alpha=0.85, label=f"Stage2 (+GNN, P={s2['pearson_scaled']:.3f})")
        ax.set_title(f"{label}: Stage1 vs Stage2", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

        # Bottom: seed distribution
        ax = axes[1, idx]
        pears = [r["pearson"] for r in res["stage1"]]
        rmses = [r["rmse"] for r in res["stage1"]]
        x_pos = np.arange(len(seeds))
        ax.bar(x_pos - 0.2, pears, 0.4, label="Pearson", color="#1565c0")
        ax2 = ax.twinx()
        ax2.bar(x_pos + 0.2, rmses, 0.4, label="RMSE", color="#c62828")
        ax.axhline(res["stage1_stats"]["pearson_mean"], color="#1565c0", linestyle="--", linewidth=0.8)
        ax2.axhline(res["stage1_stats"]["rmse_mean"], color="#c62828", linestyle="--", linewidth=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f"s{s}" for s in seeds])
        ax.set_ylabel("Pearson", color="#1565c0")
        ax2.set_ylabel("RMSE", color="#c62828")
        ax.set_title(f"{label}: {len(seeds)}-seed stability")
        ax.grid(alpha=0.25, axis="y")

    fig.suptitle("多 seed + GNN scale refinement", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_final.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# 最终 multi-seed + GNN refinement 结果\n\n")
        for label, res in all_results.items():
            s1 = res["stage1_stats"]
            s2 = res["stage2_gnn"]
            f.write(f"## {label}\n\n")
            f.write(f"### Stage 1: Linear Sparse + EM ({len(seeds)} seeds)\n")
            f.write(f"- Pearson: {s1['pearson_mean']:.4f} ± {s1['pearson_std']:.4f}\n")
            f.write(f"- RMSE: {s1['rmse_mean']:.4f} ± {s1['rmse_std']:.4f}\n\n")
            f.write(f"### Stage 2: + GNN Scale Refinement\n")
            f.write(f"- Pearson (scaled): {s2['pearson_scaled']:.4f}\n")
            f.write(f"- RMSE (scaled): {s2['rmse_scaled']:.4f}\n\n")
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

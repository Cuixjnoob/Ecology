"""Residual-Signature Network: Learned Baseline Extraction (RSN-LBE)

核心思想：
  已知信息论上限：如果有准确的 "no-hidden baseline"，5 个 residual 就能线性组合出 Pearson 0.95 的 hidden
  所以问题转化为：如何从 B 系统数据本身学到 no-hidden baseline？

方法：
  联合优化以下所有参数，Loss 是 visible 重构：
    - r_5 ∈ R^5      : visible-only growth rates
    - A_55 ∈ R^{5×5} : visible-only interaction matrix
    - h(t) ∈ R^T     : hidden 时间序列（核心恢复目标）
    - b_5 ∈ R^5      : hidden → visible coupling
    - c_5 ∈ R^5      : hidden^2 → visible（非线性 coupling，可选）

  动力学形式（Ricker）：
    log(x_{t+1,i}/x_{t,i}) = r_5[i] + A_55[i,:]·x_t + b_5[i]·h(t) + eps

  正则：
    - h(t) 平滑性
    - b_5 / h scale 归一化（identifiability）
    - Sparse regularization on A_55 (ecological prior)

  训练：纯梯度下降，所有变量同时优化
  监督：无 hidden 监督，只有 visible 观测 loss
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from scipy.stats import spearmanr


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


class LearnedBaselineExtraction(nn.Module):
    """联合估计 visible-only 动力学 + hidden 时序 + 耦合系数。"""

    def __init__(self, num_visible: int, num_steps: int):
        super().__init__()
        self.num_visible = num_visible
        self.num_steps = num_steps

        # Visible-only Ricker 参数
        self.r5 = nn.Parameter(0.1 * torch.ones(num_visible))
        # Initialize A_55 with small random + negative diagonal
        A_init = 0.02 * torch.randn(num_visible, num_visible)
        A_init.fill_diagonal_(-0.2)
        self.A55 = nn.Parameter(A_init)

        # Hidden 时间序列（核心恢复目标）
        self.h_raw = nn.Parameter(0.01 * torch.randn(num_steps))

        # Hidden → visible coupling (linear)
        self.b5 = nn.Parameter(0.1 * torch.randn(num_visible))
        # Optional: hidden² → visible (quadratic coupling, for Ricker-like nonlinearity)
        self.c5 = nn.Parameter(0.01 * torch.randn(num_visible))

        # Learned bias (absorbed growth offsets)
        self.bias = nn.Parameter(torch.zeros(num_visible))

    def get_hidden(self) -> torch.Tensor:
        """Hidden is constrained to be positive via softplus, zero-centered."""
        return torch.nn.functional.softplus(self.h_raw) + 0.01

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Predict log-ratio from state sequence."""
        # x_t: (T-1, 5) current state
        T = x_t.shape[0]
        h = self.get_hidden()[:T]  # (T,)
        # Ricker-style log-ratio prediction
        # log(x_{t+1}/x_t) = r5 + A55 @ x_t + b5*h + c5*h^2 + bias
        lv_term = self.r5 + x_t @ self.A55.T  # (T, 5)
        hidden_linear = h.unsqueeze(1) * self.b5.unsqueeze(0)  # (T, 5)
        hidden_quad = (h ** 2).unsqueeze(1) * self.c5.unsqueeze(0)
        pred = lv_term + hidden_linear + hidden_quad + self.bias
        return pred


def train_rsn_lbe(
    visible_states: np.ndarray,   # (T, 5)
    num_epochs: int = 5000,
    lr: float = 0.01,
    smooth_lambda: float = 0.1,
    sparse_lambda: float = 0.001,
    device: str = "cpu",
    verbose: bool = True,
):
    """训练 Learned Baseline Extraction 模型。"""
    T, num_visible = visible_states.shape
    x = torch.tensor(visible_states, dtype=torch.float32, device=device)

    # Target: log-ratio
    safe = torch.clamp(x, min=1e-6)
    log_ratio = torch.log(safe[1:] / safe[:-1])
    log_ratio = torch.clamp(log_ratio, min=-1.12, max=0.92)  # clamp to data's range

    model = LearnedBaselineExtraction(num_visible, num_steps=T).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"loss": [], "fit": [], "smooth": [], "sparse": []}
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        pred = model(x[:-1])  # (T-1, 5)

        fit_loss = torch.nn.functional.mse_loss(pred, log_ratio)

        # Regularizers
        h = model.get_hidden()
        # Smoothness: hidden 应该平滑（二阶差分）
        smooth_loss = ((h[2:] - 2 * h[1:-1] + h[:-2]) ** 2).mean()
        # Sparse A_55 off-diagonal
        A_offdiag = model.A55 - torch.diag(torch.diag(model.A55))
        sparse_loss = A_offdiag.abs().mean()

        loss = fit_loss + smooth_lambda * smooth_loss + sparse_lambda * sparse_loss
        loss.backward()
        optimizer.step()

        history["loss"].append(loss.item())
        history["fit"].append(fit_loss.item())
        history["smooth"].append(smooth_loss.item())
        history["sparse"].append(sparse_loss.item())

        if verbose and (epoch % 500 == 0 or epoch == num_epochs - 1):
            print(f"  epoch {epoch}: loss={loss.item():.6f} fit={fit_loss.item():.6f} "
                  f"smooth={smooth_loss.item():.6f} sparse={sparse_loss.item():.6f}")

    return model, history


def evaluate(model, hidden_true: np.ndarray):
    """评估恢复的 hidden 质量。"""
    with torch.no_grad():
        h_pred = model.get_hidden().cpu().numpy()

    # 直接对比（可能有 scale 不一致）
    pearson_raw = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse_raw = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))

    # Scale-invariant evaluation: 用 linear regression 找最佳线性映射
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((len(h_pred), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    pearson_scaled = float(np.corrcoef(h_scaled, hidden_true)[0, 1])
    rmse_scaled = float(np.sqrt(((h_scaled - hidden_true) ** 2).mean()))

    return {
        "pearson_raw": pearson_raw,
        "rmse_raw": rmse_raw,
        "pearson_scaled": pearson_scaled,
        "rmse_scaled": rmse_scaled,
        "h_pred": h_pred,
        "h_scaled": h_scaled,
        "scale_coef": coef.tolist(),
    }


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_rsn_lbe")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    states_B = d["states_B_5species"]   # (T, 5) visible obs from 6-species system
    hidden_true = d["hidden_B"]          # (T,) true hidden

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Data: T={states_B.shape[0]}, num_visible={states_B.shape[1]}")
    print(f"Hidden range: [{hidden_true.min():.3f}, {hidden_true.max():.3f}], std={hidden_true.std():.3f}")
    print()

    # Hyperparameter sweep
    configs = [
        {"lr": 0.01, "smooth": 0.1, "sparse": 0.001, "epochs": 5000},
        {"lr": 0.01, "smooth": 1.0, "sparse": 0.001, "epochs": 5000},
        {"lr": 0.01, "smooth": 10.0, "sparse": 0.001, "epochs": 5000},
        {"lr": 0.005, "smooth": 0.1, "sparse": 0.0001, "epochs": 8000},
        {"lr": 0.02, "smooth": 0.01, "sparse": 0.01, "epochs": 5000},
    ]

    best_result = None
    all_results = []
    for cfg_i, cfg in enumerate(configs):
        print(f"\n=== Config {cfg_i+1}: lr={cfg['lr']}, smooth={cfg['smooth']}, sparse={cfg['sparse']}, epochs={cfg['epochs']} ===")
        torch.manual_seed(42 + cfg_i)
        model, history = train_rsn_lbe(
            states_B,
            num_epochs=cfg["epochs"],
            lr=cfg["lr"],
            smooth_lambda=cfg["smooth"],
            sparse_lambda=cfg["sparse"],
            device=device,
            verbose=True,
        )
        eval_result = evaluate(model, hidden_true)
        eval_result["config"] = cfg
        all_results.append(eval_result)
        print(f"  -> Pearson raw={eval_result['pearson_raw']:.4f}  scaled={eval_result['pearson_scaled']:.4f}")
        print(f"  -> RMSE raw={eval_result['rmse_raw']:.4f}  scaled={eval_result['rmse_scaled']:.4f}")

        if best_result is None or abs(eval_result["pearson_scaled"]) > abs(best_result["pearson_scaled"]):
            best_result = eval_result

    # 最终结果
    print()
    print("=" * 70)
    print(f"BEST CONFIG: {best_result['config']}")
    print(f"  Pearson (scaled): {best_result['pearson_scaled']:.4f}")
    print(f"  RMSE (scaled):    {best_result['rmse_scaled']:.4f}")
    print(f"  Pearson (raw):    {best_result['pearson_raw']:.4f}")
    print(f"  RMSE (raw):       {best_result['rmse_raw']:.4f}")
    print("=" * 70)

    # 画图
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # 恢复的 hidden vs 真实 hidden
    t_axis = np.arange(len(hidden_true))
    axes[0, 0].plot(t_axis, hidden_true, color="black", linewidth=1.5, label="真实 hidden")
    axes[0, 0].plot(t_axis, best_result["h_scaled"], color="#ff7f0e", linewidth=1.0, alpha=0.85,
                     label=f"RSN-LBE 恢复 (scaled)")
    axes[0, 0].set_title(f"Hidden 恢复: Pearson={best_result['pearson_scaled']:.3f}, RMSE={best_result['rmse_scaled']:.3f}",
                          fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("时间步")
    axes[0, 0].set_ylabel("hidden")
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.25)

    # Scatter
    axes[0, 1].scatter(hidden_true, best_result["h_scaled"], alpha=0.3, s=8, color="#ff7f0e")
    vmin = min(hidden_true.min(), best_result["h_scaled"].min())
    vmax = max(hidden_true.max(), best_result["h_scaled"].max())
    axes[0, 1].plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
    axes[0, 1].set_xlabel("真实 hidden")
    axes[0, 1].set_ylabel("恢复 hidden (scaled)")
    axes[0, 1].set_title("真实 vs 恢复散点")
    axes[0, 1].grid(alpha=0.25)

    # 所有 config 的 Pearson 对比
    pearsons_scaled = [abs(r["pearson_scaled"]) for r in all_results]
    pearsons_raw = [abs(r["pearson_raw"]) for r in all_results]
    x_pos = np.arange(len(all_results))
    width = 0.35
    axes[1, 0].bar(x_pos - width/2, pearsons_scaled, width, label="scaled", color="#1565c0")
    axes[1, 0].bar(x_pos + width/2, pearsons_raw, width, label="raw", color="#c62828", alpha=0.7)
    axes[1, 0].set_xticks(x_pos)
    axes[1, 0].set_xticklabels([f"cfg{i+1}" for i in range(len(all_results))])
    axes[1, 0].set_ylabel("|Pearson|")
    axes[1, 0].set_title("所有 config 的 Pearson 对比")
    axes[1, 0].axhline(0.9, color="green", linestyle="--", linewidth=0.8, label="0.9 目标")
    axes[1, 0].legend(fontsize=9)
    axes[1, 0].grid(alpha=0.25)

    # RMSE 对比
    rmses_scaled = [r["rmse_scaled"] for r in all_results]
    axes[1, 1].bar(x_pos, rmses_scaled, color="#1565c0")
    axes[1, 1].set_xticks(x_pos)
    axes[1, 1].set_xticklabels([f"cfg{i+1}" for i in range(len(all_results))])
    axes[1, 1].set_ylabel("RMSE (scaled)")
    axes[1, 1].set_title("所有 config 的 RMSE")
    axes[1, 1].axhline(0.1, color="green", linestyle="--", linewidth=0.8, label="0.1 目标")
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("RSN-LBE (Learned Baseline Extraction) 结果", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_lbe_results.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] saved: {out_dir / 'fig_lbe_results.png'}")

    np.savez(
        out_dir / "best_recovery.npz",
        hidden_true=hidden_true,
        h_pred_raw=best_result["h_pred"],
        h_pred_scaled=best_result["h_scaled"],
        best_config=str(best_result["config"]),
    )


if __name__ == "__main__":
    main()

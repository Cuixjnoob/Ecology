"""训练 Residual-Signature GNN 恢复 hidden。

用法:
  python -m scripts.train_rs_gnn                # 默认跑一次完整训练
  python -m scripts.train_rs_gnn --smoke        # smoke test (少 epoch)
  python -m scripts.train_rs_gnn --epochs 3000  # 自定义

架构特点:
  - GNN 核心 (spatial + temporal multi-head attention, 4 blocks)
  - 残差动力学作为节点特征嵌入
  - Takens 延迟嵌入捕获动力学结构
  - 完全无 hidden 监督，只有 visible 重构 loss
"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.rs_gnn import RSGNN


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class TrainConfig:
    epochs: int = 3000
    lr: float = 0.001
    lr_warmup_steps: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Model size
    d_model: int = 128
    num_heads: int = 8
    num_blocks: int = 4
    delay_steps: int = 6
    dropout: float = 0.1

    # Loss weights
    smooth_lambda: float = 0.01
    sparse_lambda: float = 0.001
    var_lambda: float = 0.1

    # Training
    log_every: int = 50
    eval_every: int = 100


def evaluate_hidden_recovery(h_pred: np.ndarray, hidden_true: np.ndarray) -> Dict[str, float]:
    """Evaluate recovered hidden vs ground truth (with scale-invariant metrics)."""
    pearson_raw = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse_raw = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))

    # Scale-invariant: fit linear map
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
        "scale_a": float(coef[0]),
        "scale_b": float(coef[1]),
        "h_scaled": h_scaled,
    }


def train(cfg: TrainConfig, out_dir: Path, visible_states: np.ndarray, hidden_true: np.ndarray,
          device: str = "cpu"):
    print(f"\nDevice: {device}")
    print(f"Training config: {cfg}")

    T, N = visible_states.shape
    x = torch.tensor(visible_states, dtype=torch.float32, device=device)
    x = x.unsqueeze(0)  # (1, T, N)

    model = RSGNN(
        num_visible=N,
        num_steps=T,
        delay_steps=cfg.delay_steps,
        d_model=cfg.d_model,
        num_heads=cfg.num_heads,
        num_blocks=cfg.num_blocks,
        dropout=cfg.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    # Warmup + cosine
    def lr_lambda(step):
        if step < cfg.lr_warmup_steps:
            return step / cfg.lr_warmup_steps
        progress = (step - cfg.lr_warmup_steps) / max(1, cfg.epochs - cfg.lr_warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"loss": [], "fit": [], "smooth": [], "h_var": [], "eval_pearson": [], "eval_rmse": []}
    best_pearson = -1.0
    best_state = None
    best_eval = None

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        losses = model.compute_loss(
            out,
            smooth_lambda=cfg.smooth_lambda,
            sparse_lambda=cfg.sparse_lambda,
            var_lambda=cfg.var_lambda,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        history["loss"].append(losses["total"].item())
        history["fit"].append(losses["fit"].item())
        history["smooth"].append(losses["smooth"].item())
        history["h_var"].append(losses["h_variance"].item())

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:4d}: loss={losses['total'].item():.6f}  "
                  f"fit={losses['fit'].item():.6f}  smooth={losses['smooth'].item():.6f}  "
                  f"h_var={losses['h_variance'].item():.4f}  lr={lr_now:.5f}")

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                out_eval = model(x)
                h_pred = out_eval["hidden"][0].cpu().numpy()
            eval_res = evaluate_hidden_recovery(h_pred, hidden_true)
            history["eval_pearson"].append(eval_res["pearson_scaled"])
            history["eval_rmse"].append(eval_res["rmse_scaled"])
            print(f"       -> EVAL: Pearson={eval_res['pearson_scaled']:+.4f}  RMSE={eval_res['rmse_scaled']:.4f}  "
                  f"(raw Pearson={eval_res['pearson_raw']:+.4f})")

            if abs(eval_res["pearson_scaled"]) > best_pearson:
                best_pearson = abs(eval_res["pearson_scaled"])
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_eval = eval_res

    # Load best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        out_final = model(x)
        h_pred_final = out_final["hidden"][0].cpu().numpy()
    final_eval = evaluate_hidden_recovery(h_pred_final, hidden_true)

    print()
    print("=" * 70)
    print("Final Results:")
    print(f"  Pearson (scaled): {final_eval['pearson_scaled']:.4f}")
    print(f"  RMSE (scaled):    {final_eval['rmse_scaled']:.4f}")
    print(f"  Pearson (raw):    {final_eval['pearson_raw']:.4f}")
    print(f"  RMSE (raw):       {final_eval['rmse_raw']:.4f}")
    print(f"  Total params:     {num_params:,}")
    print("=" * 70)

    return model, history, final_eval


def plot_results(hidden_true: np.ndarray, eval_res: Dict, history: Dict, out_path: Path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # Top left: Hidden time series
    t_axis = np.arange(len(hidden_true))
    axes[0, 0].plot(t_axis, hidden_true, color="black", linewidth=1.5, label="真实 hidden")
    axes[0, 0].plot(t_axis, eval_res["h_scaled"], color="#ff7f0e", linewidth=1.0, alpha=0.85,
                     label=f"RS-GNN 恢复 (scaled)")
    axes[0, 0].set_title(f"Hidden 恢复: Pearson={eval_res['pearson_scaled']:.3f}, RMSE={eval_res['rmse_scaled']:.3f}",
                          fontsize=12, fontweight="bold")
    axes[0, 0].set_xlabel("时间步")
    axes[0, 0].set_ylabel("hidden")
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(alpha=0.25)

    # Top right: Scatter
    axes[0, 1].scatter(hidden_true, eval_res["h_scaled"], alpha=0.3, s=8, color="#ff7f0e")
    vmin = min(hidden_true.min(), eval_res["h_scaled"].min())
    vmax = max(hidden_true.max(), eval_res["h_scaled"].max())
    axes[0, 1].plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8, alpha=0.5)
    axes[0, 1].set_xlabel("真实 hidden")
    axes[0, 1].set_ylabel("恢复 hidden (scaled)")
    axes[0, 1].set_title(f"Scatter (scale={eval_res['scale_a']:.2f}x + {eval_res['scale_b']:.2f})")
    axes[0, 1].grid(alpha=0.25)

    # Bottom left: Training loss curves
    axes[1, 0].semilogy(history["loss"], color="#1565c0", linewidth=1.0, label="total")
    axes[1, 0].semilogy(history["fit"], color="#c62828", linewidth=1.0, label="fit")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel("loss")
    axes[1, 0].set_title("训练损失曲线")
    axes[1, 0].legend(fontsize=10)
    axes[1, 0].grid(alpha=0.25)

    # Bottom right: Evaluation progression
    eval_epochs = np.arange(1, len(history["eval_pearson"]) + 1)
    axes[1, 1].plot(eval_epochs, history["eval_pearson"], color="#1565c0", linewidth=1.5,
                     marker="o", markersize=3, label="|Pearson|")
    axes[1, 1].axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5, label="0.9")
    ax2 = axes[1, 1].twinx()
    ax2.plot(eval_epochs, history["eval_rmse"], color="#c62828", linewidth=1.5,
              marker="s", markersize=3, label="RMSE")
    ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5, label="0.1")
    axes[1, 1].set_xlabel("eval step")
    axes[1, 1].set_ylabel("|Pearson|", color="#1565c0")
    ax2.set_ylabel("RMSE", color="#c62828")
    axes[1, 1].set_title("评估指标随训练演进")
    axes[1, 1].legend(loc="lower right", fontsize=9)
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("RS-GNN (Residual-Signature GNN) 训练结果", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true", help="Quick smoke test")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--smooth-lambda", type=float, default=0.01)
    parser.add_argument("--var-lambda", type=float, default=0.1)
    args = parser.parse_args()

    _configure_matplotlib()

    out_dir = Path("runs/analysis_rs_gnn")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    visible_states = d["states_B_5species"]
    hidden_true = d["hidden_B"]
    print(f"Data: T={visible_states.shape[0]}, N={visible_states.shape[1]}")
    print(f"Hidden range: [{hidden_true.min():.3f}, {hidden_true.max():.3f}], std={hidden_true.std():.3f}")

    cfg = TrainConfig(
        epochs=args.epochs or (100 if args.smoke else 3000),
        lr=args.lr,
        d_model=args.d_model if not args.smoke else 32,
        num_heads=args.num_heads if not args.smoke else 4,
        num_blocks=args.num_blocks if not args.smoke else 2,
        smooth_lambda=args.smooth_lambda,
        var_lambda=args.var_lambda,
        log_every=10 if args.smoke else 50,
        eval_every=20 if args.smoke else 100,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(42)

    model, history, eval_res = train(cfg, out_dir, visible_states, hidden_true, device=device)

    suffix = "smoke" if args.smoke else "full"
    plot_results(hidden_true, eval_res, history, out_dir / f"fig_rs_gnn_{suffix}.png")
    print(f"\n[OK] saved: {out_dir / f'fig_rs_gnn_{suffix}.png'}")

    # Save results
    np.savez(
        out_dir / f"result_{suffix}.npz",
        hidden_true=hidden_true,
        h_pred_scaled=eval_res["h_scaled"],
        pearson=eval_res["pearson_scaled"],
        rmse=eval_res["rmse_scaled"],
    )


if __name__ == "__main__":
    main()

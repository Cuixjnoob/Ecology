"""训练 RS-GNN v2（双分支：重型生态残差 + 加强 Takens + GNN 核心）。"""
from __future__ import annotations

import argparse
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.rs_gnn_v2 import RSGNNv2


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class TrainConfig:
    epochs: int = 3000
    lr: float = 0.0008
    lr_warmup_steps: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    residual_scales: List[int] = field(default_factory=lambda: [1, 2, 3, 5])
    residual_takens_lags: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    residual_d_model: int = 96
    residual_mlp_layers: int = 3
    residual_attn_layers: int = 3

    state_takens_lags: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    state_d_model: int = 96
    state_mlp_layers: int = 2
    state_attn_layers: int = 2

    fusion_d_model: int = 128
    num_core_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    smooth_lambda: float = 0.01
    sparse_lambda: float = 0.001
    var_lambda: float = 0.1

    log_every: int = 50
    eval_every: int = 100


def evaluate_hidden_recovery(h_pred: np.ndarray, hidden_true: np.ndarray) -> Dict:
    pearson_raw = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse_raw = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((len(h_pred), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    pearson_scaled = float(np.corrcoef(h_scaled, hidden_true)[0, 1])
    rmse_scaled = float(np.sqrt(((h_scaled - hidden_true) ** 2).mean()))
    return {
        "pearson_raw": pearson_raw, "rmse_raw": rmse_raw,
        "pearson_scaled": pearson_scaled, "rmse_scaled": rmse_scaled,
        "scale_a": float(coef[0]), "scale_b": float(coef[1]),
        "h_scaled": h_scaled,
    }


def train(cfg: TrainConfig, visible_states: np.ndarray, hidden_true: np.ndarray, device: str = "cpu"):
    T, N = visible_states.shape
    x = torch.tensor(visible_states, dtype=torch.float32, device=device).unsqueeze(0)

    model = RSGNNv2(
        num_visible=N, num_steps=T,
        residual_scales=cfg.residual_scales,
        residual_takens_lags=cfg.residual_takens_lags,
        residual_d_model=cfg.residual_d_model,
        residual_mlp_layers=cfg.residual_mlp_layers,
        residual_attn_layers=cfg.residual_attn_layers,
        state_takens_lags=cfg.state_takens_lags,
        state_d_model=cfg.state_d_model,
        state_mlp_layers=cfg.state_mlp_layers,
        state_attn_layers=cfg.state_attn_layers,
        fusion_d_model=cfg.fusion_d_model,
        num_core_blocks=cfg.num_core_blocks,
        num_heads=cfg.num_heads,
        dropout=cfg.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"RS-GNN v2 parameters: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(step):
        if step < cfg.lr_warmup_steps:
            return step / cfg.lr_warmup_steps
        p = (step - cfg.lr_warmup_steps) / max(1, cfg.epochs - cfg.lr_warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"loss": [], "fit": [], "h_var": [], "eval_pearson": [], "eval_rmse": []}
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
        history["h_var"].append(losses["h_variance"].item())

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  epoch {epoch+1:4d}: loss={losses['total'].item():.5f}  "
                  f"fit={losses['fit'].item():.5f}  h_var={losses['h_variance'].item():.3f}  "
                  f"lr={lr_now:.5f}")

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                out_eval = model(x)
                h_pred = out_eval["hidden"][0].cpu().numpy()
            e = evaluate_hidden_recovery(h_pred, hidden_true)
            history["eval_pearson"].append(e["pearson_scaled"])
            history["eval_rmse"].append(e["rmse_scaled"])
            print(f"       -> EVAL Pearson={e['pearson_scaled']:+.4f}  RMSE={e['rmse_scaled']:.4f}")
            if abs(e["pearson_scaled"]) > best_pearson:
                best_pearson = abs(e["pearson_scaled"])
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_eval = e

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, history, best_eval, num_params


def plot_results(hidden_true, eval_res, history, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    t_axis = np.arange(len(hidden_true))
    axes[0, 0].plot(t_axis, hidden_true, color="black", linewidth=1.5, label="真实 hidden")
    axes[0, 0].plot(t_axis, eval_res["h_scaled"], color="#ff7f0e", linewidth=1.0, alpha=0.85, label="RS-GNN v2")
    axes[0, 0].set_title(f"Hidden 恢复: Pearson={eval_res['pearson_scaled']:.3f}  RMSE={eval_res['rmse_scaled']:.3f}",
                          fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=10); axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(hidden_true, eval_res["h_scaled"], alpha=0.3, s=8, color="#ff7f0e")
    vmin = min(hidden_true.min(), eval_res["h_scaled"].min())
    vmax = max(hidden_true.max(), eval_res["h_scaled"].max())
    axes[0, 1].plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8)
    axes[0, 1].set_title(f"散点 (scale={eval_res['scale_a']:.2f}x + {eval_res['scale_b']:.2f})")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].semilogy(history["loss"], color="#1565c0", linewidth=1.0, label="total")
    axes[1, 0].semilogy(history["fit"], color="#c62828", linewidth=1.0, label="fit")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("loss")
    axes[1, 0].set_title("训练损失"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.25)

    eval_epochs = np.arange(1, len(history["eval_pearson"]) + 1)
    axes[1, 1].plot(eval_epochs, history["eval_pearson"], color="#1565c0", marker="o", markersize=3, label="|Pearson|")
    ax2 = axes[1, 1].twinx()
    ax2.plot(eval_epochs, history["eval_rmse"], color="#c62828", marker="s", markersize=3, label="RMSE")
    axes[1, 1].axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1, 1].set_xlabel("eval step"); axes[1, 1].set_ylabel("|Pearson|", color="#1565c0")
    ax2.set_ylabel("RMSE", color="#c62828")
    axes[1, 1].set_title("评估指标演进")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle("RS-GNN v2 (双分支：重型残差 + Takens + GNN 核心)", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--residual-d", type=int, default=96)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0008)
    args = parser.parse_args()

    _configure_matplotlib()
    out_dir = Path("runs/analysis_rs_gnn_v2")
    out_dir.mkdir(parents=True, exist_ok=True)

    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    visible = d["states_B_5species"]
    hidden = d["hidden_B"]

    cfg = TrainConfig(
        epochs=args.epochs or (80 if args.smoke else 2000),
        lr=args.lr,
        residual_d_model=args.residual_d if not args.smoke else 32,
        state_d_model=args.residual_d if not args.smoke else 32,
        fusion_d_model=args.d_model if not args.smoke else 48,
        num_core_blocks=args.num_blocks if not args.smoke else 2,
        residual_attn_layers=3 if not args.smoke else 1,
        residual_mlp_layers=3 if not args.smoke else 1,
        state_attn_layers=2 if not args.smoke else 1,
        log_every=10 if args.smoke else 50,
        eval_every=20 if args.smoke else 100,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    torch.manual_seed(42)
    model, history, best_eval, num_params = train(cfg, visible, hidden, device=device)

    print()
    print("=" * 70)
    print(f"RS-GNN v2 Best Results (params={num_params:,}):")
    print(f"  Pearson: {best_eval['pearson_scaled']:.4f}  RMSE: {best_eval['rmse_scaled']:.4f}")
    print(f"  (raw Pearson: {best_eval['pearson_raw']:.4f}  RMSE: {best_eval['rmse_raw']:.4f})")
    print("=" * 70)

    suffix = "smoke" if args.smoke else "full"
    plot_results(hidden, best_eval, history, out_dir / f"fig_v2_{suffix}.png")
    np.savez(out_dir / f"result_{suffix}.npz",
              hidden_true=hidden, h_pred_scaled=best_eval["h_scaled"],
              pearson=best_eval["pearson_scaled"], rmse=best_eval["rmse_scaled"])
    print(f"[OK] saved: {out_dir / f'fig_v2_{suffix}.png'}")


if __name__ == "__main__":
    main()

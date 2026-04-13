"""严格无 hidden 监督训练 RS-GNN。

保证:
  1. 神经网络输入只有 visible_states，无任何 hidden 数据
  2. Loss 只来自 visible reconstruction + 结构正则，无 hidden 真值
  3. Model selection（早停）基于 visible reconstruction 在 val 段上的 loss，
     不基于 hidden recovery 指标
  4. hidden_true 只在最终 evaluation 时使用，且不影响训练
  5. Train/Val split: 前 P% 时间步 = train, 后 (100-P)% = val (held-out)
  6. 输出目录带时间戳

数据流确认:
  visible_states (B, T, N) → Model → hidden (B, T) + reconstructed (B, T-1, N)
  Loss = MSE(reconstructed_train, actual_train) + reg
  Val metric = MSE(reconstructed_val, actual_val)  ← 用于早停
  Final eval (仅用于报告) = Pearson(hidden, hidden_true) ← 不影响训练
"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.rs_gnn import RSGNN
from models.rs_gnn_v2 import RSGNNv2


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class StrictTrainConfig:
    epochs: int = 3000
    lr: float = 0.001
    lr_warmup_steps: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0

    # Time split for train/val
    train_ratio: float = 0.75   # 前 75% 训练，后 25% 验证

    # Architecture
    arch: str = "v1"            # "v1" or "v2"
    d_model: int = 128
    num_heads: int = 8
    num_blocks: int = 4
    delay_steps: int = 6
    dropout: float = 0.1

    # Regularization
    smooth_lambda: float = 0.05
    sparse_lambda: float = 0.001
    var_lambda: float = 0.2

    # Logging
    log_every: int = 50
    eval_every: int = 50

    seed: int = 42


def compute_visible_loss_on_segment(
    outputs: Dict[str, torch.Tensor],
    start_idx: int,
    end_idx: int,
) -> torch.Tensor:
    """在指定时间段上计算 visible reconstruction loss (log-ratio MSE)."""
    actual = outputs["actual_log_ratio"]  # (B, T-1, N)
    reconstructed = outputs["reconstructed_log_ratio"]  # (B, T-1, N)
    # Slice: log-ratio index t corresponds to state transitions [t, t+1]
    # If we want loss on time range [start, end), we need transitions that both
    # current and next fall within; so transitions t in [start, end-1)
    s = max(0, start_idx)
    e = min(actual.shape[1], end_idx - 1)
    if e <= s:
        return torch.tensor(0.0, device=actual.device)
    return torch.nn.functional.mse_loss(reconstructed[:, s:e], actual[:, s:e])


def evaluate_final(h_pred: np.ndarray, hidden_true: np.ndarray) -> Dict:
    """仅用于最终报告，不影响训练。"""
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


def build_model(cfg: StrictTrainConfig, T: int, N: int) -> torch.nn.Module:
    if cfg.arch == "v1":
        return RSGNN(
            num_visible=N, num_steps=T,
            delay_steps=cfg.delay_steps,
            d_model=cfg.d_model,
            num_heads=cfg.num_heads,
            num_blocks=cfg.num_blocks,
            dropout=cfg.dropout,
        )
    elif cfg.arch == "v2":
        return RSGNNv2(
            num_visible=N, num_steps=T,
            residual_d_model=cfg.d_model,
            state_d_model=cfg.d_model,
            fusion_d_model=cfg.d_model,
            num_core_blocks=cfg.num_blocks,
            num_heads=cfg.num_heads,
            dropout=cfg.dropout,
        )
    else:
        raise ValueError(f"Unknown arch: {cfg.arch}")


def train_strict(
    cfg: StrictTrainConfig,
    visible_states: np.ndarray,       # (T, N) 输入模型的数据
    hidden_true: np.ndarray,          # (T,) 仅用于最终评估，NEVER touched in training
    device: str = "cpu",
):
    """严格无 hidden 监督训练。"""
    T, N = visible_states.shape
    train_end = int(cfg.train_ratio * T)
    val_start = train_end
    val_end = T
    print(f"Time split: [0, {train_end}) train | [{val_start}, {val_end}) val")

    x = torch.tensor(visible_states, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(cfg.seed)
    model = build_model(cfg, T, N).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"{cfg.arch} params: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(step):
        if step < cfg.lr_warmup_steps:
            return step / cfg.lr_warmup_steps
        p = (step - cfg.lr_warmup_steps) / max(1, cfg.epochs - cfg.lr_warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "train_fit": [], "val_fit": [], "h_var": [],
        "eval_pearson": [], "eval_rmse": [], "eval_epoch": [],
    }
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)

        # === 核心：loss 只在 train 段的 visible 重构上算 ===
        train_fit = compute_visible_loss_on_segment(out, start_idx=0, end_idx=train_end)

        # Regularizers (structural, 不用任何 hidden 真值)
        hidden = out["hidden"]
        smooth = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        # Sparse baseline A
        if cfg.arch == "v1":
            A = model.A55
        else:
            A = model.baseline.A55
        A_offdiag = A - torch.diag(torch.diag(A))
        sparse = A_offdiag.abs().mean()
        h_var = hidden.var(dim=-1).mean()
        var_loss = torch.nn.functional.relu(0.05 - h_var)

        total = train_fit + cfg.smooth_lambda * smooth + cfg.sparse_lambda * sparse + cfg.var_lambda * var_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        # Val loss (仅用于早停，无 hidden 监督)
        with torch.no_grad():
            val_fit = compute_visible_loss_on_segment(out, start_idx=val_start, end_idx=val_end).item()

        history["train_fit"].append(train_fit.item())
        history["val_fit"].append(val_fit)
        history["h_var"].append(h_var.item())

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {epoch+1:4d}: train_fit={train_fit.item():.5f}  val_fit={val_fit:.5f}  "
                  f"h_var={h_var.item():.3f}  smooth={smooth.item():.3f}  lr={lr_now:.5f}")

        # Early stopping: 基于 val visible reconstruction loss (无 hidden 监督)
        if val_fit < best_val_loss:
            best_val_loss = val_fit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        # Final evaluation of hidden (仅用于监控，不用于 model selection)
        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                out_e = model(x)
                h_pred = out_e["hidden"][0].cpu().numpy()
            e = evaluate_final(h_pred, hidden_true)
            history["eval_pearson"].append(e["pearson_scaled"])
            history["eval_rmse"].append(e["rmse_scaled"])
            history["eval_epoch"].append(epoch + 1)
            print(f"       [monitor] Pearson={e['pearson_scaled']:+.4f} RMSE={e['rmse_scaled']:.4f} "
                  f"(仅监控，不影响训练)")
            model.train()

    # 载入 best state（基于 val visible loss 选择）
    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest epoch (by val visible loss): {best_epoch}")

    # 最终评估
    model.eval()
    with torch.no_grad():
        out_final = model(x)
        h_pred_final = out_final["hidden"][0].cpu().numpy()
    final_eval = evaluate_final(h_pred_final, hidden_true)

    return model, history, final_eval, num_params, best_epoch


def plot_results(hidden_true, eval_res, history, best_epoch, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    t_axis = np.arange(len(hidden_true))
    axes[0, 0].plot(t_axis, hidden_true, color="black", linewidth=1.5, label="真实 hidden")
    axes[0, 0].plot(t_axis, eval_res["h_scaled"], color="#ff7f0e", linewidth=1.0, alpha=0.85, label="恢复 hidden")
    axes[0, 0].set_title(f"Hidden 恢复 (严格无监督): Pearson={eval_res['pearson_scaled']:.3f}  RMSE={eval_res['rmse_scaled']:.3f}",
                          fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=10); axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(hidden_true, eval_res["h_scaled"], alpha=0.3, s=8, color="#ff7f0e")
    vmin, vmax = hidden_true.min(), hidden_true.max()
    axes[0, 1].plot([vmin, vmax], [vmin, vmax], "k--", linewidth=0.8)
    axes[0, 1].set_xlabel("真实"); axes[0, 1].set_ylabel("恢复 (scaled)")
    axes[0, 1].set_title(f"Scatter (best by val loss @ epoch {best_epoch})")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].semilogy(history["train_fit"], color="#1565c0", linewidth=1.0, label="train visible loss")
    axes[1, 0].semilogy(history["val_fit"], color="#c62828", linewidth=1.0, label="val visible loss (早停依据)")
    axes[1, 0].axvline(best_epoch - 1, color="green", linestyle="--", linewidth=0.8, alpha=0.7, label=f"best @ {best_epoch}")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("visible reconstruction loss")
    axes[1, 0].set_title("Train/Val 重构损失（用于无监督早停）")
    axes[1, 0].legend(fontsize=9); axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(history["eval_epoch"], history["eval_pearson"], color="#1565c0", marker="o", markersize=3, label="|Pearson|")
    ax2 = axes[1, 1].twinx()
    ax2.plot(history["eval_epoch"], history["eval_rmse"], color="#c62828", marker="s", markersize=3, label="RMSE")
    axes[1, 1].axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1, 1].axvline(best_epoch, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("|Pearson|", color="#1565c0")
    ax2.set_ylabel("RMSE", color="#c62828")
    axes[1, 1].set_title("Hidden 恢复质量（仅监控）")
    axes[1, 1].grid(alpha=0.25)

    fig.suptitle(f"严格无监督 RS-GNN 训练结果 (best epoch={best_epoch} by val visible loss)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--arch", type=str, default="v1", choices=["v1", "v2"])
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--smooth-lambda", type=float, default=0.05)
    parser.add_argument("--var-lambda", type=float, default=0.2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_matplotlib()

    # 时间戳输出目录
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_rs_gnn_strict_{args.arch}")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"输出目录: {out_dir}")

    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    visible = d["states_B_5species"]
    hidden = d["hidden_B"]
    print(f"Data: T={visible.shape[0]}, N={visible.shape[1]}")
    print(f"Hidden (仅最终评估用): range=[{hidden.min():.3f}, {hidden.max():.3f}] std={hidden.std():.3f}")

    cfg = StrictTrainConfig(
        epochs=args.epochs or (150 if args.smoke else 2000),
        lr=args.lr,
        arch=args.arch,
        d_model=args.d_model if not args.smoke else 48,
        num_heads=args.num_heads if not args.smoke else 4,
        num_blocks=args.num_blocks if not args.smoke else 2,
        train_ratio=args.train_ratio,
        smooth_lambda=args.smooth_lambda,
        var_lambda=args.var_lambda,
        dropout=args.dropout,
        log_every=10 if args.smoke else 50,
        eval_every=20 if args.smoke else 50,
        seed=args.seed,
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: {cfg}\n")

    model, history, eval_res, num_params, best_epoch = train_strict(cfg, visible, hidden, device=device)

    print()
    print("=" * 70)
    print(f"FINAL (严格无监督, best by val loss @ epoch {best_epoch}):")
    print(f"  Pearson (scaled): {eval_res['pearson_scaled']:.4f}")
    print(f"  RMSE (scaled):    {eval_res['rmse_scaled']:.4f}")
    print(f"  Params: {num_params:,}")
    print("=" * 70)

    suffix = "smoke" if args.smoke else "full"
    plot_results(hidden, eval_res, history, best_epoch, out_dir / f"fig_{suffix}.png")
    np.savez(out_dir / f"result_{suffix}.npz",
              hidden_true=hidden, h_pred_scaled=eval_res["h_scaled"],
              pearson=eval_res["pearson_scaled"], rmse=eval_res["rmse_scaled"],
              best_epoch=best_epoch)
    # 保存 config
    with open(out_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(f"Architecture: {cfg.arch}\n")
        f.write(f"d_model: {cfg.d_model}\n")
        f.write(f"num_blocks: {cfg.num_blocks}\n")
        f.write(f"num_heads: {cfg.num_heads}\n")
        f.write(f"epochs: {cfg.epochs}\n")
        f.write(f"lr: {cfg.lr}\n")
        f.write(f"train_ratio: {cfg.train_ratio}\n")
        f.write(f"smooth_lambda: {cfg.smooth_lambda}\n")
        f.write(f"var_lambda: {cfg.var_lambda}\n")
        f.write(f"seed: {cfg.seed}\n")
        f.write(f"num_params: {num_params}\n")
        f.write(f"\nFinal Pearson: {eval_res['pearson_scaled']:.4f}\n")
        f.write(f"Final RMSE: {eval_res['rmse_scaled']:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
    print(f"[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

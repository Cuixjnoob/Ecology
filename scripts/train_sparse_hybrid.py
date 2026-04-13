"""训练 SparseHybridGNN（严格无 hidden 监督，时间戳目录）。"""
from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.sparse_hybrid_gnn import SparseHybridGNN


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class TrainConfig:
    epochs: int = 1500
    lr: float = 0.0008
    lr_warmup_steps: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_ratio: float = 0.75

    baseline_d_hidden: int = 32
    baseline_num_layers: int = 2
    baseline_top_k: int = 3

    hidden_d_model: int = 128
    hidden_num_blocks: int = 4
    hidden_num_heads: int = 8
    hidden_takens_lags: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    dropout: float = 0.1
    sparse_lambda: float = 0.5    # 关键！
    smooth_lambda: float = 0.02
    var_lambda: float = 0.2

    log_every: int = 50
    eval_every: int = 50
    seed: int = 42


def compute_visible_loss_on_segment(pred, actual, start_idx, end_idx):
    s = max(0, start_idx)
    e = min(actual.shape[1], end_idx - 1)
    if e <= s:
        return torch.tensor(0.0, device=actual.device)
    return torch.nn.functional.mse_loss(pred[:, s:e], actual[:, s:e])


def evaluate_final(h_pred, hidden_true):
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


def train(cfg, visible_states, hidden_true, device="cpu"):
    T, N = visible_states.shape
    train_end = int(cfg.train_ratio * T)
    print(f"Time split: train [0, {train_end}), val [{train_end}, {T})")

    x = torch.tensor(visible_states, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(cfg.seed)
    model = SparseHybridGNN(
        num_visible=N,
        baseline_d_hidden=cfg.baseline_d_hidden,
        baseline_num_layers=cfg.baseline_num_layers,
        baseline_top_k=cfg.baseline_top_k,
        hidden_takens_lags=cfg.hidden_takens_lags,
        hidden_d_model=cfg.hidden_d_model,
        hidden_num_blocks=cfg.hidden_num_blocks,
        hidden_num_heads=cfg.hidden_num_heads,
        dropout=cfg.dropout,
    ).to(device)

    baseline_params = sum(p.numel() for p in model.baseline_gnn.parameters())
    decoder_params = sum(p.numel() for p in model.hidden_decoder.parameters())
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Params: baseline={baseline_params:,}  decoder={decoder_params:,}  total={total_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(step):
        if step < cfg.lr_warmup_steps:
            return step / cfg.lr_warmup_steps
        p = (step - cfg.lr_warmup_steps) / max(1, cfg.epochs - cfg.lr_warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {
        "train_fit": [], "val_fit": [], "sparse": [], "h_var": [],
        "eval_pearson": [], "eval_rmse": [], "eval_epoch": [],
    }
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)

        # Train loss on train segment only
        train_fit = compute_visible_loss_on_segment(
            out["predicted_log_ratio"], out["actual_log_ratio"],
            start_idx=0, end_idx=train_end,
        )

        # Regularizers
        sparse_reg = model.baseline_gnn.l1_regularization()
        hidden = out["hidden"]
        smooth = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        h_var = hidden.var(dim=-1).mean()
        var_loss = torch.nn.functional.relu(0.05 - h_var)

        total = train_fit + cfg.sparse_lambda * sparse_reg + cfg.smooth_lambda * smooth + cfg.var_lambda * var_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            val_fit = compute_visible_loss_on_segment(
                out["predicted_log_ratio"], out["actual_log_ratio"],
                start_idx=train_end, end_idx=T,
            ).item()

        history["train_fit"].append(train_fit.item())
        history["val_fit"].append(val_fit)
        history["sparse"].append(sparse_reg.item())
        history["h_var"].append(h_var.item())

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"  ep {epoch+1:4d}: train_fit={train_fit.item():.5f}  val_fit={val_fit:.5f}  "
                  f"sparse={sparse_reg.item():.4f}  h_var={h_var.item():.3f}  lr={lr_now:.5f}")

        # Early stopping by val visible loss
        if val_fit < best_val_loss:
            best_val_loss = val_fit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                out_e = model(x)
                h_pred = out_e["hidden"][0].cpu().numpy()
            e = evaluate_final(h_pred, hidden_true)
            history["eval_pearson"].append(e["pearson_scaled"])
            history["eval_rmse"].append(e["rmse_scaled"])
            history["eval_epoch"].append(epoch + 1)
            print(f"       [monitor] Pearson={e['pearson_scaled']:+.4f}  RMSE={e['rmse_scaled']:.4f}")
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"\nBest epoch (by val visible loss): {best_epoch}")

    # Final eval
    model.eval()
    with torch.no_grad():
        out_final = model(x)
        h_pred_final = out_final["hidden"][0].cpu().numpy()
    final_eval = evaluate_final(h_pred_final, hidden_true)
    return model, history, final_eval, best_epoch, total_params


def plot_results(hidden_true, eval_res, history, best_epoch, out_path):
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    t_axis = np.arange(len(hidden_true))
    axes[0, 0].plot(t_axis, hidden_true, color="black", linewidth=1.5, label="真实 hidden")
    axes[0, 0].plot(t_axis, eval_res["h_scaled"], color="#ff7f0e", linewidth=1.0, alpha=0.85, label="恢复")
    axes[0, 0].set_title(f"Hidden 恢复: Pearson={eval_res['pearson_scaled']:.3f}  RMSE={eval_res['rmse_scaled']:.3f}",
                          fontsize=12, fontweight="bold")
    axes[0, 0].legend(fontsize=10); axes[0, 0].grid(alpha=0.25)

    axes[0, 1].scatter(hidden_true, eval_res["h_scaled"], alpha=0.3, s=8, color="#ff7f0e")
    vmin, vmax = hidden_true.min(), hidden_true.max()
    axes[0, 1].plot([vmin, vmax], [vmin, vmax], "k--")
    axes[0, 1].set_title(f"Scatter (best by val loss @ ep {best_epoch})")
    axes[0, 1].grid(alpha=0.25)

    axes[1, 0].semilogy(history["train_fit"], color="#1565c0", label="train")
    axes[1, 0].semilogy(history["val_fit"], color="#c62828", label="val (早停)")
    axes[1, 0].axvline(best_epoch - 1, color="green", linestyle="--", linewidth=0.8)
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("visible loss")
    axes[1, 0].set_title("Train/Val 重构损失"); axes[1, 0].legend(); axes[1, 0].grid(alpha=0.25)

    axes[1, 1].plot(history["eval_epoch"], history["eval_pearson"], color="#1565c0", marker="o", markersize=3, label="|Pearson|")
    ax2 = axes[1, 1].twinx()
    ax2.plot(history["eval_epoch"], history["eval_rmse"], color="#c62828", marker="s", markersize=3, label="RMSE")
    axes[1, 1].axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1, 1].axvline(best_epoch, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1, 1].set_title("Hidden 质量（仅监控）"); axes[1, 1].grid(alpha=0.25)

    fig.suptitle(f"SparseHybrid GNN (Baseline GNN + Hidden Decoder GNN) best @ ep {best_epoch}",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--hidden-d", type=int, default=128)
    parser.add_argument("--hidden-blocks", type=int, default=4)
    parser.add_argument("--baseline-d", type=int, default=32)
    parser.add_argument("--baseline-top-k", type=int, default=3)
    parser.add_argument("--baseline-layers", type=int, default=2)
    parser.add_argument("--sparse-lambda", type=float, default=0.5)
    parser.add_argument("--smooth-lambda", type=float, default=0.02)
    parser.add_argument("--var-lambda", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.0008)
    parser.add_argument("--train-ratio", type=float, default=0.75)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_sparse_hybrid_gnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")

    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    visible = d["states_B_5species"]
    hidden = d["hidden_B"]
    print(f"Data: T={visible.shape[0]}, N={visible.shape[1]}")

    cfg = TrainConfig(
        epochs=args.epochs or (150 if args.smoke else 1500),
        lr=args.lr,
        train_ratio=args.train_ratio,
        baseline_d_hidden=args.baseline_d if not args.smoke else 16,
        baseline_num_layers=args.baseline_layers,
        baseline_top_k=args.baseline_top_k,
        hidden_d_model=args.hidden_d if not args.smoke else 48,
        hidden_num_blocks=args.hidden_blocks if not args.smoke else 2,
        sparse_lambda=args.sparse_lambda,
        smooth_lambda=args.smooth_lambda,
        var_lambda=args.var_lambda,
        log_every=10 if args.smoke else 50,
        eval_every=20 if args.smoke else 50,
        seed=args.seed,
    )
    print(f"Config: {cfg}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    model, history, eval_res, best_epoch, num_params = train(cfg, visible, hidden, device=device)

    print()
    print("=" * 70)
    print(f"FINAL (sparse hybrid, 严格无监督, best @ ep {best_epoch}):")
    print(f"  Pearson (scaled): {eval_res['pearson_scaled']:.4f}")
    print(f"  RMSE (scaled):    {eval_res['rmse_scaled']:.4f}")
    print(f"  Pearson (raw):    {eval_res['pearson_raw']:.4f}")
    print("=" * 70)

    suffix = "smoke" if args.smoke else "full"
    plot_results(hidden, eval_res, history, best_epoch, out_dir / f"fig_{suffix}.png")
    np.savez(out_dir / f"result_{suffix}.npz",
              hidden_true=hidden, h_pred_scaled=eval_res["h_scaled"],
              pearson=eval_res["pearson_scaled"], rmse=eval_res["rmse_scaled"],
              best_epoch=best_epoch)

    with open(out_dir / "config.txt", "w", encoding="utf-8") as f:
        f.write(f"baseline: d_hidden={cfg.baseline_d_hidden}, layers={cfg.baseline_num_layers}, top_k={cfg.baseline_top_k}\n")
        f.write(f"hidden_decoder: d_model={cfg.hidden_d_model}, blocks={cfg.hidden_num_blocks}, heads={cfg.hidden_num_heads}\n")
        f.write(f"takens_lags: {cfg.hidden_takens_lags}\n")
        f.write(f"sparse_lambda={cfg.sparse_lambda}, smooth_lambda={cfg.smooth_lambda}, var_lambda={cfg.var_lambda}\n")
        f.write(f"epochs={cfg.epochs}, lr={cfg.lr}, train_ratio={cfg.train_ratio}, seed={cfg.seed}\n")
        f.write(f"num_params={num_params}\n")
        f.write(f"\nFinal Pearson: {eval_res['pearson_scaled']:.4f}\n")
        f.write(f"Final RMSE: {eval_res['rmse_scaled']:.4f}\n")
        f.write(f"Best epoch: {best_epoch}\n")
    print(f"[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

"""训练 HNSR (Hybrid Neural-Sparse Recovery) — 严格无 hidden 监督。

融合方法:
  Linear Sparse Baseline
  + Hand-crafted Library (LV/Holling/Allee basis)
  + GNN Basis Discovery (learned basis)
  + GNN Hidden Correction

在 LV 和 Holling 数据上分别测试。
"""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.hnsr import HNSR


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class TrainConfig:
    epochs: int = 1500
    lr: float = 0.001
    lr_warmup_steps: int = 150
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_ratio: float = 0.75

    num_learned_basis: int = 6
    takens_lags: List[int] = field(default_factory=lambda: [1, 2, 4, 8])
    basis_gnn_d: int = 32
    basis_gnn_layers: int = 2
    basis_gnn_top_k: int = 3
    correction_d: int = 64
    correction_blocks: int = 2
    correction_heads: int = 4
    dropout: float = 0.1

    lam_A_sparse: float = 0.3
    lam_crafted_sparse: float = 0.05
    lam_learned_sparse: float = 0.05
    lam_smooth: float = 0.02
    lam_correction_mag: float = 0.05
    lam_var: float = 0.15

    log_every: int = 50
    eval_every: int = 50
    seed: int = 42


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
        "h_scaled": h_scaled, "scale_a": float(coef[0]), "scale_b": float(coef[1]),
    }


def train_on_dataset(cfg, visible, hidden_true, device="cpu", label=""):
    T, N = visible.shape
    train_end = int(cfg.train_ratio * T)
    print(f"  Time split: train [0, {train_end}), val [{train_end}, {T})")

    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(cfg.seed)
    model = HNSR(
        num_visible=N,
        num_learned_basis=cfg.num_learned_basis,
        takens_lags=cfg.takens_lags,
        basis_gnn_d=cfg.basis_gnn_d,
        basis_gnn_layers=cfg.basis_gnn_layers,
        basis_gnn_top_k=cfg.basis_gnn_top_k,
        correction_d=cfg.correction_d,
        correction_blocks=cfg.correction_blocks,
        correction_heads=cfg.correction_heads,
        dropout=cfg.dropout,
    ).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Params: {num_params:,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    def lr_lambda(step):
        if step < cfg.lr_warmup_steps:
            return step / cfg.lr_warmup_steps
        p = (step - cfg.lr_warmup_steps) / max(1, cfg.epochs - cfg.lr_warmup_steps)
        return 0.5 * (1 + np.cos(np.pi * p))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    history = {"train_fit": [], "val_fit": [], "h_var": [], "eval_pearson": [], "eval_rmse": [], "eval_epoch": []}
    best_val_loss = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        optimizer.zero_grad()
        out = model(x)
        losses = model.compute_loss(
            out,
            lam_A_sparse=cfg.lam_A_sparse,
            lam_crafted_sparse=cfg.lam_crafted_sparse,
            lam_learned_sparse=cfg.lam_learned_sparse,
            lam_smooth=cfg.lam_smooth,
            lam_correction_mag=cfg.lam_correction_mag,
            lam_var=cfg.lam_var,
        )

        # Train on train segment
        actual = out["actual_log_ratio"]
        pred = out["predicted_log_ratio"]
        train_fit = torch.nn.functional.mse_loss(pred[:, :train_end-1], actual[:, :train_end-1])
        # Regularizers (computed above in `losses`)
        total = train_fit + (losses["total"] - losses["fit"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            val_fit = torch.nn.functional.mse_loss(pred[:, train_end-1:], actual[:, train_end-1:]).item()

        history["train_fit"].append(train_fit.item())
        history["val_fit"].append(val_fit)
        history["h_var"].append(losses["h_variance"].item())

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            lr_now = optimizer.param_groups[0]["lr"]
            print(f"    ep {epoch+1:4d}: train_fit={train_fit.item():.5f}  val_fit={val_fit:.5f}  "
                  f"h_var={losses['h_variance'].item():.3f}  lr={lr_now:.5f}")

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
            print(f"         [monitor] Pearson={e['pearson_scaled']:+.4f}  RMSE={e['rmse_scaled']:.4f}")
            model.train()

    if best_state is not None:
        model.load_state_dict(best_state)
    print(f"  Best epoch (by val fit): {best_epoch}")

    model.eval()
    with torch.no_grad():
        out_final = model(x)
        h_pred_final = out_final["hidden"][0].cpu().numpy()
    eval_res = evaluate_final(h_pred_final, hidden_true)
    return model, history, eval_res, best_epoch, num_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--correction-scale", type=float, default=1.0)
    args = parser.parse_args()

    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_hnsr_hybrid")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    cfg_lv = TrainConfig(
        epochs=args.epochs or (100 if args.smoke else 800),
        lr=args.lr,
        basis_gnn_d=32 if not args.smoke else 16,
        correction_d=64 if not args.smoke else 32,
        correction_blocks=2 if not args.smoke else 1,
        log_every=10 if args.smoke else 50,
        eval_every=20 if args.smoke else 50,
    )
    cfg_holling = cfg_lv

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    print(f"Config: epochs={cfg_lv.epochs}, lr={cfg_lv.lr}\n")

    results = {}

    # LV
    print("=" * 70)
    print("Training on LV data")
    print("=" * 70)
    d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
    _, hist_lv, eval_lv, ep_lv, _ = train_on_dataset(cfg_lv, d_lv["states_B_5species"], d_lv["hidden_B"], device=device, label="LV")
    results["LV"] = {"hist": hist_lv, "eval": eval_lv, "best_epoch": ep_lv, "hidden_true": d_lv["hidden_B"]}

    # Holling
    print()
    print("=" * 70)
    print("Training on Holling data")
    print("=" * 70)
    holling_files = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
    d_h = np.load(holling_files[-1])
    _, hist_h, eval_h, ep_h, _ = train_on_dataset(cfg_holling, d_h["states_B_5species"], d_h["hidden_B"], device=device, label="Holling")
    results["Holling"] = {"hist": hist_h, "eval": eval_h, "best_epoch": ep_h, "hidden_true": d_h["hidden_B"]}

    print()
    print("=" * 70)
    print("HNSR FINAL RESULTS:")
    for label, res in results.items():
        e = res["eval"]
        print(f"  {label}: Pearson={e['pearson_scaled']:.4f}  RMSE={e['rmse_scaled']:.4f}")
    print("=" * 70)

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for idx, (label, res) in enumerate(results.items()):
        ax = axes[0, idx]
        hidden_true = res["hidden_true"]
        e = res["eval"]
        t_axis = np.arange(len(hidden_true))
        ax.plot(t_axis, hidden_true, color="black", linewidth=1.2, label="真实")
        ax.plot(t_axis, e["h_scaled"], color="#ff7f0e", linewidth=1.0, alpha=0.85, label="HNSR")
        ax.set_title(f"{label}: Pearson={e['pearson_scaled']:.3f}  RMSE={e['rmse_scaled']:.3f}", fontsize=12, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

        ax = axes[1, idx]
        h = res["hist"]
        ax.semilogy(h["train_fit"], color="#1565c0", label="train")
        ax.semilogy(h["val_fit"], color="#c62828", label="val")
        ax.axvline(res["best_epoch"] - 1, color="green", linestyle="--", linewidth=0.8)
        ax.set_title(f"{label}: Loss curves (best @ ep {res['best_epoch']})")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

    fig.suptitle("HNSR: Hybrid Neural-Sparse Recovery 结果", fontsize=14, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_hnsr.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    np.savez(out_dir / "results.npz",
              lv_pearson=results["LV"]["eval"]["pearson_scaled"],
              lv_rmse=results["LV"]["eval"]["rmse_scaled"],
              holling_pearson=results["Holling"]["eval"]["pearson_scaled"],
              holling_rmse=results["Holling"]["eval"]["rmse_scaled"])

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# HNSR: Hybrid Neural-Sparse Recovery 结果\n\n")
        for label, res in results.items():
            e = res["eval"]
            f.write(f"## {label}\n\n")
            f.write(f"- Pearson (scaled): {e['pearson_scaled']:.4f}\n")
            f.write(f"- RMSE (scaled): {e['rmse_scaled']:.4f}\n")
            f.write(f"- Best epoch: {res['best_epoch']}\n\n")
    print(f"[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

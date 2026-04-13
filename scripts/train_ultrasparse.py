"""训练 UltraSparseHiddenModel — 容量和 linear sparse 同量级的 GNN。

2x2 实验: LV/Holling × UltraSparseGNN
对比 Linear Sparse 的结果。
"""
from __future__ import annotations

import argparse
import glob
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from models.ultrasparse_gnn import UltraSparseHiddenModel


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


@dataclass
class Config:
    epochs: int = 2000
    lr: float = 0.005
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_ratio: float = 0.75

    d_hidden: int = 16
    num_layers: int = 2
    top_k: int = 2
    lam_l1: float = 0.05
    lam_smooth: float = 0.02
    lam_var: float = 0.15

    log_every: int = 100
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
    return {"pearson_raw": pearson_raw, "rmse_raw": rmse_raw,
            "pearson_scaled": pearson_scaled, "rmse_scaled": rmse_scaled,
            "h_scaled": h_scaled}


def train_on_dataset(cfg, states, hidden_true, device, label):
    T, N = states.shape
    train_end = int(cfg.train_ratio * T)
    x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(cfg.seed)
    model = UltraSparseHiddenModel(
        num_visible=N, d_hidden=cfg.d_hidden, num_layers=cfg.num_layers, top_k=cfg.top_k,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {label}: UltraSparseGNN params = {num_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.epochs)

    history = {"train_fit": [], "val_fit": [], "eval_pearson": [], "eval_rmse": [], "eval_epoch": []}
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.epochs):
        model.train()
        opt.zero_grad()
        out = model(x)
        actual = out["actual_log_ratio"]
        pred = out["predicted_log_ratio"]
        hidden = out["hidden"]

        # Train loss on train segment
        train_fit = torch.nn.functional.mse_loss(pred[:, :train_end-1], actual[:, :train_end-1])
        l1 = model.baseline.l1_reg()
        smooth = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        h_var = hidden.var(dim=-1).mean()
        var_loss = torch.nn.functional.relu(0.05 - h_var)
        total = train_fit + cfg.lam_l1 * l1 + cfg.lam_smooth * smooth + cfg.lam_var * var_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        sched.step()

        with torch.no_grad():
            val_fit = torch.nn.functional.mse_loss(pred[:, train_end-1:], actual[:, train_end-1:]).item()

        history["train_fit"].append(train_fit.item())
        history["val_fit"].append(val_fit)

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            print(f"    ep {epoch+1:4d}: train={train_fit.item():.5f} val={val_fit:.5f} h_var={h_var.item():.3f} l1={l1.item():.3f}")

        if val_fit < best_val:
            best_val = val_fit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                o = model(x)
                h_pred = o["hidden"][0].cpu().numpy()
            h_true = hidden_true[:len(h_pred)]
            e = evaluate_final(h_pred, h_true)
            history["eval_pearson"].append(e["pearson_scaled"])
            history["eval_rmse"].append(e["rmse_scaled"])
            history["eval_epoch"].append(epoch + 1)
            if (epoch + 1) % cfg.log_every == 0:
                print(f"         [monitor] P={e['pearson_scaled']:+.4f} RMSE={e['rmse_scaled']:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        o = model(x)
        h_pred = o["hidden"][0].cpu().numpy()
    h_true = hidden_true[:len(h_pred)]
    eval_res = evaluate_final(h_pred, h_true)
    return model, history, eval_res, best_epoch, num_params


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--d-hidden", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--top-k", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.005)
    parser.add_argument("--lam-l1", type=float, default=0.05)
    args = parser.parse_args()

    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_ultrasparse_gnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = Config(
        epochs=args.epochs, lr=args.lr,
        d_hidden=args.d_hidden, num_layers=args.num_layers, top_k=args.top_k,
        lam_l1=args.lam_l1,
    )

    # Datasets
    d_lv = np.load("runs/analysis_5vs6_species/trajectories.npz")
    holling_dirs = sorted(glob.glob("runs/*_5vs6_holling/trajectories.npz"))
    d_h = np.load(holling_dirs[-1])
    datasets = {
        "LV": (d_lv["states_B_5species"], d_lv["hidden_B"]),
        "Holling": (d_h["states_B_5species"], d_h["hidden_B"]),
    }

    results = {}
    for label, (states, hidden) in datasets.items():
        print(f"\n{'='*70}\n{label} data (UltraSparseGNN)\n{'='*70}")
        model, hist, eval_res, best_epoch, num_params = train_on_dataset(cfg, states, hidden[:-1], device, label)
        results[label] = {"eval": eval_res, "hist": hist, "best_epoch": best_epoch, "num_params": num_params, "hidden_true": hidden[:-1]}
        print(f"\n  BEST: Pearson={eval_res['pearson_scaled']:.4f}  RMSE={eval_res['rmse_scaled']:.4f}  (epoch {best_epoch}, params {num_params})")

    # Compare with linear sparse (from previous results)
    print()
    print("=" * 70)
    print("COMPARISON vs Linear Sparse (previous):")
    print("=" * 70)
    print(f"  LV:      Linear Sparse Pearson=0.977, UltraSparseGNN Pearson={results['LV']['eval']['pearson_scaled']:.4f}")
    print(f"  Holling: Linear Sparse Pearson=0.876, UltraSparseGNN Pearson={results['Holling']['eval']['pearson_scaled']:.4f}")

    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for idx, (label, res) in enumerate(results.items()):
        ax = axes[0, idx]
        ht = res["hidden_true"]
        e = res["eval"]
        ax.plot(ht, color="black", linewidth=1.2, label="真实")
        ax.plot(e["h_scaled"], color="#ff7f0e", linewidth=1.0, alpha=0.85, label="UltraSparseGNN")
        ax.set_title(f"{label}: Pearson={e['pearson_scaled']:.3f}  RMSE={e['rmse_scaled']:.3f}  ({res['num_params']} params)", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(alpha=0.25)

        ax = axes[1, idx]
        h = res["hist"]
        ax.semilogy(h["train_fit"], color="#1565c0", label="train")
        ax.semilogy(h["val_fit"], color="#c62828", label="val")
        ax.axvline(res["best_epoch"] - 1, color="green", linestyle="--", linewidth=0.8)
        ax.set_title(f"{label}: loss (best @ {res['best_epoch']})")
        ax.legend(); ax.grid(alpha=0.25)

    fig.suptitle("UltraSparseGNN: 容量等价于 Linear Sparse, 但可拟合非线性", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_ultrasparse.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    np.savez(out_dir / "results.npz",
              lv_pearson=results["LV"]["eval"]["pearson_scaled"],
              lv_rmse=results["LV"]["eval"]["rmse_scaled"],
              holling_pearson=results["Holling"]["eval"]["pearson_scaled"],
              holling_rmse=results["Holling"]["eval"]["rmse_scaled"],
              lv_params=results["LV"]["num_params"],
              holling_params=results["Holling"]["num_params"])

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# UltraSparse GNN Baseline 结果\n\n")
        f.write("容量等价 linear sparse，但 GNN 结构可捕非线性。\n\n")
        for label, res in results.items():
            e = res["eval"]
            f.write(f"## {label}\n")
            f.write(f"- Params: {res['num_params']}\n")
            f.write(f"- Pearson: {e['pearson_scaled']:.4f}  RMSE: {e['rmse_scaled']:.4f}\n\n")
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

"""训练 Linear-Seeded GNN。

关键设计：Linear sparse 作为 GNN 的 anchor/input feature，GNN 学 residual correction。
严格无 hidden 监督。图每 PNG 一张。
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

from models.linear_seeded_gnn import LinearSeededGNN


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 12


@dataclass
class Config:
    epochs: int = 2000
    lr: float = 0.001
    lr_warmup: int = 200
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    train_ratio: float = 0.75
    d_model: int = 128
    num_blocks: int = 4
    num_heads: int = 8
    dropout: float = 0.1
    lam_A: float = 0.3
    lam_smooth: float = 0.02
    lam_correction: float = 0.01
    lam_var: float = 0.15
    log_every: int = 100
    eval_every: int = 50
    seed: int = 42


def evaluate_final(h_pred, hidden_true):
    L = min(len(h_pred), len(hidden_true))
    h_pred = h_pred[:L]
    hidden_true = hidden_true[:L]
    pearson_raw = float(np.corrcoef(h_pred, hidden_true)[0, 1])
    rmse_raw = float(np.sqrt(((h_pred - hidden_true) ** 2).mean()))
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((len(h_pred), 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    pearson_scaled = float(np.corrcoef(h_scaled, hidden_true)[0, 1])
    rmse_scaled = float(np.sqrt(((h_scaled - hidden_true) ** 2).mean()))
    return {"pearson_raw": pearson_raw, "rmse_raw": rmse_raw,
            "pearson_scaled": pearson_scaled, "rmse_scaled": rmse_scaled,
            "h_scaled": h_scaled, "hidden_true_aligned": hidden_true}


def train_on_dataset(cfg, states, hidden_true, device, label):
    T, N = states.shape
    train_end = int(cfg.train_ratio * T)
    x = torch.tensor(states, dtype=torch.float32, device=device).unsqueeze(0)

    torch.manual_seed(cfg.seed)
    model = LinearSeededGNN(
        num_visible=N, d_model=cfg.d_model, num_blocks=cfg.num_blocks,
        num_heads=cfg.num_heads, dropout=cfg.dropout,
    ).to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  {label}: params={num_params:,}")

    # Stagewise training strategy:
    # Phase 1 (epochs 0 to P1): 只训 linear sparse baseline (冻结 GNN)
    # Phase 2 (P1 to P2): 冻结 linear, 训练 GNN + hidden decoder + coupling
    # Phase 3 (P2+): 联合 fine-tune 小 lr
    P1 = cfg.epochs // 4    # 25% pure linear
    P2 = cfg.epochs * 3 // 4  # 50% GNN only
    print(f"  Training phases: Phase1=[0,{P1}) linear-only; Phase2=[{P1},{P2}) GNN-only; Phase3=[{P2},{cfg.epochs}) joint")

    linear_params = list(model.linear.parameters())
    other_params = [p for n, p in model.named_parameters() if not n.startswith("linear.")]
    opt_linear = torch.optim.AdamW(linear_params, lr=cfg.lr * 5, weight_decay=cfg.weight_decay)
    opt_other = torch.optim.AdamW(other_params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    def get_phase(epoch):
        return 1 if epoch < P1 else (2 if epoch < P2 else 3)

    history = {"train_fit": [], "val_fit": [], "alpha": [], "h_var": [],
               "eval_pearson": [], "eval_rmse": [], "eval_epoch": []}
    best_val = float("inf")
    best_state = None
    best_epoch = -1

    for epoch in range(cfg.epochs):
        phase = get_phase(epoch)
        model.train()
        # Phase 1: 仅 linear; Phase 2: 仅 GNN+coupling+hidden; Phase 3: joint
        opt_linear.zero_grad()
        opt_other.zero_grad()
        out = model(x)
        losses = model.compute_loss(out, lam_A=cfg.lam_A, lam_smooth=cfg.lam_smooth,
                                     lam_correction=cfg.lam_correction, lam_var=cfg.lam_var)
        actual = out["actual_log_ratio"]
        pred = out["predicted_log_ratio"]

        if phase == 1:
            # Phase 1: train linear only, use LINEAR prediction directly (bypass GNN)
            linear_pred = out["linear_pred"]
            train_fit = torch.nn.functional.mse_loss(linear_pred[:, :train_end-1], actual[:, :train_end-1])
            total = train_fit + cfg.lam_A * losses["l1_A"]
        else:
            train_fit = torch.nn.functional.mse_loss(pred[:, :train_end-1], actual[:, :train_end-1])
            total = train_fit + (losses["total"] - losses["fit"])
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if phase == 1:
            opt_linear.step()
        elif phase == 2:
            opt_other.step()
        else:
            opt_linear.step()
            opt_other.step()

        with torch.no_grad():
            val_fit = torch.nn.functional.mse_loss(pred[:, train_end-1:], actual[:, train_end-1:]).item()

        history["train_fit"].append(train_fit.item())
        history["val_fit"].append(val_fit)
        history["alpha"].append(float(losses["alpha"]))
        history["h_var"].append(losses["h_variance"].item())

        if (epoch + 1) % cfg.log_every == 0 or epoch == 0:
            print(f"    [phase{phase}] ep {epoch+1:4d}: train={train_fit.item():.5f} val={val_fit:.5f} "
                  f"α={float(losses['alpha']):.3f} h_var={losses['h_variance'].item():.3f} l1_A={losses['l1_A'].item():.3f}")

        if val_fit < best_val:
            best_val = val_fit
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.epochs - 1:
            model.eval()
            with torch.no_grad():
                o = model(x)
                h_pred = o["hidden"][0].cpu().numpy()
            h_true_aligned = hidden_true[:len(h_pred)]
            e = evaluate_final(h_pred, h_true_aligned)
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
    h_true_aligned = hidden_true[:len(h_pred)]
    eval_res = evaluate_final(h_pred, h_true_aligned)
    return model, history, eval_res, best_epoch, num_params, h_true_aligned, h_pred


def save_fig_single(fig_title, plot_fn, out_path, figsize=(10, 6)):
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    plot_fn(ax)
    fig.suptitle(fig_title, fontsize=13, fontweight="bold")
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--num-blocks", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.001)
    args = parser.parse_args()

    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_linear_seeded_gnn")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")

    cfg = Config(epochs=args.epochs, lr=args.lr, d_model=args.d_model, num_blocks=args.num_blocks)

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
        print(f"\n{'='*70}\n{label} data (Linear-Seeded GNN)\n{'='*70}")
        _, hist, e, ep, np_params, h_true, h_pred = train_on_dataset(
            cfg, states, hidden[:-1], device, label,
        )
        results[label] = {
            "eval": e, "hist": hist, "best_epoch": ep, "num_params": np_params,
            "hidden_true": h_true, "h_pred": h_pred,
        }
        print(f"\n  BEST: Pearson={e['pearson_scaled']:.4f} RMSE={e['rmse_scaled']:.4f} (ep {ep}, params {np_params:,})")

    # --- Individual figures ---
    for label, res in results.items():
        safe_label = label.lower()
        e = res["eval"]
        ht = res["hidden_true"]

        # Fig 1: Hidden time series comparison
        def plot_hidden(ax, res=res, e=e, ht=ht, label=label):
            t_axis = np.arange(len(ht))
            ax.plot(t_axis, ht, color="black", linewidth=1.5, label="真实 hidden")
            ax.plot(t_axis, e["h_scaled"], color="#ff7f0e", linewidth=1.2, alpha=0.85,
                    label=f"Linear-Seeded GNN (P={e['pearson_scaled']:.3f})")
            ax.set_xlabel("时间步"); ax.set_ylabel("丰度")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=11)
        save_fig_single(
            f"{label}: Hidden 恢复  Pearson={e['pearson_scaled']:.3f}  RMSE={e['rmse_scaled']:.3f}",
            plot_hidden, out_dir / f"fig_{safe_label}_hidden_recovery.png",
            figsize=(12, 5),
        )

        # Fig 2: Scatter (true vs recovered)
        def plot_scatter(ax, res=res, e=e, ht=ht, label=label):
            ax.scatter(ht, e["h_scaled"], alpha=0.3, s=10, color="#1565c0")
            vmin = min(ht.min(), e["h_scaled"].min())
            vmax = max(ht.max(), e["h_scaled"].max())
            ax.plot([vmin, vmax], [vmin, vmax], "k--", linewidth=1.0, alpha=0.5, label="y=x")
            ax.set_xlabel("真实 hidden"); ax.set_ylabel("恢复 hidden (scaled)")
            ax.grid(alpha=0.25)
            ax.legend(fontsize=10)
            ax.set_aspect("equal")
        save_fig_single(
            f"{label}: 真实 vs 恢复散点",
            plot_scatter, out_dir / f"fig_{safe_label}_scatter.png",
            figsize=(7, 7),
        )

        # Fig 3: Training losses
        def plot_loss(ax, res=res, label=label):
            h = res["hist"]
            ax.semilogy(h["train_fit"], color="#1565c0", linewidth=1.2, label="train visible loss")
            ax.semilogy(h["val_fit"], color="#c62828", linewidth=1.2, label="val visible loss (早停依据)")
            ax.axvline(res["best_epoch"] - 1, color="green", linestyle="--", linewidth=1.0, alpha=0.7,
                       label=f"best @ ep {res['best_epoch']}")
            ax.set_xlabel("epoch"); ax.set_ylabel("MSE (log-ratio)")
            ax.legend(fontsize=10); ax.grid(alpha=0.25)
        save_fig_single(
            f"{label}: 训练损失曲线",
            plot_loss, out_dir / f"fig_{safe_label}_loss.png",
            figsize=(11, 5.5),
        )

        # Fig 4: α gate + hidden variance over training
        def plot_alpha(ax, res=res, label=label):
            h = res["hist"]
            ax.plot(h["alpha"], color="#1565c0", linewidth=1.2, label="α (GNN 贡献 gate)")
            ax2 = ax.twinx()
            ax2.plot(h["h_var"], color="#c62828", linewidth=1.2, label="hidden variance")
            ax.set_xlabel("epoch"); ax.set_ylabel("α", color="#1565c0")
            ax2.set_ylabel("h_var", color="#c62828")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left", fontsize=10)
            ax2.legend(loc="upper right", fontsize=10)
        save_fig_single(
            f"{label}: α Gate 和 Hidden 方差",
            plot_alpha, out_dir / f"fig_{safe_label}_alpha.png",
            figsize=(11, 5.5),
        )

        # Fig 5: Eval curves (Pearson / RMSE over epochs, monitor only)
        def plot_eval(ax, res=res, label=label):
            h = res["hist"]
            ax.plot(h["eval_epoch"], h["eval_pearson"], color="#1565c0", marker="o", markersize=3, linewidth=1.2, label="|Pearson|")
            ax.axhline(0.9, color="green", linestyle="--", linewidth=0.8, alpha=0.5)
            ax2 = ax.twinx()
            ax2.plot(h["eval_epoch"], h["eval_rmse"], color="#c62828", marker="s", markersize=3, linewidth=1.2, label="RMSE")
            ax2.axhline(0.1, color="orange", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.axvline(res["best_epoch"], color="green", linestyle="--", linewidth=0.8, alpha=0.5)
            ax.set_xlabel("epoch"); ax.set_ylabel("|Pearson|", color="#1565c0")
            ax2.set_ylabel("RMSE", color="#c62828")
            ax.grid(alpha=0.25)
            ax.legend(loc="lower left", fontsize=10)
            ax2.legend(loc="lower right", fontsize=10)
        save_fig_single(
            f"{label}: Hidden 恢复质量演进（仅 monitor，不影响训练）",
            plot_eval, out_dir / f"fig_{safe_label}_eval.png",
            figsize=(11, 5.5),
        )

    # Summary
    print()
    print("=" * 70)
    print("LINEAR-SEEDED GNN RESULTS:")
    for label, res in results.items():
        e = res["eval"]
        print(f"  {label:8s}: Pearson={e['pearson_scaled']:.4f}  RMSE={e['rmse_scaled']:.4f}  (ep {res['best_epoch']})")
    print("=" * 70)

    # Save results npz
    np.savez(out_dir / "results.npz",
              lv_pearson=results["LV"]["eval"]["pearson_scaled"],
              lv_rmse=results["LV"]["eval"]["rmse_scaled"],
              holling_pearson=results["Holling"]["eval"]["pearson_scaled"],
              holling_rmse=results["Holling"]["eval"]["rmse_scaled"],
              lv_h_pred=results["LV"]["eval"]["h_scaled"],
              holling_h_pred=results["Holling"]["eval"]["h_scaled"])

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Linear-Seeded GNN 结果\n\n")
        f.write("Linear sparse 作为 GNN 的 input feature / anchor, GNN 学 residual correction (有 α gate).\n\n")
        for label, res in results.items():
            e = res["eval"]
            f.write(f"## {label}\n")
            f.write(f"- Params: {res['num_params']:,}\n")
            f.write(f"- Pearson (scaled): {e['pearson_scaled']:.4f}\n")
            f.write(f"- RMSE (scaled): {e['rmse_scaled']:.4f}\n")
            f.write(f"- Best epoch: {res['best_epoch']}\n\n")
    print(f"[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

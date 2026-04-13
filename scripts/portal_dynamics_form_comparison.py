"""Portal 真实数据：不同动力学形式对 hidden recovery 的影响.

研究问题:
  log(x_{t+1}/x_t) 的 Ricker 形式是必要的吗?
  更自由的形式是否能在真实 rodent 数据上给出更好的 hidden 恢复?

6 种形式对比:
  A: log-ratio 线性            log(x_{t+1}/x_t) = r + A·x + B·H
  B: log-ratio 二次            log(x_{t+1}/x_t) = r + A·x + Q·x² + B·H
  C: Gompertz (log-state 线性) log(x_{t+1}) = r + A·log(x) + B·H
  D: 加性增量                  x_{t+1} - x_t = r + A·x + B·H
  E: 线性+神经残差             log(x_{t+1}/x_t) = r + A·x + MLP(x) + B·H
  F: 完全神经 (+log)           log(x_{t+1}) = MLP(log(x)) + B·H

评估 (target = OT):
  - Visible reconstruction RMSE (scale normalized)
  - Hidden(OT) recovery Pearson (主要指标)
  - 3 seeds 稳定性
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

TOP12 = ["PP", "DM", "PB", "DO", "OT", "RM", "PE", "DS", "PF", "NA", "OL", "PM"]


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


def aggregate_portal(csv_path, species_list):
    counts = defaultdict(lambda: defaultdict(int))
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                year = int(row['year']); month = int(row['month'])
            except (ValueError, KeyError):
                continue
            sp = row['species']
            if sp in species_list:
                counts[(year, month)][sp] += 1
    all_months = sorted(counts.keys())
    matrix = np.zeros((len(all_months), len(species_list)), dtype=np.float32)
    for t, (y, m) in enumerate(all_months):
        for j, sp in enumerate(species_list):
            matrix[t, j] = counts[(y, m)].get(sp, 0)
    return matrix, all_months


class DynamicsModel(nn.Module):
    """Unified interface for 6 dynamics forms.

    All forms have:
      - B·H hidden contribution (same for all forms)
      - Target is always log-ratio of visible (but predicted differently)
    """
    def __init__(self, form, N, k_latent, T_m1, hidden_dim=32, seed=42):
        super().__init__()
        torch.manual_seed(seed)
        self.form = form
        self.N = N
        self.k = k_latent
        self.r = nn.Parameter(torch.zeros(N))
        self.A = nn.Parameter(torch.zeros(N, N))
        with torch.no_grad():
            self.A.data.fill_diagonal_(-0.2)
            self.A.data += 0.01 * torch.randn(N, N)
        self.B = nn.Parameter(0.1 * torch.randn(N, k_latent))
        self.H = nn.Parameter(0.1 * torch.randn(T_m1, k_latent))

        if form == "B":
            self.Q = nn.Parameter(torch.zeros(N, N))
        elif form == "C":
            # log(x_{t+1}) = r + A·log(x) + B·H
            pass
        elif form == "D":
            # x_{t+1} - x_t = r + A·x + B·H
            pass
        elif form == "E":
            self.mlp = nn.Sequential(
                nn.Linear(N, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, N),
            )
        elif form == "F":
            # log(x_{t+1}) = MLP(log(x), H) directly
            self.mlp = nn.Sequential(
                nn.Linear(N + k_latent, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, hidden_dim), nn.Tanh(),
                nn.Linear(hidden_dim, N),
            )

    def predict_target(self, x_states):
        """Return (pred_target, target_name). x_states: (T, N) raw abundance."""
        T = x_states.shape[0]
        T_m1 = T - 1
        x_t = x_states[:-1]          # (T-1, N) raw
        x_next = x_states[1:]        # (T-1, N) raw
        log_x_t = torch.log(torch.clamp(x_t, min=1e-3))
        log_x_next = torch.log(torch.clamp(x_next, min=1e-3))
        log_ratio = log_x_next - log_x_t

        H = self.H  # (T-1, k)
        BH = H @ self.B.T  # (T-1, N)

        if self.form == "A":
            pred = self.r.view(1, -1) + x_t @ self.A.T + BH
            target = log_ratio
            return pred, target, "log_ratio"
        elif self.form == "B":
            pred = self.r.view(1, -1) + x_t @ self.A.T + (x_t ** 2) @ self.Q.T + BH
            target = log_ratio
            return pred, target, "log_ratio"
        elif self.form == "C":
            # log(x_{t+1}) = r + A·log(x) + B·H
            pred = self.r.view(1, -1) + log_x_t @ self.A.T + BH
            target = log_x_next
            return pred, target, "log_state"
        elif self.form == "D":
            # x_{t+1} - x_t = r + A·x + B·H
            pred = self.r.view(1, -1) + x_t @ self.A.T + BH
            target = x_next - x_t
            return pred, target, "delta"
        elif self.form == "E":
            pred = self.r.view(1, -1) + x_t @ self.A.T + self.mlp(x_t) + BH
            target = log_ratio
            return pred, target, "log_ratio"
        elif self.form == "F":
            inp = torch.cat([log_x_t, H], dim=-1)
            pred = self.mlp(inp)
            target = log_x_next
            return pred, target, "log_state"
        raise ValueError(f"Unknown form: {self.form}")


def fit_form(form, visible, n_iter=3000, lr=0.01, lam_A=0.3, lam_H=0.02, lam_B=0.02,
              k_latent=3, seed=42, device="cpu"):
    """Fit one dynamics form on visible data."""
    T, N = visible.shape
    T_m1 = T - 1
    x_all = torch.tensor(visible, dtype=torch.float32, device=device)
    model = DynamicsModel(form, N, k_latent, T_m1, seed=seed).to(device)

    # Param groups: weight decay only on A, B, mlp
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for _ in range(n_iter):
        opt.zero_grad()
        pred, target, t_name = model.predict_target(x_all)
        # For numerical stability clip targets
        target = torch.clamp(target, -3.0, 3.0) if t_name != "delta" else target
        recon = ((pred - target) ** 2).mean()
        A_off = model.A - torch.diag(torch.diag(model.A))
        loss = recon + lam_A * A_off.abs().mean() \
                    + lam_B * model.B.abs().mean() \
                    + lam_H * model.H.abs().mean()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
    with torch.no_grad():
        pred, target, t_name = model.predict_target(x_all)
        if t_name != "delta":
            target = torch.clamp(target, -3.0, 3.0)
        final_recon = float(((pred - target) ** 2).mean())
        H_fit = model.H.cpu().numpy()
    return H_fit, final_recon


def rotate_by_variance(H):
    U, S, Vt = np.linalg.svd(H - H.mean(axis=0, keepdims=True), full_matrices=False)
    return U * S[None, :], S


def combined_pearson(H, target):
    L = min(H.shape[0], len(target))
    X = np.concatenate([H[:L], np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, target[:L], rcond=None)
    h_combined = X @ coef
    return float(np.corrcoef(h_combined, target[:L])[0, 1]), h_combined


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_dynamics_form_comparison")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # Load data
    matrix, months = aggregate_portal("data/real_datasets/portal_rodent.csv", TOP12)
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w//2, w//2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w//2:w//2+x.shape[0]]
    matrix_s = smooth(matrix, w=3)
    valid = matrix_s.sum(axis=1) > 10
    matrix_s = matrix_s[valid]
    months_valid = [m for m, v in zip(months, valid) if v]
    time_axis = np.array([y + m/12 for (y, m) in months_valid])
    print(f"T={len(months_valid)} months\n")

    # Try each of the top-ranked candidates as hidden species
    # Focus on OT (our best target)
    forms = ["A", "B", "C", "D", "E", "F"]
    form_labels = {
        "A": "log-ratio 线性",
        "B": "log-ratio 二次",
        "C": "Gompertz (log-state 线性)",
        "D": "加性增量",
        "E": "线性+神经残差",
        "F": "完全神经网络",
    }
    target_species = ["OT", "DO", "PP"]  # top 3 candidates by linear baseline
    seeds = [42, 123, 456]

    all_results = {}
    for sp in target_species:
        print(f"\n{'='*75}\nTarget hidden = {sp}\n{'='*75}")
        h_idx = TOP12.index(sp)
        keep = [i for i in range(len(TOP12)) if i != h_idx]
        visible = matrix_s[:, keep] + 0.5
        hidden = matrix_s[:, h_idx] + 0.5

        form_results = {}
        for form in forms:
            pearsons = []
            recons = []
            best_H = None
            best_p = -np.inf
            for seed in seeds:
                H, recon = fit_form(form, visible, seed=seed, device=device)
                H_rot, _ = rotate_by_variance(H)
                p, _ = combined_pearson(H_rot, hidden[:-1])
                pearsons.append(p)
                recons.append(recon)
                if abs(p) > abs(best_p):
                    best_p = p
                    best_H = H_rot
            pears_arr = np.array(pearsons)
            form_results[form] = {
                "pearsons": pears_arr,
                "mean_p": float(pears_arr.mean()),
                "std_p": float(pears_arr.std()),
                "max_abs_p": float(np.max(np.abs(pears_arr))),
                "recon": float(np.mean(recons)),
                "best_H": best_H,
            }
            print(f"  Form {form} ({form_labels[form]}):")
            print(f"    Pearsons: {[f'{p:+.3f}' for p in pearsons]}")
            print(f"    Mean = {pears_arr.mean():+.4f} ± {pears_arr.std():.4f}, "
                  f"max|P| = {np.max(np.abs(pears_arr)):.4f}, recon = {np.mean(recons):.4f}")
        all_results[sp] = {"form_results": form_results, "hidden": hidden}

    # ===== Summary table =====
    print(f"\n{'='*75}\nSUMMARY: max|Pearson| across seeds for each form × species\n{'='*75}")
    print(f"{'Form':<6}{'Description':<28}", end="")
    for sp in target_species:
        print(f"{sp:<12}", end="")
    print()
    for form in forms:
        print(f"{form:<6}{form_labels[form][:27]:<28}", end="")
        for sp in target_species:
            r = all_results[sp]["form_results"][form]
            print(f"{r['max_abs_p']:<+12.3f}", end="")
        print()

    # ========== Plots ==========
    def save_single(title, plot_fn, path, figsize=(12, 5)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    # Fig 1: Heatmap max|Pearson|
    def plot_heatmap(ax):
        M = np.array([[all_results[sp]["form_results"][f]["max_abs_p"] for sp in target_species]
                      for f in forms])
        im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=0.7, aspect="auto")
        ax.set_xticks(range(len(target_species)))
        ax.set_xticklabels(target_species)
        ax.set_yticks(range(len(forms)))
        ax.set_yticklabels([f"{f}: {form_labels[f]}" for f in forms])
        for i, f in enumerate(forms):
            for j, sp in enumerate(target_species):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center",
                        color="white" if M[i, j] < 0.3 else "black", fontsize=11)
        plt.colorbar(im, ax=ax, label="max|Pearson|")
    save_single("6 种动力学形式 × 3 个目标物种: max|Pearson| (3 seeds)",
                 plot_heatmap, out_dir / "fig_01_heatmap_max_pearson.png",
                 figsize=(10, 6))

    # Fig 2: Bar: mean ± std per form, for OT
    def plot_bars(ax):
        x = np.arange(len(forms))
        means = [all_results["OT"]["form_results"][f]["mean_p"] for f in forms]
        stds = [all_results["OT"]["form_results"][f]["std_p"] for f in forms]
        maxes = [all_results["OT"]["form_results"][f]["max_abs_p"] for f in forms]
        colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(forms)))
        ax.bar(x - 0.2, means, 0.4, yerr=stds, capsize=4, color="#1565c0",
               label="mean ± std")
        ax.bar(x + 0.2, maxes, 0.4, color="#ff7f0e", label="max |P| across seeds")
        ax.set_xticks(x); ax.set_xticklabels(forms)
        ax.set_ylabel("Pearson to OT"); ax.axhline(0, color="grey", linewidth=0.5)
        ax.axhline(0.3534, color="red", linestyle="--", alpha=0.5,
                   label="Linear baseline (rank=1, k=1) = 0.35")
        ax.legend(fontsize=10); ax.grid(alpha=0.25, axis="y")
    save_single("OT: 6 种动力学形式对比 (3 seeds)",
                 plot_bars, out_dir / "fig_02_OT_form_bars.png", figsize=(12, 5))

    # Fig 3: Best form's hidden recovery for each species
    for sp in target_species:
        fr = all_results[sp]["form_results"]
        best_form = max(forms, key=lambda f: fr[f]["max_abs_p"])
        best_H = fr[best_form]["best_H"]
        hidden = all_results[sp]["hidden"]
        p, h_comb = combined_pearson(best_H, hidden[:-1])

        def make_plot(sp=sp, best_form=best_form, h_comb=h_comb, p=p, hidden=hidden):
            def _p(ax):
                ht = hidden[:len(h_comb)]
                t_axis = time_axis[:len(h_comb)]
                ax.plot(t_axis, ht, color="black", linewidth=1.8, label=f"真实 {sp}")
                ax.plot(t_axis, h_comb, color="#1565c0", linewidth=1.3, alpha=0.85,
                        label=f"Form {best_form} 恢复 (P={p:.3f})")
                ax.set_xlabel("Year"); ax.set_ylabel(f"{sp} abundance")
                ax.legend(fontsize=11); ax.grid(alpha=0.25)
            return _p
        save_single(
            f"Best form for {sp}: Form {best_form} ({form_labels[best_form]})",
            make_plot(), out_dir / f"fig_best_form_{sp}.png",
        )

    # Summary md
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Portal 动力学形式对比 (6 forms x 3 species x 3 seeds)\n\n")
        f.write("| Form | Description | ")
        for sp in target_species:
            f.write(f"{sp} max|P| | ")
        f.write("\n| --- | --- |")
        for _ in target_species:
            f.write(" --- |")
        f.write("\n")
        for form in forms:
            f.write(f"| {form} | {form_labels[form]} | ")
            for sp in target_species:
                r = all_results[sp]["form_results"][form]
                f.write(f"{r['max_abs_p']:+.4f} | ")
            f.write("\n")
        f.write(f"\n## 各 target 最佳 form\n\n")
        for sp in target_species:
            fr = all_results[sp]["form_results"]
            best = max(forms, key=lambda f: fr[f]["max_abs_p"])
            f.write(f"- **{sp}**: Form {best} ({form_labels[best]}), "
                    f"max|P|={fr[best]['max_abs_p']:.4f}\n")

    np.savez(out_dir / "results.npz",
              forms=np.array(forms),
              target_species=np.array(target_species),
              max_p_matrix=np.array([[all_results[sp]["form_results"][f]["max_abs_p"]
                                      for sp in target_species] for f in forms]),
              mean_p_matrix=np.array([[all_results[sp]["form_results"][f]["mean_p"]
                                       for sp in target_species] for f in forms]))
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

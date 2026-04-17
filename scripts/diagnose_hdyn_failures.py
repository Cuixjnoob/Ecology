"""Diagnose why sp3 and sp6 fail with h_dyn while others improve.

Hypothesis candidates:
  A. Autocorrelation: failing species have low self-correlation, so
     h_{t+1} is not predictable from (h_t, x_t)
  B. Coupling strength: failing species are more loosely coupled to visibles
  C. Dynamics complexity: failing species have longer-memory dynamics
     (need more than 1-step lookback)
  D. Spectral spread: failing species have broader frequency content
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def autocorr_decay(x, max_lag=30):
    """Return lag at which autocorrelation drops below 0.3"""
    x = (x - x.mean()) / (x.std() + 1e-8)
    T = len(x)
    for lag in range(1, max_lag):
        if T - lag < 10:
            return lag
        c = np.corrcoef(x[:-lag], x[lag:])[0, 1]
        if c < 0.3:
            return lag
    return max_lag


def mi_proxy(x, y, bins=20):
    """Simple binned MI estimate between two time series."""
    h, _, _ = np.histogram2d(x, y, bins=bins)
    pxy = h / h.sum()
    px = pxy.sum(axis=1); py = pxy.sum(axis=0)
    # mask zeros
    nz = pxy > 0
    log_term = np.zeros_like(pxy)
    log_term[nz] = np.log(pxy[nz] / (np.outer(px, py)[nz] + 1e-12))
    return float((pxy * log_term).sum())


def next_step_predictability(x, y_vis, lag=1):
    """How well is x[t+lag] predicted by (x[t], y_vis[t])?
    Uses simple linear regression R2.
    """
    T = len(x)
    if T <= lag + 5:
        return 0.0
    # Features: x[t], y_vis[t, :]
    features = np.concatenate([x[:-lag, None], y_vis[:-lag]], axis=1)
    target = x[lag:]
    # Simple linear regression
    X = features - features.mean(axis=0)
    y = target - target.mean()
    # Least squares
    try:
        w, _, _, _ = np.linalg.lstsq(X, y, rcond=None)
        y_pred = X @ w
        ss_res = ((y - y_pred) ** 2).sum()
        ss_tot = (y ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-12)
        return float(max(0, r2))
    except Exception:
        return 0.0


def main():
    out_dir = Path("runs/hdyn_failure_diagnosis")
    out_dir.mkdir(parents=True, exist_ok=True)

    data = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = data["N_all"]  # (T, 6)
    R = data["resources"]   # (T, 5)
    full = np.concatenate([N_all, R], axis=1)   # (T, 11)

    # Normalize
    full_norm = full / (full.mean(axis=0, keepdims=True) + 1e-8)

    # h_dyn experimental results
    hdyn_delta = {
        "sp1": +0.12, "sp2": +0.13, "sp3": -0.14,
        "sp4": +0.25, "sp5": +0.15, "sp6": -0.12,
    }
    species_list = ["sp1", "sp2", "sp3", "sp4", "sp5", "sp6"]

    print("="*80)
    print("DIAGNOSIS: why sp3 & sp6 fail with h_dyn")
    print("="*80)

    # Metric 1: autocorrelation decay
    print("\nA. Autocorrelation decay (lag for ACF < 0.3):")
    print(f"{'sp':<6}{'decay_lag':<12}{'std':<10}{'CV':<8}{'D h_dyn':<12}")
    acf_data = []
    for i in range(6):
        x = N_all[:, i]
        decay = autocorr_decay(x)
        std = x.std()
        cv = x.std() / (x.mean() + 1e-8)
        d = hdyn_delta[f"sp{i+1}"]
        status = "[OK]" if d > 0 else "[FAIL]"
        acf_data.append(decay)
        print(f"sp{i+1}   {decay:<12}{std:<10.3f}{cv:<8.3f}{d:<+12.3f} {status}")

    # Metric 2: predictability R2 of h_{t+1} from (h_t, visible_t)
    print("\nB. Next-step R2 from (h_t, visibles_t) [higher = easier for g MLP]:")
    print(f"{'sp':<6}{'R2':<10}{'D h_dyn':<12}")
    r2_data = []
    for i in range(6):
        hidden = N_all[:, i]
        visible = np.delete(full_norm, i, axis=1)
        r2 = next_step_predictability(hidden, visible)
        d = hdyn_delta[f"sp{i+1}"]
        status = "[OK]" if d > 0 else "[FAIL]"
        r2_data.append(r2)
        print(f"sp{i+1}   {r2:<10.3f}{d:<+12.3f} {status}")

    # Metric 3: coupling to other species
    print("\nC. Mean |correlation| with other species + resources:")
    print(f"{'sp':<6}{'mean|corr|':<14}{'D h_dyn':<12}")
    coupling_data = []
    for i in range(6):
        x = N_all[:, i]
        other = np.delete(full_norm, i, axis=1)
        corrs = [np.corrcoef(x, other[:, j])[0, 1] for j in range(other.shape[1])]
        m = np.mean(np.abs(corrs))
        d = hdyn_delta[f"sp{i+1}"]
        status = "[OK]" if d > 0 else "[FAIL]"
        coupling_data.append(m)
        print(f"sp{i+1}   {m:<14.3f}{d:<+12.3f} {status}")

    # Metric 4: spectral entropy (dominant period)
    print("\nD. Dominant oscillation period (time steps):")
    print(f"{'sp':<6}{'dom_period':<14}{'D h_dyn':<12}")
    dom_periods = []
    for i in range(6):
        x = N_all[:, i] - N_all[:, i].mean()
        # FFT
        freqs = np.fft.rfftfreq(len(x), d=2.0)   # dt=2
        ps = np.abs(np.fft.rfft(x)) ** 2
        ps[0] = 0   # exclude DC
        dom_f = freqs[np.argmax(ps)]
        dom_T = 1.0 / (dom_f + 1e-8) if dom_f > 0 else float("nan")
        d = hdyn_delta[f"sp{i+1}"]
        status = "[OK]" if d > 0 else "[FAIL]"
        dom_periods.append(dom_T)
        print(f"sp{i+1}   {dom_T:<14.1f}{d:<+12.3f} {status}")

    # Metric 5: Hurst-like roughness (is trajectory smooth or noisy)
    print("\nE. Trajectory roughness (std of 2nd-order differences):")
    print(f"{'sp':<6}{'roughness':<14}{'D h_dyn':<12}")
    rough_data = []
    for i in range(6):
        x = N_all[:, i]
        d2x = np.diff(np.diff(x))
        rough = d2x.std() / (x.std() + 1e-8)   # normalized
        d = hdyn_delta[f"sp{i+1}"]
        status = "[OK]" if d > 0 else "[FAIL]"
        rough_data.append(rough)
        print(f"sp{i+1}   {rough:<14.4f}{d:<+12.3f} {status}")

    # Summary correlation: do any metrics correlate with h_dyn success?
    print("\n" + "="*80)
    print("CORRELATION of each metric with h_dyn improvement (D):")
    print("="*80)
    deltas = [hdyn_delta[sp] for sp in species_list]
    metrics = {
        "A. ACF decay lag": acf_data,
        "B. Next-step R2": r2_data,
        "C. Mean |corr| w/ vis": coupling_data,
        "D. Dominant period": dom_periods,
        "E. Roughness": rough_data,
    }
    for name, vals in metrics.items():
        c = np.corrcoef(vals, deltas)[0, 1]
        interp = ""
        if abs(c) > 0.6:
            interp = " <--- strong predictor"
        elif abs(c) > 0.3:
            interp = " <-- moderate"
        print(f"  {name:<28s}  corr with D = {c:+.3f}{interp}")

    # Plot: time series of each species colored by success/fail
    fig, axes = plt.subplots(3, 2, figsize=(13, 9), constrained_layout=True)
    t_axis = np.arange(N_all.shape[0]) * 2.0  # dt=2
    for i, ax in enumerate(axes.flat):
        x = N_all[:, i]
        d = hdyn_delta[f"sp{i+1}"]
        color = "#2e7d32" if d > 0 else "#c62828"
        ax.plot(t_axis, x, color=color, lw=0.9)
        ax.set_title(f"sp{i+1}  (D h_dyn = {d:+.2f})   "
                     f"{'[OK] improves' if d > 0 else '[FAIL] hurts'}",
                     fontsize=11)
        ax.set_xlabel("day"); ax.set_ylabel("abundance")
        ax.grid(alpha=0.25)
    fig.suptitle("Huisman species dynamics — which ones benefit from h_dyn?",
                 fontweight="bold")
    fig.savefig(out_dir / "fig_trajectories_by_outcome.png", dpi=130,
                bbox_inches="tight")
    plt.close(fig)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

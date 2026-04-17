"""Generate Huisman & Weissing 1999 chaotic chemostat data (5+1 setup).

Reference: Huisman & Weissing 1999 Nature 402:407-410
  "Biodiversity of plankton by species oscillations and chaos"

Model (Eqs 1-3):
  dN_i/dt = N_i * (mu_i - m_i)
  dR_j/dt = D*(S_j - R_j) - sum_i c_ji * mu_i * N_i
  mu_i = min_j { r_i * R_j / (K_ji + R_j) }   [Liebig min]

Parameters: Fig 4 matrices (first 6 columns), which produce chaos
at 5 species competing for 5 resources (Fig 2).

Output: trajectories.npz with
  - states_B_5species: (T, 5)  — visible species 1-5 (not incl. hidden)
  - hidden_B: (T,)             — species 6 (hidden)
  - resources: (T, 5)          — R1..R5
  - N_all: (T, 6)              — all 6 species for reference

This matches the format of existing LV/Holling data in the project.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
from scipy.integrate import odeint

# Parameters from Huisman 1999 Methods (Fig 4 uses these)
R_MAX = 1.0      # r_i = 1 /day for all species
MORT = 0.25      # m_i = 0.25 /day for all
DIL = 0.25       # D = 0.25 /day

# Fig 4: S = [6, 10, 14, 4, 9]
S_VEC = np.array([6.0, 10.0, 14.0, 4.0, 9.0])

# Fig 4 K matrix (5 resources × 12 species). First 6 cols for our N=6 setup.
# Transcribed from Nature 402:410 Methods section.
K_FIG4 = np.array([
    [0.39, 0.34, 0.30, 0.24, 0.23, 0.41, 0.20, 0.45, 0.14, 0.15, 0.38, 0.28],
    [0.22, 0.39, 0.34, 0.30, 0.27, 0.16, 0.15, 0.05, 0.38, 0.29, 0.37, 0.31],
    [0.27, 0.22, 0.39, 0.34, 0.30, 0.07, 0.11, 0.05, 0.38, 0.41, 0.24, 0.25],
    [0.30, 0.24, 0.22, 0.39, 0.34, 0.28, 0.12, 0.13, 0.27, 0.33, 0.04, 0.41],
    [0.34, 0.30, 0.22, 0.20, 0.39, 0.40, 0.50, 0.26, 0.12, 0.29, 0.09, 0.16],
])

# Fig 4 C matrix (5 resources × 12 species). First 6 cols.
C_FIG4 = np.array([
    [0.04, 0.04, 0.07, 0.04, 0.04, 0.22, 0.10, 0.08, 0.02, 0.17, 0.25, 0.03],
    [0.08, 0.08, 0.08, 0.10, 0.08, 0.14, 0.22, 0.04, 0.18, 0.06, 0.20, 0.04],
    [0.10, 0.10, 0.10, 0.10, 0.14, 0.22, 0.24, 0.12, 0.03, 0.24, 0.17, 0.01],
    [0.05, 0.03, 0.03, 0.03, 0.03, 0.09, 0.07, 0.06, 0.03, 0.03, 0.11, 0.05],
    [0.07, 0.09, 0.07, 0.07, 0.07, 0.05, 0.24, 0.05, 0.08, 0.10, 0.02, 0.04],
])

N_SPECIES = 6   # 5 visible + 1 hidden (first 6 cols)
N_RESOURCES = 5
K = K_FIG4[:, :N_SPECIES].copy()     # (5, 6)
C = C_FIG4[:, :N_SPECIES]             # (5, 6)
# Shift K41 from default 0.30 to 0.26 to strengthen chaos (λ=0.022 → 0.043, ~3/4 Beninca)
K[3, 0] = 0.26


def derivs(y, t):
    """y = [N_1..N_6, R_1..R_5]."""
    N = y[:N_SPECIES]
    R = y[N_SPECIES:]
    R_clipped = np.clip(R, 1e-8, None)
    # mu_ji = r * R_j / (K_ji + R_j), shape (n_res, n_sp)
    mu_per_res = R_MAX * R_clipped[:, None] / (K + R_clipped[:, None])
    # Liebig min across resources
    mu = np.min(mu_per_res, axis=0)           # (n_sp,)
    mu = np.clip(mu, 0, None)                 # no negative growth

    dN = N * (mu - MORT)
    # resource consumption: sum_i c_ji * mu_i * N_i
    consumption = (C * (mu * N)[None, :]).sum(axis=1)   # (n_res,)
    dR = DIL * (S_VEC - R) - consumption
    return np.concatenate([dN, dR])


def generate(t_transient: float = 1000.0, t_record: float = 2000.0,
             dt: float = 1.0, seed_pert: float = 0.0):
    """Integrate Huisman 1999 model, discard transient, return recorded segment.

    t_transient: discard first t_transient days
    t_record: record next t_record days
    dt: sampling interval

    Returns:
      N_record (T, 6), R_record (T, 5), t_axis (T,)
    """
    # Initial conditions per paper:
    # R_j = S_j, N_i = 0.1 + i/100 for i=1..6 (not 0-indexed)
    N0 = np.array([0.1 + (i + 1) / 100.0 for i in range(N_SPECIES)])
    N0 = N0 + seed_pert * np.random.randn(N_SPECIES) * 0.01
    R0 = S_VEC.copy().astype(float)
    y0 = np.concatenate([N0, R0])

    # Integrate transient + record, fine time steps for ODE accuracy
    t_full = np.arange(0, t_transient + t_record + dt, dt * 0.2)  # 5x finer integration
    sol = odeint(derivs, y0, t_full, rtol=1e-8, atol=1e-10, mxstep=5000)

    # Extract recorded period at dt sampling
    n_transient_fine = int(t_transient / (dt * 0.2))
    stride = 5
    sol_record = sol[n_transient_fine::stride]
    T_record = sol_record.shape[0]
    t_axis = np.arange(T_record) * dt

    N_record = sol_record[:, :N_SPECIES]       # (T, 6)
    R_record = sol_record[:, N_SPECIES:]       # (T, 5)

    return N_record, R_record, t_axis


def main():
    out_dir = Path("runs") / "huisman1999_chaos"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Generating Huisman 1999 chaos (6 species + 5 resources)...")
    N, R, t = generate(t_transient=1000.0, t_record=2000.0, dt=2.0)
    print(f"  T = {len(t)} days, dt = 2 day, after 1000-day transient")
    print(f"  N species ranges (mean | std | max):")
    for i in range(N_SPECIES):
        print(f"    sp{i+1}:  {N[:, i].mean():.3f} | {N[:, i].std():.3f} | {N[:, i].max():.3f}")
    print(f"  R resource ranges (mean | std):")
    for j in range(N_RESOURCES):
        print(f"    R{j+1}:  {R[:, j].mean():.3f} | {R[:, j].std():.3f}")

    # Setup: species 6 = hidden, species 1-5 = visible + 5 resources = 10 visible
    visible_species = N[:, :5]      # (T, 5)
    hidden_species = N[:, 5]        # (T,)
    resources = R                    # (T, 5)

    # Simple chaos indicator: compute autocorrelation decay time
    for i in range(N_SPECIES):
        x = N[:, i] - N[:, i].mean()
        lags_test = min(50, len(x) // 4)
        acf = np.array([np.corrcoef(x[:-l], x[l:])[0, 1]
                         for l in range(1, lags_test + 1)])
        # Find first crossing of 0.1
        below = np.where(acf < 0.1)[0]
        decay = below[0] + 1 if len(below) > 0 else lags_test
        print(f"  sp{i+1} ACF decay to 0.1 at lag {decay}  (chaos sign: short decay)")

    out_npz = out_dir / "trajectories.npz"
    np.savez(
        out_npz,
        states_B_5species=visible_species.astype(np.float32),
        hidden_B=hidden_species.astype(np.float32),
        resources=resources.astype(np.float32),
        N_all=N.astype(np.float32),
        t_axis=t.astype(np.float32),
    )
    print(f"\n[OK] Saved: {out_npz}")
    print(f"     states_B_5species shape: {visible_species.shape}")
    print(f"     hidden_B shape: {hidden_species.shape}")

    # Plot for sanity
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 1, figsize=(12, 7), constrained_layout=True)
        ax = axes[0]
        for i in range(N_SPECIES):
            ax.plot(t, N[:, i], label=f"sp{i+1}", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("day"); ax.set_ylabel("species abundance")
        ax.set_title("Huisman 1999 chaos: 6 species competing for 5 resources")
        ax.legend(fontsize=9, ncol=6); ax.grid(alpha=0.25)

        ax = axes[1]
        for j in range(N_RESOURCES):
            ax.plot(t, R[:, j], label=f"R{j+1}", linewidth=1.0, alpha=0.8)
        ax.set_xlabel("day"); ax.set_ylabel("resource concentration")
        ax.legend(fontsize=9, ncol=5); ax.grid(alpha=0.25)

        fig.savefig(out_dir / "fig_trajectories.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"[OK] fig: {out_dir / 'fig_trajectories.png'}")
    except Exception as e:
        print(f"[warn] plot failed: {e}")


if __name__ == "__main__":
    main()

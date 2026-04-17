"""Scan K41 to find stronger-chaos Huisman. Target λ ≈ 0.045 /day.

Huisman 1999 Fig 3: K41 ∈ [0.24, 0.35] is chaos. Default 0.30 gives λ=0.022.
Move toward bifurcation edge to amplify chaos.
"""
from __future__ import annotations
import numpy as np
from scipy.integrate import odeint
from scripts.generate_huisman1999 import (
    K_FIG4, C_FIG4, S_VEC, N_SPECIES, R_MAX, MORT, DIL
)
from scripts.compute_lyapunov import lyapunov_benettin_ode


def make_derivs(K):
    """Factory: returns ODE derivs function with given K matrix."""
    C = C_FIG4[:, :N_SPECIES]

    def derivs(y, t):
        N = y[:N_SPECIES]
        R = y[N_SPECIES:]
        Rc = np.clip(R, 1e-8, None)
        mu_per_res = R_MAX * Rc[:, None] / (K + Rc[:, None])
        mu = np.clip(np.min(mu_per_res, axis=0), 0, None)
        dN = N * (mu - MORT)
        consumption = (C * (mu * N)[None, :]).sum(axis=1)
        dR = DIL * (S_VEC - R) - consumption
        return np.concatenate([dN, dR])
    return derivs


def get_post_transient_y0(derivs_fn, t_transient=1000.0):
    N0 = np.array([0.1 + (i + 1) / 100.0 for i in range(N_SPECIES)])
    y0 = np.concatenate([N0, S_VEC.astype(float)])
    t = np.arange(0, t_transient, 0.5)
    sol = odeint(derivs_fn, y0, t, rtol=1e-8, atol=1e-10, mxstep=5000)
    return sol[-1]


def main():
    K_base = K_FIG4[:, :N_SPECIES].copy()   # (5, 6)
    # K41 = K[3, 0] (0-indexed: row 3 = resource 4, col 0 = species 1)
    print(f"Default K41 = {K_base[3, 0]:.3f}")
    print(f"Huisman Fig 3 chaos range: K41 ∈ [0.24, 0.35]\n")

    scan_values = [0.25, 0.26, 0.27, 0.28, 0.30, 0.32]
    results = []
    for k41 in scan_values:
        np.random.seed(42)
        K = K_base.copy()
        K[3, 0] = k41
        derivs = make_derivs(K)
        y0 = get_post_transient_y0(derivs, t_transient=1000.0)
        lam, _, _ = lyapunov_benettin_ode(
            derivs, y0, t_max=150, dt=0.1, eps=1e-8, n_skip=50)
        # Also check stability: verify system didn't go extinct
        t_check = np.arange(0, 200, 1.0)
        sol = odeint(derivs, y0, t_check, rtol=1e-8, atol=1e-10, mxstep=5000)
        species_mins = sol[:, :N_SPECIES].min(axis=0)
        alive = (species_mins > 0.01).sum()
        print(f"K41={k41:.3f}  λ_Benettin={lam:+.4f} /day  alive_species={alive}/6")
        results.append((k41, lam, alive))

    # Pick one closest to target 0.045
    target = 0.045
    best = min([(k, l, a) for k, l, a in results if a == 6],
                key=lambda x: abs(x[1] - target))
    print(f"\nClosest to target λ={target}: K41={best[0]:.3f}  λ={best[1]:+.4f}  alive={best[2]}")
    print(f"(If no K41 gives λ near target, might need other changes)")

    return best


if __name__ == "__main__":
    main()

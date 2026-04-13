"""Holling 数据上的 5 物种 vs 6 物种对比。

和 compare_5vs6_species_dynamics.py 对应，但用 Holling + Allee + 时滞动力学。

目的:
  1. 确认 Holling 数据上 hidden 对 visible 也有大影响
  2. 保存轨迹数据供后续 sparse baseline 实验使用
"""
from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.partial_nonlinear_mvp import (
    generate_partial_nonlinear_mvp_system,
    _holling2,
)


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def simulate_reduced_5species_holling(
    growth_rates_5: torch.Tensor,
    interaction_5x5: torch.Tensor,
    environment_loadings_5: torch.Tensor,
    pulse_loadings_5: torch.Tensor,
    allee_thresholds_5: torch.Tensor,
    delay_coefficients_5: torch.Tensor,
    initial_state_5: torch.Tensor,
    environment_driver: torch.Tensor,
    pulse_driver: torch.Tensor,
    noise_seed: int,
    process_noise: float = 0.006,
    max_state_value: float = 5.5,
    half_saturation: float = 0.45,
    delay_lag: int = 4,
) -> torch.Tensor:
    """只演化 5 物种 (Holling + Allee + 时滞)，和 6 物种系统相同的 env/pulse/noise。"""
    T = int(environment_driver.shape[0])
    states = torch.zeros(T, 5, dtype=torch.float32)
    states[0] = initial_state_5.clone()

    gen = torch.Generator().manual_seed(noise_seed)

    for t in range(T - 1):
        current = states[t]
        drive = torch.zeros(5, dtype=torch.float32)

        # Self limit
        self_limit = -0.12 * current.square() / (0.45 + current)

        # Holling type II interactions (5x5 only)
        for i in range(5):
            for j in range(5):
                if i == j:
                    continue
                a_ij = float(interaction_5x5[i, j].item())
                if abs(a_ij) < 1e-4:
                    continue
                if a_ij > 0:
                    drive[i] += _holling2(current[j], a_ij, half_saturation)
                else:
                    drive[i] += a_ij * current[j] * current[i] / (0.3 + current[i])

        # Allee
        for i in range(5):
            A_i = float(allee_thresholds_5[i].item())
            if A_i > 0.0:
                drive[i] += 0.25 * (current[i] - A_i) / (0.5 + current[i])

        # Time delay
        if t >= delay_lag:
            past_state = states[t - delay_lag]
            drive = drive - delay_coefficients_5 * (current - past_state)

        # Env + pulse
        drive = drive + environment_loadings_5 * environment_driver[t, 0]
        drive = drive + pulse_loadings_5 * pulse_driver[t, 0]
        drive = drive + growth_rates_5 + self_limit
        drive = drive + process_noise * 0.70 * torch.randn(5, generator=gen)

        next_state = current * torch.exp(torch.clamp(drive, min=-1.10, max=0.90))
        next_state = next_state + process_noise * 0.30 * torch.randn(5, generator=gen)
        states[t + 1] = torch.clamp(next_state, min=1e-4, max=max_state_value)

    return states


def main():
    _configure_matplotlib()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{timestamp}_5vs6_holling")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    # Step 1: Generate 6-species Holling system
    print("Generating 6-species Holling system...")
    system_B = generate_partial_nonlinear_mvp_system(seed=42)
    states_B = system_B.full_states  # (T, 6)
    visible_B = states_B[:, :5]
    hidden_B = states_B[:, 5]
    env_driver = system_B.environment_driver
    pulse_driver = system_B.pulse_driver
    T = int(states_B.shape[0])
    print(f"  T = {T}, moderate_complexity = {system_B.diagnostics['moderate_complexity']}")

    # Step 2: Extract 5-species subset params
    growth_5 = system_B.growth_rates[:5].clone()
    interaction_6x6 = system_B.interaction_matrix.clone()
    interaction_5x5 = interaction_6x6[:5, :5].clone()
    env_loadings_5 = system_B.environment_loadings[:5].clone()
    pulse_loadings_5 = system_B.pulse_loadings[:5].clone()
    allee_5 = system_B.allee_thresholds[:5].clone()
    delay_5 = system_B.delay_coefficients[:5].clone()
    initial_5 = states_B[0, :5].clone()

    # Step 3: Simulate pure 5-species Holling system
    print("\nSimulating pure 5-species Holling system...")
    states_A = simulate_reduced_5species_holling(
        growth_5, interaction_5x5, env_loadings_5, pulse_loadings_5,
        allee_5, delay_5, initial_5, env_driver, pulse_driver,
        noise_seed=42 + 1000,
    )
    visible_A = states_A  # (T, 5)

    # Step 4: Compute metrics per species
    from scipy.stats import spearmanr
    per_species = []
    print("\n5 vs 6 trajectory differences (Holling):")
    for i in range(5):
        a = visible_A[:, i].numpy()
        b = visible_B[:, i].numpy()
        rmse = float(np.sqrt(((a - b) ** 2).mean()))
        rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))
        pearson = float(np.corrcoef(a, b)[0, 1])
        per_species.append({"species": f"v{i+1}", "rmse": rmse, "rel_l2": rel_l2, "pearson": pearson})
        print(f"  v{i+1}: RMSE={rmse:.4f}  rel_L2={rel_l2:.3f}  Pearson={pearson:+.3f}")

    all_rmse = float(np.sqrt(((visible_A.numpy() - visible_B.numpy()) ** 2).mean()))
    mean_pearson = float(np.mean([m["pearson"] for m in per_species]))
    rel_signal = float(np.abs(visible_A.numpy() - visible_B.numpy()).mean() / visible_B.numpy().std())
    print(f"\n  Overall: RMSE={all_rmse:.4f}  mean_Pearson={mean_pearson:.3f}  rel_signal={rel_signal:.3f}")

    # Step 5: Plots
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    time_axis = np.arange(T)
    for i, ax in enumerate(axes):
        a = visible_A[:, i].numpy()
        b = visible_B[:, i].numpy()
        ax.plot(time_axis, b, color="black", linewidth=1.4, label=f"B: 6-species v{i+1}")
        ax.plot(time_axis, a, color="#ff7f0e", linewidth=1.0, linestyle="--", alpha=0.85, label=f"A: pure 5-species v{i+1}")
        m = per_species[i]
        ax.set_title(f"v{i+1}: Pearson={m['pearson']:+.3f} RMSE={m['rmse']:.3f}", fontsize=11)
        ax.set_ylabel("丰度")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("时间步")
    fig.suptitle("Holling 5物种 vs 6物种系统 v1-v5 轨迹对比", fontsize=13, fontweight="bold", y=1.01)
    fig.savefig(out_dir / "fig_trajectories.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Step 6: Save trajectories for downstream sparse baseline experiment
    np.savez(
        out_dir / "trajectories.npz",
        states_A_5species=visible_A.numpy(),
        states_B_5species=visible_B.numpy(),
        hidden_B=hidden_B.numpy(),
        environment_driver=env_driver.numpy(),
        pulse_driver=pulse_driver.numpy(),
        growth_rates_full=system_B.growth_rates.numpy(),
        interaction_matrix_full=interaction_6x6.numpy(),
        environment_loadings_full=system_B.environment_loadings.numpy(),
        pulse_loadings_full=system_B.pulse_loadings.numpy(),
        allee_thresholds_full=system_B.allee_thresholds.numpy(),
        delay_coefficients_full=system_B.delay_coefficients.numpy(),
    )

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Holling 5 vs 6 对比\n\n")
        f.write("## 每物种差异\n\n")
        f.write("| 物种 | RMSE | rel_L2 | Pearson |\n|---|---|---|---|\n")
        for m in per_species:
            f.write(f"| {m['species']} | {m['rmse']:.4f} | {m['rel_l2']:.3f} | {m['pearson']:+.3f} |\n")
        f.write(f"\n**整体**: RMSE={all_rmse:.4f}, mean_Pearson={mean_pearson:.3f}, rel_signal={rel_signal:.3f}\n")

    print(f"\n[OK] saved to: {out_dir}")
    return out_dir


if __name__ == "__main__":
    main()

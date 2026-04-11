"""合成生态系统数据生成器（部分观测 Lotka-Volterra + 环境驱动）。

生成一个包含以下组件的模拟生态系统：
  - 5 个可见物种 + 1 个隐藏物种（总计 6 物种 Lotka-Volterra 系统）
  - 1 个 OU 环境驱动变量（温度/降水等外部因子的抽象）
  - 1 个脉冲干扰信号（模拟极端事件）
  - 竞争型种间交互矩阵（对角线为负的自限制项）
  - 各物种对环境和脉冲的不同敏感系数

主要函数：
  generate_partial_lv_mvp_system(config) → PartialLVMVPSystem
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch

from data.dataset import TimeSeriesBundle


@dataclass
class PartialLVMVPSystem:
    full_states: torch.Tensor
    visible_states: torch.Tensor
    hidden_states: torch.Tensor
    environment_driver: torch.Tensor
    pulse_driver: torch.Tensor
    growth_rates: torch.Tensor
    interaction_matrix: torch.Tensor
    environment_loadings: torch.Tensor
    pulse_loadings: torch.Tensor
    diagnostics: Dict[str, Any]
    generation_config: Dict[str, Any]

    @property
    def total_steps(self) -> int:
        return int(self.full_states.shape[0])

    @property
    def num_visible(self) -> int:
        return int(self.visible_states.shape[1])

    def to_bundle(self) -> TimeSeriesBundle:
        timestamps = [str(index) for index in range(self.total_steps)]
        return TimeSeriesBundle(
            observations=self.visible_states,
            covariates=torch.zeros(self.total_steps, 0, dtype=torch.float32),
            observed_names=[f"可见物种{index + 1}" for index in range(self.num_visible)],
            covariate_names=[],
            timestamps=timestamps,
            hidden_observations=self.hidden_states,
            hidden_names=["隐藏物种1"],
        )


def _build_candidate_parameters(
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    num_species = 6
    interaction = torch.zeros(num_species, num_species, dtype=torch.float32)
    diagonal = -(0.15 + 0.12 * torch.rand(num_species, generator=generator))
    interaction.diagonal().copy_(diagonal)

    for predator, prey, base_strength in [
        (1, 0, 0.34),
        (2, 1, 0.28),
        (3, 2, 0.31),
        (4, 3, 0.26),
        (0, 4, 0.23),
    ]:
        strength = base_strength * (0.82 + 0.42 * torch.rand(1, generator=generator).item())
        interaction[predator, prey] += strength
        interaction[prey, predator] -= (0.66 + 0.28 * torch.rand(1, generator=generator).item()) * strength

    for source, target, base_strength, sign in [
        (0, 2, 0.13, -1.0),
        (2, 0, 0.16, 1.0),
        (4, 1, 0.14, -1.0),
        (1, 4, 0.10, 1.0),
        (3, 0, 0.11, 1.0),
        (0, 3, 0.10, -1.0),
        (4, 2, 0.09, 1.0),
        (2, 4, 0.08, -1.0),
    ]:
        if torch.rand(1, generator=generator).item() < 0.62:
            strength = base_strength * (0.74 + 0.44 * torch.rand(1, generator=generator).item())
            interaction[source, target] += sign * strength

    for visible_index, sign, base_strength in [
        (0, -1.0, 0.46),
        (2, 1.0, 0.34),
        (4, -1.0, 0.29),
    ]:
        strength = base_strength * (0.86 + 0.38 * torch.rand(1, generator=generator).item())
        interaction[visible_index, 5] += sign * strength

    for visible_index, sign, base_strength in [
        (1, 1.0, 0.31),
        (3, -1.0, 0.27),
        (0, 1.0, 0.12),
    ]:
        strength = base_strength * (0.84 + 0.42 * torch.rand(1, generator=generator).item())
        interaction[5, visible_index] += sign * strength

    interaction += 0.012 * torch.randn(num_species, num_species, generator=generator)
    interaction.diagonal().copy_(diagonal)

    growth_rates = 0.15 + 0.14 * torch.rand(num_species, generator=generator)
    growth_rates[5] = 0.10 + 0.07 * torch.rand(1, generator=generator).item()

    environment_loadings = torch.tensor(
        [0.19, -0.12, 0.15, -0.09, 0.11, 0.04],
        dtype=torch.float32,
    )
    environment_loadings = environment_loadings * (
        0.86 + 0.32 * torch.rand(num_species, generator=generator)
    )

    pulse_loadings = torch.tensor(
        [0.12, -0.08, 0.11, 0.05, -0.09, 0.03],
        dtype=torch.float32,
    )
    pulse_loadings = pulse_loadings * (0.84 + 0.36 * torch.rand(num_species, generator=generator))
    return growth_rates, interaction, environment_loadings, pulse_loadings


def _simulate_discrete_lv(
    growth_rates: torch.Tensor,
    interaction_matrix: torch.Tensor,
    environment_loadings: torch.Tensor,
    pulse_loadings: torch.Tensor,
    total_steps: int,
    warmup_steps: int,
    process_noise: float,
    seed: int,
    max_state_value: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    total_length = total_steps + warmup_steps
    states = torch.zeros(total_length, 6, dtype=torch.float32)
    states[0] = 0.38 + 0.76 * torch.rand(6, generator=generator)

    environment_driver = torch.zeros(total_length, 1, dtype=torch.float32)
    pulse_driver = torch.zeros(total_length, 1, dtype=torch.float32)

    environment_value = 0.0
    environment_phase = 2.0 * math.pi * torch.rand(1, generator=generator).item()
    pulse_state = 0.0

    for time_index in range(total_length - 1):
        current = states[time_index]

        phase_velocity = 0.17 + 0.028 * math.sin(time_index / 93.0) + 0.010 * torch.randn(
            1,
            generator=generator,
        ).item()
        environment_phase += phase_velocity

        carrier = (
            0.65 * math.sin(environment_phase)
            + 0.24 * math.sin(0.53 * environment_phase + 0.35 * math.sin(time_index / 77.0))
            + 0.11 * math.sin(time_index / 39.0 + 0.18 * environment_phase)
        )
        environment_value = (
            0.88 * environment_value
            + 0.20 * carrier
            + 0.032 * torch.randn(1, generator=generator).item()
        )

        pulse_state *= 0.82
        if torch.rand(1, generator=generator).item() < 0.018:
            pulse_sign = 1.0 if torch.rand(1, generator=generator).item() > 0.46 else -1.0
            pulse_state += pulse_sign * (0.10 + 0.08 * torch.rand(1, generator=generator).item())

        drive = growth_rates + interaction_matrix @ current
        drive = drive + environment_loadings * environment_value + pulse_loadings * pulse_state
        drive = drive + process_noise * 0.65 * torch.randn(6, generator=generator)

        next_state = current * torch.exp(torch.clamp(drive, min=-1.12, max=0.92))
        next_state = next_state + process_noise * 0.35 * torch.randn(6, generator=generator)

        states[time_index + 1] = torch.clamp(next_state, min=1e-4, max=max_state_value)
        environment_driver[time_index, 0] = environment_value
        pulse_driver[time_index, 0] = pulse_state

    environment_driver[-1, 0] = environment_value
    pulse_driver[-1, 0] = pulse_state
    return (
        states[warmup_steps:],
        environment_driver[warmup_steps:],
        pulse_driver[warmup_steps:],
    )


def _moving_average(series: torch.Tensor, kernel_size: int = 7) -> torch.Tensor:
    padded = torch.nn.functional.pad(
        series.view(1, 1, -1),
        (kernel_size // 2, kernel_size // 2),
        mode="replicate",
    )
    kernel = torch.ones(1, 1, kernel_size, dtype=series.dtype, device=series.device) / kernel_size
    return torch.conv1d(padded, kernel).view(-1)


def _count_local_extrema(series: torch.Tensor) -> int:
    smoothed = _moving_average(series)
    diffs = smoothed[1:] - smoothed[:-1]
    amplitude = float((smoothed.max() - smoothed.min()).item())
    threshold = 0.015 * amplitude + 1e-6
    turning_points = (
        (diffs[:-1] * diffs[1:] < 0)
        & (diffs[:-1].abs() > threshold)
        & (diffs[1:].abs() > threshold)
    )
    return int(turning_points.sum().item())


def _extrema_positions(series: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    smoothed = _moving_average(series)
    diffs = smoothed[1:] - smoothed[:-1]
    amplitude = float((smoothed.max() - smoothed.min()).item())
    threshold = 0.015 * amplitude + 1e-6
    turning_points = (
        (diffs[:-1] * diffs[1:] < 0)
        & (diffs[:-1].abs() > threshold)
        & (diffs[1:].abs() > threshold)
    )
    return torch.nonzero(turning_points, as_tuple=False).view(-1) + 1, smoothed


def _variation_cv(values: torch.Tensor) -> float:
    if values.numel() < 2:
        return 1.0
    return float((values.std(unbiased=False) / (values.abs().mean() + 1e-6)).item())


def _dominant_frequency_ratio(series: torch.Tensor) -> float:
    centered = series - series.mean()
    spectrum = torch.fft.rfft(centered)
    power = (spectrum.abs() ** 2)[1:]
    if power.numel() == 0:
        return 0.0
    return float((power.max() / (power.sum() + 1e-6)).item())


def _collect_diagnostics(
    states: torch.Tensor,
    environment_driver: torch.Tensor,
    interaction_matrix: torch.Tensor,
    environment_loadings: torch.Tensor,
    growth_rates: torch.Tensor,
    max_state_value: float,
) -> Dict[str, Any]:
    visible = states[:, :5]
    hidden = states[:, 5]
    environment = environment_driver[:, 0]

    visible_std = visible.std(dim=0, unbiased=False)
    visible_range_ratio = (visible.max(dim=0).values - visible.min(dim=0).values) / (
        visible.mean(dim=0) + 1e-6
    )
    extrema_counts = [_count_local_extrema(visible[:, index]) for index in range(5)]
    interval_cv = []
    amplitude_cv = []
    dominant_frequency_ratio = []
    for index in range(5):
        positions, smoothed = _extrema_positions(visible[:, index])
        if positions.numel() >= 5:
            intervals = (positions[1:] - positions[:-1]).float()
            values = smoothed[positions]
            interval_cv.append(_variation_cv(intervals))
            amplitude_cv.append(_variation_cv(values))
        else:
            interval_cv.append(1.0)
            amplitude_cv.append(1.0)
        dominant_frequency_ratio.append(_dominant_frequency_ratio(visible[:, index]))

    visible_corr = torch.corrcoef(visible.T)
    visible_corr.fill_diagonal_(0.0)

    hidden_std = float(hidden.std(unbiased=False).item())
    hidden_range_ratio = float(((hidden.max() - hidden.min()) / (hidden.mean() + 1e-6)).item())
    hidden_to_visible_effect = (
        interaction_matrix[:5, 5].abs() * hidden.mean() / (visible.mean(dim=0) + 1e-6)
    )
    hidden_effect_visible_count = int(sum(value > 0.11 for value in hidden_to_visible_effect.tolist()))

    environment_std = float(environment.std(unbiased=False).item())
    environment_range = float((environment.max() - environment.min()).item())
    environment_dominant_frequency = _dominant_frequency_ratio(environment)
    environment_to_visible_effect = environment_loadings[:5].abs() * environment.std(unbiased=False)
    environment_effect_visible_count = int(
        sum(value > 0.03 for value in environment_to_visible_effect.tolist())
    )

    saturation_fraction = float(((states < 0.03) | (states > max_state_value * 0.94)).float().mean().item())
    mean_std = float(visible_std.mean().item())
    mean_range_ratio = float(visible_range_ratio.mean().item())
    mean_extrema = float(sum(extrema_counts) / len(extrema_counts))
    mean_interval_cv = float(sum(interval_cv) / len(interval_cv))
    mean_amplitude_cv = float(sum(amplitude_cv) / len(amplitude_cv))

    too_flat = bool(
        mean_std < 0.17
        or sum(value > 0.10 for value in visible_std.tolist()) < 3
        or sum(value >= 4 for value in extrema_counts) < 3
        or mean_range_ratio < 0.70
        or environment_std < 0.07
    )
    periodic_species = sum(
        interval < 0.16 and amplitude < 0.20 and spectrum > 0.34
        for interval, amplitude, spectrum in zip(interval_cv, amplitude_cv, dominant_frequency_ratio)
    )
    too_periodic = bool(
        periodic_species >= 3
        or (
            float(visible_corr.abs().max().item()) > 0.94
            and sum(value > 0.36 for value in dominant_frequency_ratio) >= 3
        )
        or (
            mean_interval_cv < 0.18
            and mean_amplitude_cv < 0.18
            and environment_dominant_frequency > 0.44
        )
    )
    too_spiky_internal = bool(
        mean_extrema > 72.0
        or max(extrema_counts) > 80
        or float(visible_range_ratio.max().item()) > 7.8
        or hidden_std > 0.95
    )
    moderate_complexity = bool(
        not too_flat
        and not too_periodic
        and not too_spiky_internal
        and hidden_effect_visible_count >= 2
        and environment_effect_visible_count >= 2
        and saturation_fraction < 0.08
        and mean_interval_cv > 0.18
        and mean_amplitude_cv > 0.16
    )

    diagnostics = {
        "visible_std": [float(value) for value in visible_std.tolist()],
        "visible_range_ratio": [float(value) for value in visible_range_ratio.tolist()],
        "visible_extrema_count": extrema_counts,
        "visible_max_abs_corr": float(visible_corr.abs().max().item()),
        "visible_interval_cv": interval_cv,
        "visible_peak_amplitude_cv": amplitude_cv,
        "visible_dominant_frequency_ratio": dominant_frequency_ratio,
        "hidden_std": hidden_std,
        "hidden_range_ratio": hidden_range_ratio,
        "hidden_to_visible_effect": [float(value) for value in hidden_to_visible_effect.tolist()],
        "hidden_effect_visible_count": hidden_effect_visible_count,
        "mixed_hidden_signs": bool(
            (interaction_matrix[:5, 5] > 0).any().item()
            and (interaction_matrix[:5, 5] < 0).any().item()
            and (interaction_matrix[5, :5] > 0).any().item()
            and (interaction_matrix[5, :5] < 0).any().item()
        ),
        "environment_std": environment_std,
        "environment_range": environment_range,
        "environment_dominant_frequency_ratio": environment_dominant_frequency,
        "environment_to_visible_effect": [float(value) for value in environment_to_visible_effect.tolist()],
        "environment_effect_visible_count": environment_effect_visible_count,
        "saturation_fraction": saturation_fraction,
        "too_flat": too_flat,
        "too_periodic": too_periodic,
        "moderate_complexity": moderate_complexity,
        "too_spiky_internal": too_spiky_internal,
        "growth_rates": [float(value) for value in growth_rates.tolist()],
    }
    return diagnostics


def _passes_screening(diagnostics: Dict[str, Any]) -> bool:
    visible_std = diagnostics["visible_std"]
    visible_range_ratio = diagnostics["visible_range_ratio"]
    extrema_counts = diagnostics["visible_extrema_count"]
    hidden_effect = diagnostics["hidden_to_visible_effect"]
    environment_effect = diagnostics["environment_to_visible_effect"]

    return bool(
        sum(value > 0.10 for value in visible_std) >= 3
        and sum(value > 0.55 for value in visible_range_ratio) >= 3
        and sum(value >= 4 for value in extrema_counts) >= 3
        and diagnostics["hidden_std"] > 0.05
        and diagnostics["hidden_range_ratio"] > 0.24
        and sum(value > 0.11 for value in hidden_effect) >= 2
        and sum(value > 0.03 for value in environment_effect) >= 2
        and diagnostics["mixed_hidden_signs"]
        and diagnostics["environment_std"] > 0.07
        and diagnostics["saturation_fraction"] < 0.08
        and not diagnostics["too_flat"]
        and not diagnostics["too_periodic"]
        and not diagnostics["too_spiky_internal"]
        and diagnostics["moderate_complexity"]
    )


def generate_partial_lv_mvp_system(
    total_steps: int = 820,
    warmup_steps: int = 160,
    process_noise: float = 0.006,
    seed: int = 42,
    max_attempts: int = 640,
    max_state_value: float = 5.5,
) -> PartialLVMVPSystem:
    best_payload = None
    best_score = float("-inf")

    for attempt_index in range(max_attempts):
        generator = torch.Generator().manual_seed(seed + 97 * attempt_index)
        growth_rates, interaction_matrix, environment_loadings, pulse_loadings = _build_candidate_parameters(
            generator
        )
        states, environment_driver, pulse_driver = _simulate_discrete_lv(
            growth_rates=growth_rates,
            interaction_matrix=interaction_matrix,
            environment_loadings=environment_loadings,
            pulse_loadings=pulse_loadings,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            process_noise=process_noise,
            seed=seed + 1000 * (attempt_index + 1),
            max_state_value=max_state_value,
        )
        diagnostics = _collect_diagnostics(
            states=states,
            environment_driver=environment_driver,
            interaction_matrix=interaction_matrix,
            environment_loadings=environment_loadings,
            growth_rates=growth_rates,
            max_state_value=max_state_value,
        )
        mean_std = sum(diagnostics["visible_std"]) / 5.0
        mean_range_ratio = sum(diagnostics["visible_range_ratio"]) / 5.0
        mean_extrema = sum(diagnostics["visible_extrema_count"]) / 5.0
        mean_interval_cv = sum(diagnostics["visible_interval_cv"]) / 5.0
        mean_amplitude_cv = sum(diagnostics["visible_peak_amplitude_cv"]) / 5.0
        score = (
            -abs(mean_std - 0.46)
            - 0.20 * abs(mean_range_ratio - 1.70)
            - 0.03 * abs(mean_extrema - 18.0)
            + 0.22 * diagnostics["hidden_std"]
            + 0.16 * diagnostics["hidden_range_ratio"]
            + 0.12 * diagnostics["hidden_effect_visible_count"]
            + 0.10 * diagnostics["environment_effect_visible_count"]
            + 0.10 * mean_interval_cv
            + 0.10 * mean_amplitude_cv
            - 0.22 * diagnostics["visible_max_abs_corr"]
            - 0.08 * diagnostics["environment_dominant_frequency_ratio"]
            - 1.20 * float(diagnostics["too_flat"])
            - 1.20 * float(diagnostics["too_periodic"])
            - 0.80 * float(diagnostics["too_spiky_internal"])
            - 1.50 * diagnostics["saturation_fraction"]
        )

        payload = (
            states,
            environment_driver,
            pulse_driver,
            growth_rates,
            interaction_matrix,
            environment_loadings,
            pulse_loadings,
            diagnostics,
            attempt_index,
        )
        if score > best_score:
            best_score = score
            best_payload = payload
        if _passes_screening(diagnostics):
            best_payload = payload
            break

    if best_payload is None:
        raise RuntimeError("Failed to generate a valid partial LV MVP system.")

    (
        states,
        environment_driver,
        pulse_driver,
        growth_rates,
        interaction_matrix,
        environment_loadings,
        pulse_loadings,
        diagnostics,
        attempt_index,
    ) = best_payload
    diagnostics = {
        **diagnostics,
        "screening_passed": _passes_screening(diagnostics),
        "selected_attempt": int(attempt_index),
    }

    return PartialLVMVPSystem(
        full_states=states,
        visible_states=states[:, :5],
        hidden_states=states[:, 5:].contiguous(),
        environment_driver=environment_driver.contiguous(),
        pulse_driver=pulse_driver.contiguous(),
        growth_rates=growth_rates,
        interaction_matrix=interaction_matrix,
        environment_loadings=environment_loadings,
        pulse_loadings=pulse_loadings,
        diagnostics=diagnostics,
        generation_config={
            "model": "discrete_generalized_lv_ricker_with_hidden_and_environment",
            "regime": "moderate_complexity_partial_observation",
            "visible_species": 5,
            "hidden_species": 1,
            "environment_latents": 1,
            "total_species": 6,
            "total_steps": total_steps,
            "warmup_steps": warmup_steps,
            "process_noise": process_noise,
            "seed": seed,
            "max_attempts": max_attempts,
            "max_state_value": max_state_value,
            "environment_ar": 0.88,
            "pulse_decay": 0.82,
            "pulse_probability": 0.018,
        },
    )

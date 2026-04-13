"""合成生态系统数据生成器（非 LV 版本：Holling + Allee + 时滞）。

核心区别于 partial_lv_mvp.py：
  - 种间相互作用是 Holling type II（饱和响应），不是 LV 的线性 x·y 项
  - 部分物种有 Allee effect（低密度时增长率变负）
  - 部分物种有时滞反馈（t-τ 步的自身影响）
  - 保留 5 visible + 1 hidden + environment + pulse 的结构，便于和 LV 数据做公平对比

目的：检验 LV 先验在"真实动力学不是 LV"时是帮助还是拖累。

主要函数：
  generate_partial_nonlinear_mvp_system(config) → PartialNonlinearMVPSystem
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

import torch

from data.dataset import TimeSeriesBundle


@dataclass
class PartialNonlinearMVPSystem:
    full_states: torch.Tensor
    visible_states: torch.Tensor
    hidden_states: torch.Tensor
    environment_driver: torch.Tensor
    pulse_driver: torch.Tensor
    growth_rates: torch.Tensor
    interaction_matrix: torch.Tensor  # holling attack rates (signed by predator/prey direction)
    environment_loadings: torch.Tensor
    pulse_loadings: torch.Tensor
    allee_thresholds: torch.Tensor
    delay_coefficients: torch.Tensor
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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构建非 LV 系统的参数。

    interaction_matrix[i,j] 在非 LV 系统中的含义：
      > 0 → i 捕食 j（i 是 predator）
      < 0 → i 被 j 捕食
      绝对值 = Holling attack rate (a_ij)

    另外每个物种有：
      allee_thresholds[i]：Allee 效应阈值（低于此密度增长率为负）
      delay_coefficients[i]：时滞反馈系数
    """
    num_species = 6

    # 基本捕食链 0 → 1 → 2 → 3 → 4 → 0（保持和 LV 版本相同的拓扑）
    interaction = torch.zeros(num_species, num_species, dtype=torch.float32)

    for predator, prey, base_strength in [
        (1, 0, 0.95),
        (2, 1, 0.85),
        (3, 2, 0.90),
        (4, 3, 0.78),
        (0, 4, 0.68),
    ]:
        strength = base_strength * (0.82 + 0.42 * torch.rand(1, generator=generator).item())
        interaction[predator, prey] += strength   # predator 吃 prey
        interaction[prey, predator] -= (0.82 + 0.30 * torch.rand(1, generator=generator).item()) * strength

    # Hidden 物种和 visible 的交互
    for visible_index, sign, base_strength in [
        (0, -1.0, 0.38),
        (2, 1.0, 0.30),
        (4, -1.0, 0.25),
    ]:
        strength = base_strength * (0.86 + 0.38 * torch.rand(1, generator=generator).item())
        interaction[visible_index, 5] += sign * strength

    for visible_index, sign, base_strength in [
        (1, 1.0, 0.27),
        (3, -1.0, 0.23),
        (0, 1.0, 0.10),
    ]:
        strength = base_strength * (0.84 + 0.42 * torch.rand(1, generator=generator).item())
        interaction[5, visible_index] += sign * strength

    interaction += 0.010 * torch.randn(num_species, num_species, generator=generator)

    growth_rates = 0.18 + 0.12 * torch.rand(num_species, generator=generator)
    growth_rates[5] = 0.12 + 0.06 * torch.rand(1, generator=generator).item()

    environment_loadings = torch.tensor(
        [0.17, -0.10, 0.13, -0.08, 0.09, 0.04],
        dtype=torch.float32,
    )
    environment_loadings = environment_loadings * (
        0.86 + 0.32 * torch.rand(num_species, generator=generator)
    )

    pulse_loadings = torch.tensor(
        [0.11, -0.07, 0.10, 0.04, -0.08, 0.03],
        dtype=torch.float32,
    )
    pulse_loadings = pulse_loadings * (0.84 + 0.36 * torch.rand(num_species, generator=generator))

    # Allee thresholds: 只有部分物种有 Allee effect
    # 阈值 0 表示没有 Allee effect, > 0 的物种在低密度时增长率会转负
    allee_thresholds = torch.zeros(num_species, dtype=torch.float32)
    allee_thresholds[0] = 0.15 + 0.08 * torch.rand(1, generator=generator).item()  # visible_1 有 Allee
    allee_thresholds[2] = 0.12 + 0.06 * torch.rand(1, generator=generator).item()  # visible_3 有 Allee
    allee_thresholds[5] = 0.18 + 0.10 * torch.rand(1, generator=generator).item()  # hidden 有 Allee

    # Delay coefficients: 时滞反馈强度
    delay_coefficients = 0.05 + 0.08 * torch.rand(num_species, generator=generator)
    delay_coefficients[4] *= 1.6  # visible_5 时滞最强
    delay_coefficients[5] *= 1.4  # hidden 也强

    return growth_rates, interaction, environment_loadings, pulse_loadings, allee_thresholds, delay_coefficients


def _holling2(prey: torch.Tensor, attack_rate: float, half_saturation: float) -> torch.Tensor:
    """Holling type II functional response: a*x/(1+h*x)。饱和响应。"""
    return attack_rate * prey / (1.0 + half_saturation * prey)


def _simulate_nonlinear(
    growth_rates: torch.Tensor,
    interaction_matrix: torch.Tensor,
    environment_loadings: torch.Tensor,
    pulse_loadings: torch.Tensor,
    allee_thresholds: torch.Tensor,
    delay_coefficients: torch.Tensor,
    total_steps: int,
    warmup_steps: int,
    process_noise: float,
    seed: int,
    max_state_value: float,
    delay_lag: int = 4,
    half_saturation: float = 0.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    total_length = total_steps + warmup_steps
    num_species = 6
    states = torch.zeros(total_length, num_species, dtype=torch.float32)
    states[0] = 0.40 + 0.72 * torch.rand(num_species, generator=generator)

    environment_driver = torch.zeros(total_length, 1, dtype=torch.float32)
    pulse_driver = torch.zeros(total_length, 1, dtype=torch.float32)

    environment_value = 0.0
    environment_phase = 2.0 * math.pi * torch.rand(1, generator=generator).item()
    pulse_state = 0.0

    for time_index in range(total_length - 1):
        current = states[time_index]

        # 环境驱动：多频正弦 + AR(1) 平滑
        phase_velocity = 0.17 + 0.028 * math.sin(time_index / 93.0) + 0.010 * torch.randn(
            1, generator=generator
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

        # 脉冲
        pulse_state *= 0.82
        if torch.rand(1, generator=generator).item() < 0.018:
            pulse_sign = 1.0 if torch.rand(1, generator=generator).item() > 0.46 else -1.0
            pulse_state += pulse_sign * (0.10 + 0.08 * torch.rand(1, generator=generator).item())

        # ---- 非 LV 动力学 ----
        drive = torch.zeros(num_species, dtype=torch.float32)

        # 1. 自身自限制（非 LV：用更强的 logistic 而不是简单 -d*x^2）
        self_limit = -0.12 * current.square() / (0.45 + current)  # saturating self-limitation

        # 2. Holling type II 种间相互作用（这是非 LV 的核心）
        for i in range(num_species):
            for j in range(num_species):
                if i == j:
                    continue
                a_ij = float(interaction_matrix[i, j].item())
                if abs(a_ij) < 1e-4:
                    continue
                # i 吃 j 时 (a_ij > 0): i 的增长率 += Holling(x_j)
                # i 被 j 吃 (a_ij < 0):   i 的增长率 -= Holling(x_i) * x_j 强度
                if a_ij > 0:
                    drive[i] += _holling2(current[j], a_ij, half_saturation)
                else:
                    # 被捕食的损失：x_j 越多，i 被吃得越多，但受 x_i 自身限制
                    drive[i] += a_ij * current[j] * current[i] / (0.3 + current[i])

        # 3. Allee effect: 只对 allee_thresholds > 0 的物种生效
        for i in range(num_species):
            A_i = float(allee_thresholds[i].item())
            if A_i > 0.0:
                # (x - A)/(K + x) 低于 A 时为负，高于 A 时为正
                drive[i] += 0.25 * (current[i] - A_i) / (0.5 + current[i])

        # 4. 时滞反馈
        if time_index >= delay_lag:
            past_state = states[time_index - delay_lag]
            drive = drive - delay_coefficients * (current - past_state)  # 回到过去的状态趋势

        # 5. 环境 + 脉冲（加性进入增长率）
        drive = drive + environment_loadings * environment_value + pulse_loadings * pulse_state

        # 6. 基础增长率
        drive = drive + growth_rates + self_limit

        # 7. 过程噪声
        drive = drive + process_noise * 0.70 * torch.randn(num_species, generator=generator)

        # Ricker-style 更新（保持和 LV 版本类似的数值范围，便于对比）
        next_state = current * torch.exp(torch.clamp(drive, min=-1.10, max=0.90))
        next_state = next_state + process_noise * 0.30 * torch.randn(num_species, generator=generator)

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


# ---------------------------------------------------------------------------
# 以下诊断函数和 partial_lv_mvp.py 基本一致（复用），便于筛选数据质量
# ---------------------------------------------------------------------------
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


def _collect_diagnostics(
    states: torch.Tensor,
    environment_driver: torch.Tensor,
    interaction_matrix: torch.Tensor,
    max_state_value: float,
) -> Dict[str, Any]:
    visible = states[:, :5]
    hidden = states[:, 5]

    visible_std = visible.std(dim=0, unbiased=False)
    visible_range_ratio = (visible.max(dim=0).values - visible.min(dim=0).values) / (
        visible.mean(dim=0) + 1e-6
    )
    extrema_counts = [_count_local_extrema(visible[:, index]) for index in range(5)]
    hidden_std = float(hidden.std(unbiased=False).item())
    hidden_range_ratio = float(((hidden.max() - hidden.min()) / (hidden.mean() + 1e-6)).item())

    visible_corr = torch.corrcoef(visible.T)
    visible_corr.fill_diagonal_(0.0)

    saturation_fraction = float(((states < 0.03) | (states > max_state_value * 0.94)).float().mean().item())
    mean_std = float(visible_std.mean().item())
    mean_range_ratio = float(visible_range_ratio.mean().item())
    mean_extrema = float(sum(extrema_counts) / len(extrema_counts))

    hidden_to_visible_effect = (
        interaction_matrix[:5, 5].abs() * hidden.mean() / (visible.mean(dim=0) + 1e-6)
    )
    hidden_effect_visible_count = int(sum(value > 0.10 for value in hidden_to_visible_effect.tolist()))

    too_flat = bool(
        mean_std < 0.15
        or sum(value > 0.08 for value in visible_std.tolist()) < 3
        or sum(value >= 3 for value in extrema_counts) < 3
        or mean_range_ratio < 0.55
    )
    too_spiky_internal = bool(
        mean_extrema > 75.0
        or float(visible_range_ratio.max().item()) > 8.0
        or hidden_std > 1.0
    )
    moderate_complexity = bool(
        not too_flat
        and not too_spiky_internal
        and hidden_effect_visible_count >= 2
        and saturation_fraction < 0.10
    )

    return {
        "visible_std": [float(value) for value in visible_std.tolist()],
        "visible_range_ratio": [float(value) for value in visible_range_ratio.tolist()],
        "visible_extrema_count": extrema_counts,
        "visible_max_abs_corr": float(visible_corr.abs().max().item()),
        "hidden_std": hidden_std,
        "hidden_range_ratio": hidden_range_ratio,
        "hidden_to_visible_effect": [float(value) for value in hidden_to_visible_effect.tolist()],
        "hidden_effect_visible_count": hidden_effect_visible_count,
        "saturation_fraction": saturation_fraction,
        "too_flat": too_flat,
        "too_periodic": False,  # non-linear systems less periodic, skip this check
        "moderate_complexity": moderate_complexity,
        "too_spiky_internal": too_spiky_internal,
    }


def _passes_screening(diagnostics: Dict[str, Any]) -> bool:
    visible_std = diagnostics["visible_std"]
    visible_range_ratio = diagnostics["visible_range_ratio"]
    extrema_counts = diagnostics["visible_extrema_count"]
    hidden_effect = diagnostics["hidden_to_visible_effect"]

    return bool(
        sum(value > 0.08 for value in visible_std) >= 3
        and sum(value > 0.40 for value in visible_range_ratio) >= 3
        and sum(value >= 3 for value in extrema_counts) >= 3
        and diagnostics["hidden_std"] > 0.04
        and diagnostics["hidden_range_ratio"] > 0.20
        and sum(value > 0.10 for value in hidden_effect) >= 2
        and diagnostics["saturation_fraction"] < 0.10
        and not diagnostics["too_flat"]
        and not diagnostics["too_spiky_internal"]
        and diagnostics["moderate_complexity"]
    )


def generate_partial_nonlinear_mvp_system(
    total_steps: int = 820,
    warmup_steps: int = 160,
    process_noise: float = 0.006,
    seed: int = 42,
    max_attempts: int = 640,
    max_state_value: float = 5.5,
) -> PartialNonlinearMVPSystem:
    best_payload = None
    best_score = float("-inf")

    for attempt_index in range(max_attempts):
        generator = torch.Generator().manual_seed(seed + 97 * attempt_index)
        (
            growth_rates,
            interaction_matrix,
            environment_loadings,
            pulse_loadings,
            allee_thresholds,
            delay_coefficients,
        ) = _build_candidate_parameters(generator)

        states, environment_driver, pulse_driver = _simulate_nonlinear(
            growth_rates=growth_rates,
            interaction_matrix=interaction_matrix,
            environment_loadings=environment_loadings,
            pulse_loadings=pulse_loadings,
            allee_thresholds=allee_thresholds,
            delay_coefficients=delay_coefficients,
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
            max_state_value=max_state_value,
        )
        mean_std = sum(diagnostics["visible_std"]) / 5.0
        mean_range_ratio = sum(diagnostics["visible_range_ratio"]) / 5.0
        mean_extrema = sum(diagnostics["visible_extrema_count"]) / 5.0
        score = (
            -abs(mean_std - 0.42)
            - 0.20 * abs(mean_range_ratio - 1.50)
            - 0.03 * abs(mean_extrema - 16.0)
            + 0.22 * diagnostics["hidden_std"]
            + 0.16 * diagnostics["hidden_range_ratio"]
            + 0.12 * diagnostics["hidden_effect_visible_count"]
            - 0.22 * diagnostics["visible_max_abs_corr"]
            - 1.20 * float(diagnostics["too_flat"])
            - 0.80 * float(diagnostics["too_spiky_internal"])
            - 1.50 * diagnostics["saturation_fraction"]
        )

        payload = (
            states, environment_driver, pulse_driver,
            growth_rates, interaction_matrix, environment_loadings, pulse_loadings,
            allee_thresholds, delay_coefficients,
            diagnostics, attempt_index,
        )
        if score > best_score:
            best_score = score
            best_payload = payload
        if _passes_screening(diagnostics):
            best_payload = payload
            break

    if best_payload is None:
        raise RuntimeError("Failed to generate a valid partial non-linear MVP system.")

    (
        states, environment_driver, pulse_driver,
        growth_rates, interaction_matrix, environment_loadings, pulse_loadings,
        allee_thresholds, delay_coefficients,
        diagnostics, attempt_index,
    ) = best_payload
    diagnostics = {
        **diagnostics,
        "screening_passed": _passes_screening(diagnostics),
        "selected_attempt": int(attempt_index),
    }

    return PartialNonlinearMVPSystem(
        full_states=states,
        visible_states=states[:, :5],
        hidden_states=states[:, 5:].contiguous(),
        environment_driver=environment_driver.contiguous(),
        pulse_driver=pulse_driver.contiguous(),
        growth_rates=growth_rates,
        interaction_matrix=interaction_matrix,
        environment_loadings=environment_loadings,
        pulse_loadings=pulse_loadings,
        allee_thresholds=allee_thresholds,
        delay_coefficients=delay_coefficients,
        diagnostics=diagnostics,
        generation_config={
            "model": "nonlinear_holling2_allee_delay",
            "regime": "non_lv_partial_observation",
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
            "holling_half_saturation": 0.45,
            "delay_lag": 4,
            "has_allee_effect": True,
            "has_time_delay": True,
            "non_lv_mechanisms": ["holling_type_2", "allee_effect", "delayed_feedback"],
        },
    )

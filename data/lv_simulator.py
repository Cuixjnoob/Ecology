from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from data.dataset import TimeSeriesBundle


@dataclass
class HiddenLotkaVolterraSimulation:
    full_states: torch.Tensor
    observed_states: torch.Tensor
    hidden_states: torch.Tensor
    growth_rates: torch.Tensor
    interaction_matrix: torch.Tensor
    observed_names: List[str]
    hidden_names: List[str]
    timestamps: List[str]
    dt: float

    @property
    def num_observed(self) -> int:
        return int(self.observed_states.shape[1])

    @property
    def num_hidden(self) -> int:
        return int(self.hidden_states.shape[1])

    @property
    def total_steps(self) -> int:
        return int(self.observed_states.shape[0])

    def to_observed_bundle(self) -> TimeSeriesBundle:
        return TimeSeriesBundle(
            observations=self.observed_states,
            covariates=torch.zeros(self.total_steps, 0, dtype=torch.float32),
            observed_names=self.observed_names,
            covariate_names=[],
            timestamps=self.timestamps,
            hidden_observations=self.hidden_states,
            hidden_names=self.hidden_names,
        )


def _build_structured_interaction_matrix(
    total_species: int,
    num_observed: int,
    num_hidden: int,
    generator: torch.Generator,
) -> torch.Tensor:
    interaction = torch.zeros(total_species, total_species, dtype=torch.float32)

    diagonal = -0.22 - 0.06 * torch.rand(total_species, generator=generator)
    interaction.diagonal().copy_(diagonal)

    ring_strength = 0.035 + 0.015 * torch.rand(total_species - 1, generator=generator)
    for species_index in range(total_species - 1):
        interaction[species_index, species_index + 1] += ring_strength[species_index]
        interaction[species_index + 1, species_index] -= 0.8 * ring_strength[species_index]
    interaction[total_species - 1, 0] += 0.04
    interaction[0, total_species - 1] -= 0.03

    random_component = 0.018 * torch.randn(total_species, total_species, generator=generator)
    interaction += random_component
    interaction.diagonal().copy_(diagonal)

    hidden_offset = num_observed
    for hidden_index in range(num_hidden):
        hidden_species = hidden_offset + hidden_index
        if hidden_index == 0:
            positive_targets = slice(0, num_observed // 2)
            negative_targets = slice(num_observed // 2, num_observed)
        else:
            positive_targets = slice(1, num_observed, 2)
            negative_targets = slice(0, num_observed, 2)

        interaction[positive_targets, hidden_species] += 0.06
        interaction[negative_targets, hidden_species] -= 0.05
        interaction[hidden_species, positive_targets] -= 0.05
        interaction[hidden_species, negative_targets] += 0.035

    return interaction


def _lotka_volterra_rhs(
    state: torch.Tensor,
    growth_rates: torch.Tensor,
    interaction_matrix: torch.Tensor,
) -> torch.Tensor:
    return state * (growth_rates + interaction_matrix @ state)


def _simulate_glv_trajectory(
    initial_state: torch.Tensor,
    growth_rates: torch.Tensor,
    interaction_matrix: torch.Tensor,
    total_steps: int,
    dt: float,
    process_noise: float,
    generator: torch.Generator,
) -> torch.Tensor:
    total_species = initial_state.shape[0]
    states = torch.zeros(total_steps, total_species, dtype=torch.float32)
    states[0] = initial_state

    for step_index in range(total_steps - 1):
        current_state = states[step_index]

        k1 = _lotka_volterra_rhs(current_state, growth_rates, interaction_matrix)
        k2 = _lotka_volterra_rhs(current_state + 0.5 * dt * k1, growth_rates, interaction_matrix)
        k3 = _lotka_volterra_rhs(current_state + 0.5 * dt * k2, growth_rates, interaction_matrix)
        k4 = _lotka_volterra_rhs(current_state + dt * k3, growth_rates, interaction_matrix)

        next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        noise = process_noise * torch.randn(total_species, generator=generator)
        next_state = torch.clamp(next_state + noise, min=1e-4, max=6.0)
        states[step_index + 1] = next_state
    return states


def _dynamics_is_rich_enough(states: torch.Tensor) -> tuple[bool, float]:
    mean_values = states.mean(dim=0).clamp_min(1e-6)
    std_values = states.std(dim=0, unbiased=False)
    cv = std_values / mean_values
    range_ratio = (states.max(dim=0).values - states.min(dim=0).values) / mean_values

    first_diff = states[1:] - states[:-1]
    sign_changes = ((first_diff[:-1] * first_diff[1:]) < 0).float().sum(dim=0)
    decline_fraction = (first_diff < 0).float().mean(dim=0)
    hidden_cv = float(cv[-1].item())
    hidden_range_ratio = float(range_ratio[-1].item())
    hidden_sign_changes = float(sign_changes[-1].item())
    hidden_near_ceiling = float((states[:, -1] > 5.6).float().mean().item())

    score = float(cv.mean() + 0.6 * range_ratio.mean() + 0.02 * sign_changes.mean())
    is_rich = (
        float(cv.mean()) > 0.18
        and float(range_ratio.mean()) > 0.75
        and float((sign_changes >= 6).float().mean()) > 0.45
        and float((decline_fraction > 0.8).float().mean()) < 0.5
        and hidden_cv > 0.12
        and hidden_range_ratio > 0.45
        and hidden_sign_changes >= 6.0
        and hidden_near_ceiling < 0.2
    )
    return is_rich, score


def _build_rich_interaction_matrix(
    total_species: int,
    num_observed: int,
    num_hidden: int,
    generator: torch.Generator,
) -> torch.Tensor:
    interaction = torch.zeros(total_species, total_species, dtype=torch.float32)
    diagonal = -0.08 - 0.05 * torch.rand(total_species, generator=generator)
    interaction.diagonal().copy_(diagonal)

    for species_index in range(total_species - 1):
        strength = 0.10 + 0.08 * torch.rand(1, generator=generator).item()
        if species_index % 2 == 0:
            interaction[species_index, species_index + 1] += strength
            interaction[species_index + 1, species_index] -= 1.15 * strength
        else:
            interaction[species_index, species_index + 1] -= 0.8 * strength
            interaction[species_index + 1, species_index] += 0.95 * strength

    interaction += 0.035 * torch.randn(total_species, total_species, generator=generator)
    interaction.diagonal().copy_(diagonal)

    hidden_index = total_species - 1
    interaction[hidden_index, hidden_index] = -0.10 - 0.04 * torch.rand(1, generator=generator).item()
    anchor_a = 0
    anchor_b = min(3, num_observed - 1)
    anchor_c = min(6, num_observed - 1)
    interaction[hidden_index, anchor_a] += 0.22
    interaction[anchor_a, hidden_index] -= 0.24
    interaction[hidden_index, anchor_b] -= 0.18
    interaction[anchor_b, hidden_index] += 0.16
    interaction[hidden_index, anchor_c] += 0.12
    interaction[anchor_c, hidden_index] -= 0.10
    interaction[hidden_index, :num_observed:2] += 0.05
    interaction[hidden_index, 1:num_observed:2] -= 0.04
    return interaction


def generate_hidden_lv_simulation(
    total_steps: int = 520,
    warmup_steps: int = 120,
    num_observed: int = 10,
    num_hidden: int = 2,
    dt: float = 0.04,
    process_noise: float = 0.002,
    seed: int = 42,
) -> HiddenLotkaVolterraSimulation:
    if num_hidden <= 0:
        raise ValueError("num_hidden must be positive for the hidden-species experiment.")

    total_species = num_observed + num_hidden
    generator = torch.Generator().manual_seed(seed)

    growth_rates = 0.05 + 0.06 * torch.rand(total_species, generator=generator)
    growth_rates[num_observed:] *= 0.7
    interaction_matrix = _build_structured_interaction_matrix(
        total_species=total_species,
        num_observed=num_observed,
        num_hidden=num_hidden,
        generator=generator,
    )

    total_simulation_steps = total_steps + warmup_steps
    full_states = torch.zeros(total_simulation_steps, total_species, dtype=torch.float32)
    full_states[0] = 0.4 + 0.5 * torch.rand(total_species, generator=generator)

    for step_index in range(total_simulation_steps - 1):
        current_state = full_states[step_index]

        k1 = _lotka_volterra_rhs(current_state, growth_rates, interaction_matrix)
        k2 = _lotka_volterra_rhs(current_state + 0.5 * dt * k1, growth_rates, interaction_matrix)
        k3 = _lotka_volterra_rhs(current_state + 0.5 * dt * k2, growth_rates, interaction_matrix)
        k4 = _lotka_volterra_rhs(current_state + dt * k3, growth_rates, interaction_matrix)

        next_state = current_state + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        noise = process_noise * torch.randn(total_species, generator=generator)
        next_state = torch.clamp(next_state + noise, min=1e-4, max=6.0)
        full_states[step_index + 1] = next_state

    effective_states = full_states[warmup_steps:]
    observed_states = effective_states[:, :num_observed]
    hidden_states = effective_states[:, num_observed:]

    return HiddenLotkaVolterraSimulation(
        full_states=effective_states,
        observed_states=observed_states,
        hidden_states=hidden_states,
        growth_rates=growth_rates,
        interaction_matrix=interaction_matrix,
        observed_names=[f"可见物种{i + 1}" for i in range(num_observed)],
        hidden_names=[f"隐藏物种{i + 1}" for i in range(num_hidden)],
        timestamps=[str(index) for index in range(total_steps)],
        dt=dt,
    )


def generate_rich_hidden_lv_simulation(
    total_steps: int = 620,
    warmup_steps: int = 180,
    num_observed: int = 9,
    num_hidden: int = 1,
    dt: float = 0.03,
    process_noise: float = 0.0015,
    seed: int = 42,
    max_attempts: int = 64,
) -> HiddenLotkaVolterraSimulation:
    if num_hidden <= 0:
        raise ValueError("num_hidden must be positive.")

    total_species = num_observed + num_hidden
    best_payload = None
    best_score = float("-inf")

    for attempt_index in range(max_attempts):
        generator = torch.Generator().manual_seed(seed + 97 * attempt_index)
        growth_rates = 0.10 + 0.10 * torch.rand(total_species, generator=generator)
        growth_rates[-1] = 0.03 + 0.03 * torch.rand(1, generator=generator).item()
        interaction_matrix = _build_rich_interaction_matrix(
            total_species=total_species,
            num_observed=num_observed,
            num_hidden=num_hidden,
            generator=generator,
        )
        initial_state = 0.5 + 0.8 * torch.rand(total_species, generator=generator)
        full_states = _simulate_glv_trajectory(
            initial_state=initial_state,
            growth_rates=growth_rates,
            interaction_matrix=interaction_matrix,
            total_steps=total_steps + warmup_steps,
            dt=dt,
            process_noise=process_noise,
            generator=generator,
        )
        effective_states = full_states[warmup_steps:]
        is_rich, score = _dynamics_is_rich_enough(effective_states)
        if score > best_score:
            best_score = score
            best_payload = (effective_states, growth_rates, interaction_matrix)
        if is_rich:
            best_payload = (effective_states, growth_rates, interaction_matrix)
            break

    if best_payload is None:
        raise RuntimeError("Failed to generate a valid rich LV simulation.")

    effective_states, growth_rates, interaction_matrix = best_payload
    observed_states = effective_states[:, :num_observed]
    hidden_states = effective_states[:, num_observed:]

    return HiddenLotkaVolterraSimulation(
        full_states=effective_states,
        observed_states=observed_states,
        hidden_states=hidden_states,
        growth_rates=growth_rates,
        interaction_matrix=interaction_matrix,
        observed_names=[f"可见物种{i + 1}" for i in range(num_observed)],
        hidden_names=[f"隐藏物种{i + 1}" for i in range(num_hidden)],
        timestamps=[str(index) for index in range(total_steps)],
        dt=dt,
    )

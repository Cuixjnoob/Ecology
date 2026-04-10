from __future__ import annotations

import torch


def range_penalty(
    predictions: torch.Tensor,
    min_values: torch.Tensor,
    max_values: torch.Tensor,
    margin: float = 0.0,
) -> torch.Tensor:
    lower = min_values.unsqueeze(0).unsqueeze(0) - margin
    upper = max_values.unsqueeze(0).unsqueeze(0) + margin
    above = torch.relu(predictions - upper).square()
    below = torch.relu(lower - predictions).square()
    return (above + below).mean()


def smoothness_penalty(deltas: torch.Tensor) -> torch.Tensor:
    if deltas.shape[1] < 2:
        return deltas.new_tensor(0.0)
    accelerations = deltas[:, 1:, :] - deltas[:, :-1, :]
    return accelerations.square().mean()


def sparsity_penalty(gate_history: torch.Tensor) -> torch.Tensor:
    if gate_history.numel() == 0:
        return gate_history.new_tensor(0.0)
    return gate_history.mean()


def metabolic_prior_loss(
    predicted_activity: torch.Tensor,
    target_activity: torch.Tensor | None = None,
) -> torch.Tensor:
    if target_activity is None:
        return predicted_activity.new_tensor(0.0)
    return (predicted_activity - target_activity.detach()).square().mean()


def direct_latent_balance_penalty(
    direct_deltas: torch.Tensor,
    latent_deltas: torch.Tensor,
    target_min_ratio: float = 0.15,
) -> torch.Tensor:
    total_scale = direct_deltas.abs().mean() + latent_deltas.abs().mean()
    if float(total_scale.item()) <= 1e-8:
        return direct_deltas.new_tensor(0.0)
    direct_ratio = direct_deltas.abs().mean() / total_scale
    latent_ratio = latent_deltas.abs().mean() / total_scale
    return (
        torch.relu(direct_deltas.new_tensor(target_min_ratio) - direct_ratio)
        + torch.relu(latent_deltas.new_tensor(target_min_ratio) - latent_ratio)
    )

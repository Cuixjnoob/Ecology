from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F


def build_rollout_weights(
    horizon: int,
    increasing: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    if horizon <= 0:
        raise ValueError("horizon must be positive")
    if increasing:
        weights = torch.linspace(1.0, float(horizon), horizon, device=device)
    else:
        weights = torch.ones(horizon, device=device)
    return weights / weights.sum()


def rollout_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    loss_type: str = "mse",
) -> torch.Tensor:
    if predictions.shape != targets.shape:
        raise ValueError("predictions and targets must have the same shape")
    horizon = predictions.shape[1]
    if weights is None:
        weights = build_rollout_weights(horizon=horizon, device=predictions.device)
    if loss_type not in {"mse", "huber"}:
        raise ValueError(f"Unsupported rollout loss type: {loss_type}")

    step_losses = []
    for step_index in range(horizon):
        if loss_type == "mse":
            value = F.mse_loss(predictions[:, step_index], targets[:, step_index], reduction="mean")
        else:
            value = F.smooth_l1_loss(
                predictions[:, step_index],
                targets[:, step_index],
                reduction="mean",
            )
        step_losses.append(value)

    stacked = torch.stack(step_losses)
    return torch.sum(stacked * weights)


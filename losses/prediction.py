from __future__ import annotations

import torch
import torch.nn.functional as F


def one_step_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    if loss_type == "mse":
        return F.mse_loss(predictions, targets)
    if loss_type == "huber":
        return F.smooth_l1_loss(predictions, targets)
    raise ValueError(f"Unsupported one-step loss type: {loss_type}")


from __future__ import annotations

from typing import Dict

import torch


def _flatten_rollout(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.reshape(-1, tensor.shape[-1])


def _pearson_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = _flatten_rollout(predictions)
    true = _flatten_rollout(targets)
    pred_centered = pred - pred.mean(dim=0, keepdim=True)
    true_centered = true - true.mean(dim=0, keepdim=True)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=0) * true_centered.square().sum(dim=0)
    ).clamp_min(1e-8)
    return (pred_centered * true_centered).sum(dim=0) / denominator


def _rank_columns(values: torch.Tensor) -> torch.Tensor:
    order = torch.argsort(values, dim=0)
    ranks = torch.zeros_like(values, dtype=torch.float32)
    rank_values = torch.arange(values.shape[0], device=values.device, dtype=torch.float32)
    expanded_ranks = rank_values.unsqueeze(1).expand_as(order)
    ranks.scatter_(0, order, expanded_ranks)
    return ranks


def _spearman_correlation(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    pred = _flatten_rollout(predictions)
    true = _flatten_rollout(targets)
    return _pearson_correlation(_rank_columns(pred), _rank_columns(true))


def compute_rollout_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    enforce_nonnegative: bool = False,
) -> Dict[str, float | list[float]]:
    errors = predictions - targets
    rmse = torch.sqrt(errors.square().mean()).item()
    mae = errors.abs().mean().item()
    drift = errors.abs().sum(dim=1).mean().item()

    pearson = _pearson_correlation(predictions, targets)
    spearman = _spearman_correlation(predictions, targets)

    invalid_mask = ~torch.isfinite(predictions).all(dim=(1, 2))
    if enforce_nonnegative:
        invalid_mask = invalid_mask | (predictions < 0).any(dim=(1, 2))
    stability_failure_rate = invalid_mask.float().mean().item()

    return {
        "rmse": rmse,
        "mae": mae,
        "drift": drift,
        "pearson_mean": pearson.mean().item(),
        "spearman_mean": spearman.mean().item(),
        "pearson_per_species": pearson.tolist(),
        "spearman_per_species": spearman.tolist(),
        "stability_failure_rate": stability_failure_rate,
    }


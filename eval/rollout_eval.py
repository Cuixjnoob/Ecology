from __future__ import annotations

from typing import Dict, Iterable, List

import torch

from data.transforms import LogZScoreTransform
from eval.metrics import compute_rollout_metrics


@torch.no_grad()
def evaluate_rollout_model(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
    horizons: Iterable[int],
    transform: LogZScoreTransform | None = None,
) -> Dict[str, object]:
    model.eval()
    sorted_horizons = sorted(set(int(horizon) for horizon in horizons))
    if not sorted_horizons:
        raise ValueError("At least one evaluation horizon is required.")
    max_horizon = max(sorted_horizons)

    transformed_predictions: List[torch.Tensor] = []
    transformed_targets: List[torch.Tensor] = []
    raw_predictions: List[torch.Tensor] = []
    raw_targets: List[torch.Tensor] = []
    hidden_activity: List[torch.Tensor] = []
    gate_history: List[torch.Tensor] = []
    direct_deltas: List[torch.Tensor] = []
    latent_deltas: List[torch.Tensor] = []

    for batch in data_loader:
        history = batch["history"].to(device)
        future = batch["future"].to(device)
        history_u = batch["history_u"].to(device)
        future_u = batch["future_u"].to(device)

        outputs = model(
            history_x=history,
            history_u=history_u,
            future_u=future_u[:, :max_horizon],
            rollout_horizon=max_horizon,
            teacher_forcing_targets=None,
            teacher_forcing_ratio=0.0,
        )
        predictions = outputs["predictions"][:, :max_horizon]
        targets = future[:, :max_horizon]

        transformed_predictions.append(predictions.cpu())
        transformed_targets.append(targets.cpu())
        hidden_activity.append(outputs["hidden_activity"][:, :max_horizon].cpu())
        gate_history.append(outputs["gate_history"][:, :max_horizon].cpu())
        direct_deltas.append(outputs["direct_deltas"][:, :max_horizon].cpu())
        latent_deltas.append(outputs["latent_deltas"][:, :max_horizon].cpu())

        if transform is not None:
            batch_size, horizon, num_observed = predictions.shape
            pred_raw = transform.inverse_transform(predictions.reshape(-1, num_observed).cpu())
            true_raw = transform.inverse_transform(targets.reshape(-1, num_observed).cpu())
            raw_predictions.append(pred_raw.reshape(batch_size, horizon, num_observed))
            raw_targets.append(true_raw.reshape(batch_size, horizon, num_observed))

    if not transformed_predictions:
        raise RuntimeError("Evaluation loader did not yield any batches.")

    transformed_predictions_tensor = torch.cat(transformed_predictions, dim=0)
    transformed_targets_tensor = torch.cat(transformed_targets, dim=0)
    gate_tensor = torch.cat(gate_history, dim=0)
    hidden_activity_tensor = torch.cat(hidden_activity, dim=0)
    direct_delta_tensor = torch.cat(direct_deltas, dim=0)
    latent_delta_tensor = torch.cat(latent_deltas, dim=0)
    total_delta_scale = (
        direct_delta_tensor.abs() + latent_delta_tensor.abs()
    ).clamp_min(1e-8)

    results: Dict[str, object] = {
        "transformed": {},
        "diagnostics": {
            "mean_hidden_activity": hidden_activity_tensor.mean().item(),
            "mean_gate_strength": gate_tensor.mean().item(),
            "mean_direct_contribution_ratio": (
                direct_delta_tensor.abs() / total_delta_scale
            ).mean().item(),
            "mean_latent_contribution_ratio": (
                latent_delta_tensor.abs() / total_delta_scale
            ).mean().item(),
        },
    }

    for horizon in sorted_horizons:
        transformed_metrics = compute_rollout_metrics(
            transformed_predictions_tensor[:, :horizon],
            transformed_targets_tensor[:, :horizon],
            enforce_nonnegative=False,
        )
        results["transformed"][str(horizon)] = transformed_metrics

    if raw_predictions and raw_targets:
        raw_predictions_tensor = torch.cat(raw_predictions, dim=0)
        raw_targets_tensor = torch.cat(raw_targets, dim=0)
        results["raw"] = {}
        for horizon in sorted_horizons:
            raw_metrics = compute_rollout_metrics(
                raw_predictions_tensor[:, :horizon],
                raw_targets_tensor[:, :horizon],
                enforce_nonnegative=True,
            )
            results["raw"][str(horizon)] = raw_metrics

    results["artifacts"] = {
        "predictions_transformed": transformed_predictions_tensor,
        "targets_transformed": transformed_targets_tensor,
        "gate_history": gate_tensor,
        "hidden_activity": hidden_activity_tensor,
        "direct_deltas": direct_delta_tensor,
        "latent_deltas": latent_delta_tensor,
    }
    if raw_predictions and raw_targets:
        results["artifacts"]["predictions_raw"] = torch.cat(raw_predictions, dim=0)
        results["artifacts"]["targets_raw"] = torch.cat(raw_targets, dim=0)

    return results

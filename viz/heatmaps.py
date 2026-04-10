from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence

import matplotlib.pyplot as plt
import torch


def plot_horizon_error_curve(
    metrics_by_model: Dict[str, Dict[str, Dict[str, float]]],
    output_path: str | Path,
    metric_name: str = "rmse",
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots(figsize=(8, 5))
    for model_name, horizon_metrics in metrics_by_model.items():
        horizons = sorted(int(horizon) for horizon in horizon_metrics)
        values = [horizon_metrics[str(horizon)][metric_name] for horizon in horizons]
        axis.plot(horizons, values, marker="o", linewidth=2, label=model_name)

    axis.set_xlabel("rollout horizon")
    axis.set_ylabel(metric_name.upper())
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


def plot_species_horizon_heatmap(
    absolute_errors: torch.Tensor,
    species_names: Sequence[str],
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    heatmap = absolute_errors.mean(dim=0).t().cpu()
    figure, axis = plt.subplots(figsize=(10, max(4, 0.5 * len(species_names))))
    image = axis.imshow(heatmap, aspect="auto", cmap="magma")
    axis.set_xlabel("rollout horizon")
    axis.set_ylabel("species")
    axis.set_yticks(range(len(species_names)))
    axis.set_yticklabels(species_names)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


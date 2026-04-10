from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch


def plot_rollout_vs_truth(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    species_names: Sequence[str],
    output_path: str | Path,
    max_species: int = 3,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    num_species = min(len(species_names), predictions.shape[-1], max_species)
    figure, axes = plt.subplots(num_species, 1, figsize=(10, 3 * num_species), squeeze=False)

    time_axis = list(range(predictions.shape[1]))
    for species_index in range(num_species):
        axis = axes[species_index, 0]
        axis.plot(time_axis, targets[0, :, species_index].cpu().tolist(), label="truth", linewidth=2)
        axis.plot(time_axis, predictions[0, :, species_index].cpu().tolist(), label="prediction", linewidth=2)
        axis.set_title(species_names[species_index])
        axis.set_xlabel("horizon")
        axis.set_ylabel("abundance")
        axis.grid(alpha=0.3)
        axis.legend()

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


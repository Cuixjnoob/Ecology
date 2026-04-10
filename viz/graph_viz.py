from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import torch


def plot_average_gate_heatmap(
    gate_history: torch.Tensor,
    num_observed: int,
    output_path: str | Path,
) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    average_gates = gate_history.mean(dim=(0, 1, 2)).cpu()
    figure, axes = plt.subplots(1, 2, figsize=(10, 4))

    observed_block = average_gates[:num_observed, :num_observed]
    image_0 = axes[0].imshow(observed_block, cmap="viridis")
    axes[0].set_title("observed -> observed")
    figure.colorbar(image_0, ax=axes[0], fraction=0.046, pad=0.04)

    if average_gates.shape[0] > num_observed:
        hidden_block = average_gates[num_observed:, :num_observed]
        image_1 = axes[1].imshow(hidden_block, aspect="auto", cmap="viridis")
        axes[1].set_title("hidden -> observed")
        figure.colorbar(image_1, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)


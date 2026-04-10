from __future__ import annotations

from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import torch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401


def setup_chinese_plotting() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti SC",
        "Songti SC",
        "Arial Unicode MS",
        "SimHei",
        "Noto Sans CJK SC",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _ensure_parent(output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    setup_chinese_plotting()
    return output


def plot_lv_system_overview(
    observed_states: torch.Tensor,
    hidden_states: torch.Tensor,
    observed_names: Sequence[str],
    hidden_names: Sequence[str],
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    time_axis = list(range(observed_states.shape[0]))
    for index, name in enumerate(observed_names[: min(len(observed_names), 5)]):
        axes[0].plot(time_axis, observed_states[:, index].cpu().tolist(), linewidth=1.6, label=name)
    axes[0].set_title("真实 LV 系统中的部分可见物种轨迹")
    axes[0].set_ylabel("丰度")
    axes[0].grid(alpha=0.25)
    axes[0].legend(ncol=3, fontsize=9)

    for index, name in enumerate(hidden_names):
        axes[1].plot(time_axis, hidden_states[:, index].cpu().tolist(), linewidth=2.0, label=name)
    axes[1].set_title("真实 LV 系统中的隐藏物种轨迹（仅用于生成真值）")
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("丰度")
    axes[1].grid(alpha=0.25)
    axes[1].legend()

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_lv_interaction_matrix(
    interaction_matrix: torch.Tensor,
    num_observed: int,
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    figure, axis = plt.subplots(figsize=(8, 6))
    image = axis.imshow(interaction_matrix.cpu(), cmap="coolwarm", aspect="auto")
    axis.axvline(num_observed - 0.5, color="black", linestyle="--", linewidth=1.2)
    axis.axhline(num_observed - 0.5, color="black", linestyle="--", linewidth=1.2)
    axis.set_title("真实 LV 相互作用矩阵（虚线右下角为隐藏物种区块）")
    axis.set_xlabel("作用来源物种")
    axis.set_ylabel("被作用物种")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_takens_delay_stack(
    observed_series: torch.Tensor,
    species_name: str,
    tau: int,
    m: int,
    output_path: str | Path,
    anchor_index: int | None = None,
) -> None:
    output = _ensure_parent(output_path)
    total_steps = observed_series.shape[0]
    anchor = anchor_index if anchor_index is not None else min(total_steps - 1, (m - 1) * tau + 40)
    indices = [anchor - offset * tau for offset in range(m)]
    values = [float(observed_series[index].item()) for index in indices]

    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=False)
    time_axis = list(range(total_steps))

    axes[0].plot(time_axis, observed_series.cpu().tolist(), color="#0B6E4F", linewidth=2.0)
    axes[0].scatter(indices, values, color="#C73E1D", s=48, zorder=3)
    for offset, index in enumerate(indices):
        axes[0].annotate(
            f"x(t-{offset * tau})",
            (index, float(observed_series[index].item())),
            textcoords="offset points",
            xytext=(0, 8),
            ha="center",
            fontsize=9,
        )
    axes[0].set_title(f"{species_name} 的原始时间序列与 Takens 取样点")
    axes[0].set_ylabel("丰度")
    axes[0].grid(alpha=0.25)

    bar_positions = list(range(m))
    axes[1].bar(bar_positions, values, color="#3A86FF")
    axes[1].set_xticks(bar_positions)
    axes[1].set_xticklabels([f"x(t-{offset * tau})" for offset in range(m)])
    axes[1].set_title(f"{species_name} 的 Takens 延迟向量（tau={tau}, m={m}）")
    axes[1].set_ylabel("取值")
    axes[1].grid(alpha=0.25, axis="y")

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_takens_embedding_scatter(
    observed_series: torch.Tensor,
    species_name: str,
    tau: int,
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    max_offset = 2 * tau
    if observed_series.shape[0] <= max_offset:
        return

    x_t = observed_series[max_offset:]
    x_tau = observed_series[max_offset - tau : -tau]
    x_2tau = observed_series[: -max_offset]

    figure = plt.figure(figsize=(12, 5))
    axis_2d = figure.add_subplot(1, 2, 1)
    axis_3d = figure.add_subplot(1, 2, 2, projection="3d")

    axis_2d.scatter(x_t.cpu(), x_tau.cpu(), s=10, alpha=0.65, color="#8338EC")
    axis_2d.set_title(f"{species_name} 的二维延迟嵌入")
    axis_2d.set_xlabel("x(t)")
    axis_2d.set_ylabel(f"x(t-{tau})")
    axis_2d.grid(alpha=0.2)

    axis_3d.scatter(x_t.cpu(), x_tau.cpu(), x_2tau.cpu(), s=8, alpha=0.55, color="#FF006E")
    axis_3d.set_title(f"{species_name} 的三维延迟嵌入")
    axis_3d.set_xlabel("x(t)")
    axis_3d.set_ylabel(f"x(t-{tau})")
    axis_3d.set_zlabel(f"x(t-{2 * tau})")

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_rollout_comparison(
    history_truth: torch.Tensor,
    future_truth: torch.Tensor,
    future_prediction: torch.Tensor,
    species_names: Sequence[str],
    output_path: str | Path,
    max_species: int = 4,
) -> None:
    output = _ensure_parent(output_path)
    num_species = min(max_species, len(species_names), future_truth.shape[-1])
    figure, axes = plt.subplots(num_species, 1, figsize=(12, 3.2 * num_species), squeeze=False)

    history_length = history_truth.shape[0]
    future_length = future_truth.shape[0]
    history_axis = list(range(history_length))
    future_axis = list(range(history_length, history_length + future_length))

    for species_index in range(num_species):
        axis = axes[species_index, 0]
        axis.plot(history_axis, history_truth[:, species_index].cpu().tolist(), color="#6C757D", linewidth=2.0, label="真实历史")
        axis.plot(future_axis, future_truth[:, species_index].cpu().tolist(), color="#198754", linewidth=2.0, label="真实未来")
        axis.plot(future_axis, future_prediction[:, species_index].cpu().tolist(), color="#D62828", linewidth=2.0, linestyle="--", label="模型 rollout")
        axis.axvline(history_length - 0.5, color="black", linestyle=":", linewidth=1.0)
        axis.set_title(f"{species_names[species_index]}：正向模拟对比")
        axis.set_ylabel("丰度")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right", ncol=3, fontsize=9)

    axes[-1, 0].set_xlabel("时间步")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_species_error_heatmap(
    absolute_errors: torch.Tensor,
    species_names: Sequence[str],
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    heatmap = absolute_errors.t().cpu()
    figure, axis = plt.subplots(figsize=(10, max(4.5, 0.45 * len(species_names))))
    image = axis.imshow(heatmap, aspect="auto", cmap="magma")
    axis.set_title("逐物种正向模拟绝对误差热图")
    axis.set_xlabel("rollout 步数")
    axis.set_ylabel("物种")
    axis.set_yticks(range(len(species_names)))
    axis.set_yticklabels(species_names)
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_gate_heatmap(
    gate_history: torch.Tensor,
    num_observed: int,
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    if gate_history.ndim < 2:
        raise ValueError("gate_history must have at least 2 dimensions.")
    reduction_dims = tuple(range(gate_history.ndim - 2))
    average_gate = gate_history.mean(dim=reduction_dims).cpu()
    figure, axes = plt.subplots(1, 2, figsize=(11, 4))

    image_obs = axes[0].imshow(average_gate[:num_observed, :num_observed], cmap="viridis", aspect="auto")
    axes[0].set_title("平均门控强度：可见 -> 可见")
    axes[0].set_xlabel("目标物种")
    axes[0].set_ylabel("来源物种")
    figure.colorbar(image_obs, ax=axes[0], fraction=0.046, pad=0.04)

    if average_gate.shape[0] > num_observed:
        image_hidden = axes[1].imshow(average_gate[num_observed:, :num_observed], cmap="viridis", aspect="auto")
        axes[1].set_title("平均门控强度：潜在 -> 可见")
        axes[1].set_xlabel("目标可见物种")
        axes[1].set_ylabel("潜在节点")
        figure.colorbar(image_hidden, ax=axes[1], fraction=0.046, pad=0.04)
    else:
        axes[1].axis("off")

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_hidden_reference(
    hidden_truth: torch.Tensor,
    hidden_names: Sequence[str],
    latent_activity: torch.Tensor,
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    figure, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=False)

    time_axis = list(range(hidden_truth.shape[0]))
    for index, name in enumerate(hidden_names):
        axes[0].plot(time_axis, hidden_truth[:, index].cpu().tolist(), linewidth=2.0, label=name)
    axes[0].set_title("真实隐藏物种轨迹（仅用于参考，不参与训练）")
    axes[0].set_ylabel("丰度")
    axes[0].grid(alpha=0.25)
    axes[0].legend()

    activity_axis = list(range(latent_activity.shape[0]))
    axes[1].plot(activity_axis, latent_activity.cpu().tolist(), color="#9D4EDD", linewidth=2.2)
    axes[1].set_title("模型潜在节点活性强度")
    axes[1].set_xlabel("时间步")
    axes[1].set_ylabel("活性范数")
    axes[1].grid(alpha=0.25)

    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_hidden_truth_vs_prediction(
    hidden_truth: torch.Tensor,
    hidden_prediction: torch.Tensor,
    hidden_names: Sequence[str],
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    num_hidden = min(hidden_truth.shape[-1], hidden_prediction.shape[-1], len(hidden_names))
    figure, axes = plt.subplots(num_hidden, 1, figsize=(12, 3.2 * num_hidden), squeeze=False)

    time_axis = list(range(hidden_truth.shape[0]))
    for hidden_index in range(num_hidden):
        axis = axes[hidden_index, 0]
        axis.plot(time_axis, hidden_truth[:, hidden_index].cpu().tolist(), linewidth=2.2, color="#198754", label="真实隐藏物种")
        axis.plot(time_axis, hidden_prediction[:, hidden_index].cpu().tolist(), linewidth=2.2, color="#D62828", linestyle="--", label="模型推断隐藏物种")
        axis.set_title(f"{hidden_names[hidden_index]}：真实轨迹 vs 推断轨迹")
        axis.set_ylabel("丰度")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper right")

    axes[-1, 0].set_xlabel("时间步")
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_hidden_scatter(
    hidden_truth: torch.Tensor,
    hidden_prediction: torch.Tensor,
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    figure, axis = plt.subplots(figsize=(6, 6))
    axis.scatter(hidden_truth.cpu(), hidden_prediction.cpu(), alpha=0.7, color="#3A86FF", s=18)
    min_value = float(min(hidden_truth.min().item(), hidden_prediction.min().item()))
    max_value = float(max(hidden_truth.max().item(), hidden_prediction.max().item()))
    axis.plot([min_value, max_value], [min_value, max_value], linestyle="--", color="#D62828")
    axis.set_title("真实隐藏物种 vs 模型预测隐藏物种")
    axis.set_xlabel("真实值")
    axis.set_ylabel("预测值")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)


def plot_hidden_physics_comparison(
    hidden_next_true: torch.Tensor,
    hidden_next_physics: torch.Tensor,
    output_path: str | Path,
) -> None:
    output = _ensure_parent(output_path)
    figure, axis = plt.subplots(figsize=(11, 4))
    time_axis = list(range(hidden_next_true.shape[0]))
    axis.plot(time_axis, hidden_next_true.squeeze(-1).cpu().tolist(), linewidth=2.0, color="#198754", label="真实下一时刻隐藏物种")
    axis.plot(time_axis, hidden_next_physics.squeeze(-1).cpu().tolist(), linewidth=2.0, linestyle="--", color="#D62828", label="基于真实LV方程的一步推演")
    axis.set_title("隐藏物种的一步物理一致性对比")
    axis.set_xlabel("测试窗口索引")
    axis.set_ylabel("丰度")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, dpi=180)
    plt.close(figure)

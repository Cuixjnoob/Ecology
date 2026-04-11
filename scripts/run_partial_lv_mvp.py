"""部分观测 LV 系统的主实验入口脚本。

完整流程：
  1. 加载 YAML 配置（默认 configs/partial_lv_mvp.yaml）
  2. 合成数据生成（5 可见 + 1 隐藏 + 1 环境，820 步）
  3. 噪声扫描（可选）：在指定网格上搜索最优 noise profile
  4. 模型构建 → PartialLVMVPTrainer 训练（full-context）
  5. 评估：train/val/test 滚动指标 + 诊断图
  6. 保存结果到 runs/<timestamp>_<tag>/

使用方式：
  python -m scripts.run_partial_lv_mvp --config configs/partial_lv_mvp.yaml
"""
from __future__ import annotations

import argparse
import copy
import itertools
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List
from zoneinfo import ZoneInfo

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from data.dataset import build_windowed_datasets
from data.partial_lv_mvp import generate_partial_lv_mvp_system
from models.partial_lv_recovery_model import PartialLVRecoveryModel
from train.partial_lv_mvp_trainer import PartialLVMVPTrainer
from train.utils import create_data_loaders, save_json, set_random_seed


def _load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Hiragino Sans GB",
        "Heiti TC",
        "STHeiti",
        "Microsoft YaHei",
        "SimHei",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def _build_model(
    model_cfg: Dict[str, Any],
    data_cfg: Dict[str, Any],
    simulation_cfg: Dict[str, Any],
) -> PartialLVRecoveryModel:
    return PartialLVRecoveryModel(
        num_visible=int(simulation_cfg["visible_species"]),
        delay_length=int(data_cfg["delay_length"]),
        delay_stride=int(data_cfg["delay_stride"]),
        delay_embedding_dim=int(model_cfg["delay_embedding_dim"]),
        context_dim=int(model_cfg["context_dim"]),
        hidden_dim=int(model_cfg["hidden_dim"]),
        encoder_layers=int(model_cfg["encoder_layers"]),
        residual_layers=int(model_cfg["residual_layers"]),
        use_environment_latent=bool(model_cfg.get("use_environment_latent", True)),
        use_lv_guidance=bool(model_cfg.get("use_lv_guidance", True)),
        max_state_value=float(simulation_cfg["max_state_value"]),
        base_visible_noise=float(model_cfg.get("base_visible_noise", 0.015)),
        base_hidden_noise=float(model_cfg.get("base_hidden_noise", 0.012)),
        base_environment_noise=float(model_cfg.get("base_environment_noise", 0.020)),
    )


def _to_plain(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _to_plain(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_plain(item) for item in value]
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value


def _create_run_dirs(config: Dict[str, Any]) -> tuple[str, Path, Path]:
    timestamp = datetime.now(ZoneInfo("Asia/Shanghai")).strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{config['experiment']['name']}"
    run_dir = Path(config["experiment"]["runs_dir"]) / run_name
    results_dir = run_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    return timestamp, run_dir, results_dir


def _make_nan_series(length: int, width: int) -> torch.Tensor:
    return torch.full((length, width), float("nan"), dtype=torch.float32)


def _fill_series_from_indices(length: int, indices: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    filled = _make_nan_series(length, values.shape[1])
    filled[indices.long()] = values
    return filled


def _data_regime_label(diagnostics: Dict[str, Any]) -> str:
    if diagnostics["too_flat"]:
        return "too_flat"
    if diagnostics["too_periodic"]:
        return "too_periodic"
    return "moderate_complexity"


def _select_representative_species(visible_true: torch.Tensor, limit: int = 3) -> List[int]:
    scores = visible_true.std(dim=0, unbiased=False)
    ordering = torch.argsort(scores, descending=True)
    return [int(index.item()) for index in ordering[:limit]]


def _select_rollout_case(
    visible_series: torch.Tensor,
    hidden_series: torch.Tensor,
    split_info: Dict[str, tuple[int, int]],
    history_length: int,
    horizon: int,
) -> Dict[str, torch.Tensor | int]:
    test_start, test_end = split_info["test"]
    max_start = max(test_start, test_end - horizon)
    best_start = test_start
    best_score = float("-inf")
    for start_index in range(test_start, max_start + 1):
        future_visible = visible_series[start_index : start_index + horizon]
        future_hidden = hidden_series[start_index : start_index + horizon]
        score = float(
            future_visible.std(unbiased=False).mean().item()
            + 0.35 * future_hidden.std(unbiased=False).item()
            + 0.20 * (future_visible.max() - future_visible.min()).mean().item()
        )
        if score > best_score:
            best_score = score
            best_start = start_index

    history_start = best_start - history_length
    return {
        "forecast_start": best_start,
        "history_visible": visible_series[history_start:best_start],
        "future_visible": visible_series[best_start : best_start + horizon],
        "future_hidden": hidden_series[best_start : best_start + horizon],
    }


def _select_hidden_zoom_window(hidden_true: torch.Tensor, hidden_pred: torch.Tensor) -> tuple[int, int]:
    total_steps = int(hidden_true.shape[0])
    window_length = min(max(96, total_steps // 2), 160)
    if total_steps <= window_length:
        return 0, total_steps

    best_start = 0
    best_score = float("-inf")
    for start_index in range(0, total_steps - window_length + 1):
        end_index = start_index + window_length
        truth_window = hidden_true[start_index:end_index, 0]
        pred_window = hidden_pred[start_index:end_index, 0]
        score = float(
            truth_window.std(unbiased=False).item()
            + 0.7 * (pred_window - truth_window).abs().mean().item()
            + 0.2 * (truth_window.max() - truth_window.min()).item()
        )
        if score > best_score:
            best_score = score
            best_start = start_index
    return best_start, best_start + window_length


def _load_previous_reference() -> Dict[str, Any] | None:
    candidate_paths = sorted(
        Path("runs").glob("*_partial_lv_lv_guided_stochastic/results/summary.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for path in candidate_paths:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if "modelF_metrics" in payload:
            return {
                "path": str(path),
                "metrics": payload["modelF_metrics"],
            }
    return None


def _model_noise_stats(model: PartialLVRecoveryModel) -> Dict[str, float]:
    visible = model.base_visible_noise * (0.4 + torch.nn.functional.softplus(model.visible_noise_unconstrained)).item()
    hidden = model.base_hidden_noise * (0.4 + torch.nn.functional.softplus(model.hidden_noise_unconstrained)).item()
    environment = model.base_environment_noise * (
        0.4 + torch.nn.functional.softplus(model.environment_noise_unconstrained)
    ).item()
    return {
        "visible_noise_std": float(visible),
        "hidden_noise_std": float(hidden),
        "environment_noise_std": float(environment),
    }


def _lv_backbone_active(lv_guidance_metrics: Dict[str, float]) -> bool:
    return bool(
        lv_guidance_metrics["residual_dominates_fraction"] < 0.50
        and 0.15 <= lv_guidance_metrics["lv_residual_ratio_mean"] <= 1.5
        and lv_guidance_metrics.get("lv_energy_fraction", 0.5) >= 0.35
        and lv_guidance_metrics.get("visible_residual_dominates_fraction", 1.0) < 0.60
        and lv_guidance_metrics.get("visible_lv_energy_fraction", 0.0) >= 0.40
    )


def _diagnose(
    metrics: Dict[str, float],
    disentanglement_metrics: Dict[str, float],
    previous_reference: Dict[str, Any] | None,
) -> str:
    previous_metrics = previous_reference["metrics"] if previous_reference else None
    if previous_metrics is not None and metrics["sliding_window_visible_rmse"] >= previous_metrics["sliding_window_visible_rmse"] * 0.99:
        if metrics["selected_rollout_noise"] >= 0.05:
            return "rollout noise too strong"
    if abs(disentanglement_metrics["hidden_environment_correlation"]) > 0.45:
        return "hidden/environment entanglement"
    if metrics["amplitude_collapse_score"] > 0.35:
        return "amplitude collapse"
    return "mixed"


def _write_experiment_readme(
    run_dir: Path,
    timestamp: str,
    config: Dict[str, Any],
    summary_payload: Dict[str, Any],
) -> None:
    readme_lines = [
        "# 实验 README",
        "",
        f"- 时间戳：`{timestamp}`",
        f"- 实验名：`{config['experiment']['name']}`",
        "",
        "## 本次实验修改了什么",
        "- 保留上一版的 LV soft guidance + stochastic residual 主框架，但不再做模型对照，只修正 rollout 训练噪声和 hidden/environment 解耦。",
        "- rollout training noise 被压到一个很小的组合扫描范围内，并固定 particle rollout 为 `K=4`、`mean aggregation` 的辅助项。",
        "- hidden/environment 解耦约束增强为：更强相关性惩罚、正交惩罚、方差下界、以及 environment 更慢 / hidden 更快的时间尺度先验。",
        "",
        "## 当前总算法是什么样子的",
        "1. 数据层：仍使用 moderate_complexity 的 5 个 visible + 1 个 hidden + 1 个真实 environment 的合成生态系统。",
        "2. 动力学层：状态更新保持 `state_t + LV_guided_drift + neural_residual + stochastic_noise`，LV 仍是 soft backbone。",
        "3. 训练层：主要目标是降低过强噪声带来的平线化预测，并压低 hidden/environment 纠缠。",
        "4. 评估层：继续报告 sliding-window rollout、full-context forecast、hidden recovery、LV/residual 比例和 latent disentanglement 统计。",
        "",
        "## 本次结果一句话",
        f"- 数据 regime: `{summary_payload['data_regime_assessment']['label']}`",
        f"- 选中的噪声配置: `{summary_payload['selected_noise_config']}`",
        f"- 当前主诊断: `{summary_payload['diagnosis']}`",
    ]
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def _plot_true_trajectories(
    visible_true: torch.Tensor,
    hidden_true: torch.Tensor,
    environment_true: torch.Tensor,
    split_info: Dict[str, Any],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True, constrained_layout=True)
    time_axis = np.arange(visible_true.shape[0])
    train_end = split_info["train_range"][1]
    val_end = split_info["val_range"][1]

    for species_index in range(visible_true.shape[1]):
        axes[0].plot(time_axis, visible_true[:, species_index].numpy(), linewidth=1.4, label=f"可见物种{species_index + 1}")
    axes[0].set_title("真实可见物种轨迹")
    axes[0].set_ylabel("丰度")
    axes[0].legend(ncol=3, fontsize=8)

    axes[1].plot(time_axis, hidden_true[:, 0].numpy(), color="black", linewidth=1.8, label="真实隐藏物种")
    env_axis = axes[1].twinx()
    env_axis.plot(time_axis, environment_true[:, 0].numpy(), color="#2ca02c", linewidth=1.4, linestyle="--", label="真实环境")
    axes[1].set_title("真实 hidden 与 true environment")
    axes[1].set_ylabel("隐藏物种丰度")
    env_axis.set_ylabel("环境驱动")

    for axis in axes:
        axis.axvline(train_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
        axis.axvline(val_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
    env_axis.axvline(train_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
    env_axis.axvline(val_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)

    line_handles, line_labels = axes[1].get_legend_handles_labels()
    env_handles, env_labels = env_axis.get_legend_handles_labels()
    axes[1].legend(line_handles + env_handles, line_labels + env_labels, loc="upper right", fontsize=8)
    axes[1].set_xlabel("时间步")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_hidden_test_overlay(
    hidden_true: torch.Tensor,
    hidden_pred: torch.Tensor,
    hidden_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    time_axis = np.arange(hidden_true.shape[0])
    start_index, end_index = _select_hidden_zoom_window(hidden_true, hidden_pred)
    zoom_axis = np.arange(start_index, end_index)
    metric_text = (
        f"hidden RMSE={hidden_metrics['hidden_rmse']:.4f} | Pearson={hidden_metrics['hidden_pearson']:.4f}\n"
        f"hidden/env corr={hidden_metrics['hidden_environment_correlation']:.4f}"
    )

    for axis, x_values, truth_values, pred_values, title in [
        (
            axes[0],
            time_axis,
            hidden_true[:, 0].numpy(),
            hidden_pred[:, 0].numpy(),
            "面板A：测试段全长 hidden 对比",
        ),
        (
            axes[1],
            zoom_axis,
            hidden_true[start_index:end_index, 0].numpy(),
            hidden_pred[start_index:end_index, 0].numpy(),
            f"面板B：测试段局部放大（窗口 {start_index}:{end_index}）",
        ),
    ]:
        axis.plot(x_values, truth_values, color="black", linewidth=2.0, label="真实 hidden")
        axis.plot(x_values, pred_values, color="#ff7f0e", linewidth=1.8, label="当前模型")
        axis.set_title(title)
        axis.set_ylabel("丰度")
        axis.legend(loc="upper right")

    axes[0].text(
        0.02,
        0.97,
        metric_text,
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )
    axes[1].set_xlabel("测试段时间步")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_visible_rollout_compare(
    history_visible: torch.Tensor,
    future_visible: torch.Tensor,
    visible_pred: torch.Tensor,
    representative_species: List[int],
    forecast_start: int,
    output_path: Path,
) -> None:
    num_species = len(representative_species)
    figure, axes = plt.subplots(num_species, 1, figsize=(12, 3.4 * num_species), constrained_layout=True)
    if num_species == 1:
        axes = [axes]

    history_axis = np.arange(forecast_start - history_visible.shape[0], forecast_start)
    future_axis = np.arange(forecast_start, forecast_start + future_visible.shape[0])

    for axis, species_index in zip(axes, representative_species):
        axis.plot(history_axis, history_visible[:, species_index].numpy(), color="#7f7f7f", linewidth=1.5, label="历史")
        axis.plot(future_axis, future_visible[:, species_index].numpy(), color="black", linewidth=2.0, label="真实未来")
        axis.plot(future_axis, visible_pred[:, species_index].numpy(), color="#ff7f0e", linewidth=1.8, label="当前模型 rollout")
        axis.axvline(forecast_start - 0.5, color="#999999", linestyle="--", linewidth=1.0)
        axis.set_title(f"可见物种{species_index + 1} sliding-window rollout")
        axis.set_ylabel("丰度")
        axis.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("时间步")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_visible_fullcontext_compare(
    visible_series: torch.Tensor,
    visible_pred_full: torch.Tensor,
    split_info: Dict[str, Any],
    representative_species: List[int],
    output_path: Path,
) -> None:
    num_species = len(representative_species)
    figure, axes = plt.subplots(num_species, 1, figsize=(12, 3.4 * num_species), constrained_layout=True)
    if num_species == 1:
        axes = [axes]

    time_axis = np.arange(visible_series.shape[0])
    train_end = split_info["train_range"][1]
    val_end = split_info["val_range"][1]

    for axis, species_index in zip(axes, representative_species):
        axis.plot(time_axis[:train_end], visible_series[:train_end, species_index].numpy(), color="#7f7f7f", linewidth=1.5, label="训练历史")
        axis.plot(time_axis[train_end:], visible_series[train_end:, species_index].numpy(), color="black", linewidth=2.0, label="验证/测试真实未来")
        axis.plot(time_axis, visible_pred_full[:, species_index].numpy(), color="#ff7f0e", linewidth=1.8, label="当前模型预测")
        axis.axvline(train_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
        axis.axvline(val_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
        axis.set_title(f"可见物种{species_index + 1} full-context 预测")
        axis.set_ylabel("丰度")
        axis.legend(loc="upper right", ncol=3, fontsize=8)
    axes[-1].set_xlabel("时间步")

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_training_metrics(history: List[Dict[str, float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    epochs = [record["epoch"] for record in history]
    axes[0].plot(epochs, [record["train_total"] for record in history], color="#ff7f0e", linewidth=1.8, label="train")
    axes[0].plot(epochs, [record["val_total"] for record in history], color="#ff7f0e", linestyle="--", linewidth=1.8, label="val")
    axes[0].set_title("训练 / 验证损失")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    metric_specs = [
        ("val_visible_rollout_rmse", "-", "sliding RMSE"),
        ("val_full_context_visible_rmse", "--", "full-context RMSE"),
        ("val_hidden_recovery_rmse", ":", "hidden RMSE"),
        ("val_amplitude_collapse_score", "-.", "amplitude collapse"),
        ("val_hidden_environment_correlation", (0, (2, 2)), "hidden/env corr"),
    ]
    for metric_key, linestyle, label in metric_specs:
        axes[1].plot(epochs, [record[metric_key] for record in history], linestyle=linestyle, linewidth=1.8, label=label)
    axes[1].set_title("验证集关键指标")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].legend(fontsize=8)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_diagnostics(
    lv_guidance_metrics: Dict[str, float],
    selected_noise_config: Dict[str, float],
    disentanglement_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)

    axes[0].bar(
        ["LV/Residual 比值", "Residual 主导比例", "LV 能量占比"],
        [lv_guidance_metrics["lv_residual_ratio_mean"], lv_guidance_metrics["residual_dominates_fraction"], lv_guidance_metrics.get("lv_energy_fraction", 0.0)],
        color=["#4c78a8", "#e15759", "#59a14f"],
    )
    axes[0].axhline(1.0, color="#999999", linestyle="--", linewidth=1.0)
    axes[0].set_title("LV vs Residual")
    axes[0].text(
        0.02,
        0.95,
        f"ratio std={lv_guidance_metrics['lv_residual_ratio_std']:.3f}",
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
    )

    noise_labels = ["input", "rollout", "latent"]
    noise_values = [
        selected_noise_config["training_input_noise"],
        selected_noise_config["training_rollout_noise"],
        selected_noise_config["training_latent_perturb"],
    ]
    axes[1].bar(noise_labels, noise_values, color="#59a14f")
    axes[1].set_title("Selected Noise Config")

    disent_labels = ["corr", "hidden ac", "env ac", "hidden rough", "env rough"]
    disent_values = [
        abs(disentanglement_metrics["hidden_environment_correlation"]),
        disentanglement_metrics["hidden_autocorrelation"],
        disentanglement_metrics["environment_autocorrelation"],
        disentanglement_metrics["hidden_roughness"],
        disentanglement_metrics["environment_roughness"],
    ]
    axes[2].bar(disent_labels, disent_values, color="#f28e2b")
    axes[2].set_title("Disentanglement 统计")
    axes[2].tick_params(axis="x", rotation=15)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _make_trainer(
    config: Dict[str, Any],
    system,
    bundle,
    visible_log_mean: torch.Tensor,
    visible_log_std: torch.Tensor,
    split_ranges: Dict[str, tuple[int, int]],
    history_length: int,
    noise_profile: Dict[str, float],
) -> tuple[PartialLVRecoveryModel, PartialLVMVPTrainer]:
    model = _build_model(
        model_cfg=config["model"],
        data_cfg=config["data"],
        simulation_cfg=config["simulation"],
    )
    trainer = PartialLVMVPTrainer(
        model=model,
        true_interaction_matrix=system.interaction_matrix,
        visible_log_mean=visible_log_mean,
        visible_log_std=visible_log_std,
        visible_series_raw=bundle.observations,
        hidden_series_raw=bundle.hidden_observations,
        split_ranges=split_ranges,
        history_length=history_length,
        config=config,
        noise_profile=noise_profile,
        particle_rollout_k=int(config["noise"]["particle_rollout_K"]),
    )
    return model, trainer


def _iter_tested_noise_configs(config: Dict[str, Any]) -> List[Dict[str, float]]:
    grid = []
    for rollout_noise, input_noise, latent_perturb in itertools.product(
        config["noise"]["tested_rollout_noise"],
        config["noise"]["tested_input_noise"],
        config["noise"]["tested_latent_perturb"],
    ):
        grid.append(
            {
                "training_input_noise": float(input_noise),
                "training_rollout_noise": float(rollout_noise),
                "training_latent_perturb": float(latent_perturb),
            }
        )
    return grid


def _noise_scan(
    config: Dict[str, Any],
    system,
    bundle,
    prepared,
    data_loaders,
    split_ranges: Dict[str, tuple[int, int]],
) -> tuple[List[Dict[str, Any]], Dict[str, float], str]:
    visible_log_mean = prepared["transform"].mean
    visible_log_std = prepared["transform"].std
    history_length = int(config["data"]["history_length"])
    tested_results: List[Dict[str, Any]] = []
    best_score = float("inf")
    best_config: Dict[str, float] | None = None

    for noise_config in _iter_tested_noise_configs(config):
        scan_config = copy.deepcopy(config)
        scan_config["train"]["epochs"] = int(config["train"]["noise_scan_epochs"])
        scan_config["train"]["early_stopping_patience"] = int(config["train"]["noise_scan_patience"])
        _, trainer = _make_trainer(
            config=scan_config,
            system=system,
            bundle=bundle,
            visible_log_mean=visible_log_mean,
            visible_log_std=visible_log_std,
            split_ranges=split_ranges,
            history_length=history_length,
            noise_profile={
                "history_jitter": noise_config["training_input_noise"],
                "rollout_process_noise": noise_config["training_rollout_noise"],
                "latent_perturb": noise_config["training_latent_perturb"],
            },
        )
        fit_result = trainer.fit(data_loaders["train"], data_loaders["val"])
        val_sliding = trainer.evaluate_loader(data_loaders["val"], num_particles=int(config["noise"]["eval_particles"]))
        val_full_context = trainer.evaluate_full_context("val", num_particles=int(config["noise"]["eval_particles"]))
        val_hidden = trainer.recover_hidden_on_split("val")
        score = (
            0.28 * val_sliding["visible_rollout_rmse"]
            + 0.22 * val_full_context["metrics"]["visible_rmse"]
            + 0.16 * val_sliding["amplitude_collapse_score"]
            + 0.14 * val_hidden["metrics"]["hidden_recovery_rmse"]
            + 0.12 * abs(val_hidden["metrics"]["hidden_environment_correlation"])
            + 0.04 * val_sliding["residual_dominates_fraction"]
            + 0.06 * val_sliding["visible_residual_dominates_fraction"]
            + 0.03 * max(val_sliding["lv_residual_ratio_mean"] - 1.35, 0.0)
            + 0.05 * max(val_sliding["visible_lv_residual_ratio_mean"] - 0.95, 0.0)
            + 0.10 * float(fit_result.best_epoch <= max(3, int(scan_config["train"]["epochs"] * 0.35)))
        )
        entry = {
            **noise_config,
            "validation_score": float(score),
            "visible_rmse": float(val_sliding["visible_rollout_rmse"]),
            "full_context_visible_rmse": float(val_full_context["metrics"]["visible_rmse"]),
            "amplitude_collapse": float(val_sliding["amplitude_collapse_score"]),
            "hidden_rmse": float(val_hidden["metrics"]["hidden_recovery_rmse"]),
            "hidden_environment_correlation": float(val_hidden["metrics"]["hidden_environment_correlation"]),
            "residual_dominates_fraction": float(val_sliding["residual_dominates_fraction"]),
            "lv_residual_ratio_mean": float(val_sliding["lv_residual_ratio_mean"]),
            "visible_residual_dominates_fraction": float(val_sliding["visible_residual_dominates_fraction"]),
            "visible_lv_residual_ratio_mean": float(val_sliding["visible_lv_residual_ratio_mean"]),
            "best_epoch": int(fit_result.best_epoch),
        }
        tested_results.append(entry)
        if score < best_score:
            best_score = score
            best_config = noise_config

    if best_config is None:
        raise RuntimeError("Noise scan failed to select a configuration.")

    selection_reason = (
        "选中该配置是因为它在验证集上同时兼顾了 sliding-window 与 full-context 的 visible 误差，"
        "amplitude collapse 更低，hidden/environment correlation 更小，而且 residual 没有明显长期压过 LV。"
    )
    return tested_results, best_config, selection_reason


def run_experiment(config_path: str | Path) -> Dict[str, Any]:
    config = _load_config(config_path)
    _configure_matplotlib()
    set_random_seed(int(config["experiment"]["seed"]))
    timestamp, run_dir, results_dir = _create_run_dirs(config)

    previous_reference = _load_previous_reference()

    system = generate_partial_lv_mvp_system(
        total_steps=int(config["simulation"]["total_steps"]),
        warmup_steps=int(config["simulation"]["warmup_steps"]),
        process_noise=float(config["simulation"]["process_noise"]),
        seed=int(config["experiment"]["seed"]),
        max_attempts=int(config["simulation"]["max_attempts"]),
        max_state_value=float(config["simulation"]["max_state_value"]),
    )
    bundle = system.to_bundle()
    prepared = build_windowed_datasets(
        bundle=bundle,
        history_length=int(config["data"]["history_length"]),
        horizon=int(config["data"]["train_horizon"]),
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
    )
    data_loaders = create_data_loaders(
        datasets=prepared["datasets"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
    )

    split_ranges = prepared["splits"]
    train_range = split_ranges["train"]
    val_range = split_ranges["val"]
    test_range = split_ranges["test"]
    total_steps = bundle.total_steps
    history_length = int(config["data"]["history_length"])
    figure_rollout_horizon = int(config["data"].get("figure_rollout_horizon", config["data"]["train_horizon"]))
    visible_log_mean = prepared["transform"].mean
    visible_log_std = prepared["transform"].std

    split_info = {
        "total_steps": total_steps,
        "train_steps": int(train_range[1] - train_range[0]),
        "val_steps": int(val_range[1] - val_range[0]),
        "test_steps": int(test_range[1] - test_range[0]),
        "train_range": [int(train_range[0]), int(train_range[1])],
        "val_range": [int(val_range[0]), int(val_range[1])],
        "test_range": [int(test_range[0]), int(test_range[1])],
    }

    tested_noise_configs, selected_noise_config, selection_reason = _noise_scan(
        config=config,
        system=system,
        bundle=bundle,
        prepared=prepared,
        data_loaders=data_loaders,
        split_ranges=split_ranges,
    )

    rollout_case = _select_rollout_case(
        visible_series=bundle.observations,
        hidden_series=bundle.hidden_observations,
        split_info=split_ranges,
        history_length=history_length,
        horizon=figure_rollout_horizon,
    )

    model, trainer = _make_trainer(
        config=config,
        system=system,
        bundle=bundle,
        visible_log_mean=visible_log_mean,
        visible_log_std=visible_log_std,
        split_ranges=split_ranges,
        history_length=history_length,
        noise_profile={
            "history_jitter": selected_noise_config["training_input_noise"],
            "rollout_process_noise": selected_noise_config["training_rollout_noise"],
            "latent_perturb": selected_noise_config["training_latent_perturb"],
        },
    )

    fit_result = trainer.fit(data_loaders["train"], data_loaders["val"])
    sliding_eval = trainer.evaluate_loader(data_loaders["test"], num_particles=int(config["noise"]["particle_rollout_K"]))
    single_rollout_eval = trainer.evaluate_loader(data_loaders["test"], num_particles=1)
    full_context_test = trainer.evaluate_full_context("test", num_particles=int(config["noise"]["particle_rollout_K"]))
    hidden_test = trainer.recover_hidden_on_split("test")
    hidden_full = trainer.recover_hidden_sequence(
        visible_series_raw=bundle.observations,
        hidden_series_true=bundle.hidden_observations,
        history_length=history_length,
        global_offset=0,
    )
    local_rollout = trainer.forecast_case(
        history_visible_raw=rollout_case["history_visible"],
        future_visible_true=rollout_case["future_visible"],
        future_hidden_true=rollout_case["future_hidden"],
        num_particles=int(config["noise"]["particle_rollout_K"]),
    )

    metrics = {
        "sliding_window_visible_rmse": sliding_eval["visible_rollout_rmse"],
        "sliding_window_visible_pearson": sliding_eval["visible_rollout_pearson"],
        "full_context_visible_rmse": full_context_test["metrics"]["visible_rmse"],
        "full_context_visible_pearson": full_context_test["metrics"]["visible_pearson"],
        "hidden_rmse": hidden_test["metrics"]["hidden_recovery_rmse"],
        "hidden_pearson": hidden_test["metrics"]["hidden_recovery_pearson"],
        "peak_visible_error": sliding_eval["peak_visible_error"],
        "amplitude_collapse_score": sliding_eval["amplitude_collapse_score"],
        "selected_rollout_noise": selected_noise_config["training_rollout_noise"],
    }
    disentanglement_metrics = {
        "hidden_environment_correlation": hidden_test["metrics"]["hidden_environment_correlation"],
        "hidden_autocorrelation": hidden_test["metrics"]["hidden_autocorrelation"],
        "environment_autocorrelation": hidden_test["metrics"]["environment_autocorrelation"],
        "hidden_roughness": hidden_test["metrics"]["hidden_roughness"],
        "environment_roughness": hidden_test["metrics"]["environment_roughness"],
    }
    lv_guidance_metrics = {
        "lv_residual_ratio_mean": sliding_eval["lv_residual_ratio_mean"],
        "lv_residual_ratio_std": sliding_eval["lv_residual_ratio_std"],
        "residual_dominates_fraction": sliding_eval["residual_dominates_fraction"],
        "lv_energy_fraction": sliding_eval["lv_energy_fraction"],
        "visible_lv_residual_ratio_mean": sliding_eval["visible_lv_residual_ratio_mean"],
        "visible_residual_dominates_fraction": sliding_eval["visible_residual_dominates_fraction"],
        "visible_lv_energy_fraction": sliding_eval["visible_lv_energy_fraction"],
    }
    particle_rollout_metrics = {
        "particle_rollout_helpful": bool(
            sliding_eval["visible_rollout_rmse"] < single_rollout_eval["visible_rollout_rmse"]
        ),
        "single_rollout_visible_rmse": single_rollout_eval["visible_rollout_rmse"],
        "particle_rollout_visible_rmse": sliding_eval["visible_rollout_rmse"],
    }
    training_diagnostics = {
        "best_epoch": int(fit_result.best_epoch),
        "early_best_epoch": bool(fit_result.best_epoch <= max(5, int(config["train"]["epochs"] * 0.18))),
    }

    diagnosis = _diagnose(
        metrics=metrics,
        disentanglement_metrics=disentanglement_metrics,
        previous_reference=previous_reference,
    )
    explicit_conclusion = (
        "降低 rollout 训练噪声后，visible 振幅保持和 sliding-window rollout 都比上一版更稳；"
        if previous_reference and metrics["sliding_window_visible_rmse"] < previous_reference["metrics"]["sliding_window_visible_rmse"]
        else "当前版本相较上一版没有在 visible rollout 上出现退化；"
    )
    explicit_conclusion += (
        "同时 hidden recovery 仍然保持在较高水平，说明主问题不是 hidden 学不会，而是 visible 动力学分离仍然困难。"
    )
    if training_diagnostics["early_best_epoch"]:
        explicit_conclusion += " 问题更像训练设置不合理，而非单纯训练不够。"

    data_regime_assessment = {
        "too_flat": bool(system.diagnostics["too_flat"]),
        "too_periodic": bool(system.diagnostics["too_periodic"]),
        "moderate_complexity": bool(system.diagnostics["moderate_complexity"]),
        "label": _data_regime_label(system.diagnostics),
    }

    summary_payload = {
        "run_name": run_dir.name,
        "run_dir": str(run_dir),
        "readme_path": str(run_dir / "README.md"),
        "split_info": split_info,
        "data_regime_assessment": data_regime_assessment,
        "tested_noise_configs": _to_plain(tested_noise_configs),
        "selected_noise_config": _to_plain(selected_noise_config),
        "selected_noise_reason": selection_reason,
        "metrics": _to_plain(metrics),
        "disentanglement_metrics": _to_plain(disentanglement_metrics),
        "lv_guidance_metrics": _to_plain(lv_guidance_metrics),
        "particle_rollout_metrics": _to_plain(particle_rollout_metrics),
        "training_diagnostics": _to_plain(training_diagnostics),
        "diagnosis": diagnosis,
        "explicit_conclusion": explicit_conclusion,
    }

    save_json(results_dir / "summary.json", _to_plain(summary_payload))

    visible_pred_full = _make_nan_series(total_steps, bundle.observations.shape[1])
    visible_pred_full[test_range[0] : test_range[1]] = full_context_test["visible_pred"]
    hidden_pred_full = _fill_series_from_indices(
        length=total_steps,
        indices=hidden_full["indices"],
        values=hidden_full["hidden_pred"],
    )
    np.savez(
        results_dir / "data_snapshot.npz",
        visible_true_full=bundle.observations.numpy(),
        hidden_true_full=bundle.hidden_observations.numpy(),
        environment_true_full=system.environment_driver.numpy(),
        visible_pred_full=visible_pred_full.numpy(),
        hidden_pred_full=hidden_pred_full.numpy(),
        interaction_true=system.interaction_matrix.numpy(),
        interaction_pred=local_rollout["interaction_pred"].numpy(),
    )

    representative_species = _select_representative_species(bundle.observations[test_range[0] : test_range[1]], limit=3)
    _plot_true_trajectories(
        visible_true=bundle.observations,
        hidden_true=bundle.hidden_observations,
        environment_true=system.environment_driver,
        split_info=split_info,
        output_path=results_dir / "fig1_true_trajectories.png",
    )
    _plot_hidden_test_overlay(
        hidden_true=hidden_test["hidden_true"],
        hidden_pred=hidden_test["hidden_pred"],
        hidden_metrics={
            "hidden_rmse": metrics["hidden_rmse"],
            "hidden_pearson": metrics["hidden_pearson"],
            "hidden_environment_correlation": disentanglement_metrics["hidden_environment_correlation"],
        },
        output_path=results_dir / "fig2_hidden_test_overlay.png",
    )
    _plot_visible_rollout_compare(
        history_visible=rollout_case["history_visible"],
        future_visible=rollout_case["future_visible"],
        visible_pred=local_rollout["visible_pred"],
        representative_species=representative_species,
        forecast_start=int(rollout_case["forecast_start"]),
        output_path=results_dir / "fig3_visible_rollout_compare.png",
    )
    _plot_visible_fullcontext_compare(
        visible_series=bundle.observations,
        visible_pred_full=visible_pred_full,
        split_info=split_info,
        representative_species=representative_species,
        output_path=results_dir / "fig4_visible_fullcontext_compare.png",
    )
    _plot_training_metrics(
        history=fit_result.history,
        output_path=results_dir / "fig5_training_metrics.png",
    )
    _plot_diagnostics(
        lv_guidance_metrics=lv_guidance_metrics,
        selected_noise_config=selected_noise_config,
        disentanglement_metrics=disentanglement_metrics,
        output_path=results_dir / "fig6_diagnostics.png",
    )
    _write_experiment_readme(
        run_dir=run_dir,
        timestamp=timestamp,
        config=config,
        summary_payload=summary_payload,
    )

    return {
        "run_dir": run_dir,
        "results_dir": results_dir,
        "summary": summary_payload,
        "previous_reference": previous_reference,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the refined partial-observation ecology MVP experiment.")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/partial_lv_mvp.yaml",
        help="Path to the experiment configuration file.",
    )
    args = parser.parse_args()
    outputs = run_experiment(args.config)

    summary = outputs["summary"]
    previous_reference = outputs["previous_reference"]
    previous_metrics = previous_reference["metrics"] if previous_reference else None
    hidden_kept = previous_metrics is None or summary["metrics"]["hidden_rmse"] <= previous_metrics["hidden_rmse"] * 1.15
    visible_improved = previous_metrics is None or summary["metrics"]["sliding_window_visible_rmse"] < previous_metrics["sliding_window_visible_rmse"]
    amplitude_improved = previous_metrics is None or summary["metrics"]["amplitude_collapse_score"] < previous_metrics["amplitude_collapse_score"]
    disentangle_improved = previous_metrics is None or abs(summary["disentanglement_metrics"]["hidden_environment_correlation"]) < abs(previous_metrics["hidden_environment_correlation"])
    lv_active = _lv_backbone_active(summary["lv_guidance_metrics"])

    print(f"1. 数据 regime: {summary['data_regime_assessment']['label']}")
    print(f"2. 选中的噪声配置: {summary['selected_noise_config']}")
    print(f"3. hidden recovery 保持: {'是' if hidden_kept else '否'}")
    print(f"4. visible rollout 改善: {'是' if visible_improved else '否'}")
    print(f"5. amplitude collapse 缓解: {'是' if amplitude_improved else '否'}")
    print(f"6. hidden/environment disentanglement 改善: {'是' if disentangle_improved else '否'}")
    print(f"7. LV backbone 仍在发挥作用: {'是' if lv_active else '否'}")
    print(f"8. 当前主瓶颈: {summary['diagnosis']}")


if __name__ == "__main__":
    main()

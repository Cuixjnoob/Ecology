"""部分观测 LV 系统的主实验入口脚本。

完整流程：
  1. 加载 YAML 配置（默认 configs/partial_lv_mvp.yaml）
  2. 合成数据生成（5 可见 + 1 隐藏，820 步）
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
from data.partial_nonlinear_mvp import generate_partial_nonlinear_mvp_system
from models.partial_lv_recovery_model import PartialLVRecoveryModel
from train.partial_lv_mvp_trainer import PartialLVMVPTrainer
from train.utils import create_data_loaders, save_json, set_random_seed


def _generate_system(config: Dict[str, Any]):
    """根据 config['simulation']['generator'] 选择数据生成器。
    支持: 'lv' (默认) / 'nonlinear'
    """
    sim_cfg = config["simulation"]
    generator_type = sim_cfg.get("generator", "lv")
    kwargs = dict(
        total_steps=int(sim_cfg["total_steps"]),
        warmup_steps=int(sim_cfg["warmup_steps"]),
        process_noise=float(sim_cfg["process_noise"]),
        seed=int(config["experiment"]["seed"]),
        max_attempts=int(sim_cfg["max_attempts"]),
        max_state_value=float(sim_cfg["max_state_value"]),
    )
    if generator_type == "nonlinear":
        return generate_partial_nonlinear_mvp_system(**kwargs)
    return generate_partial_lv_mvp_system(**kwargs)


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
        use_lv_guidance=bool(model_cfg.get("use_lv_guidance", True)),
        lv_form=str(model_cfg.get("lv_form", "tanh")),
        use_residual=bool(model_cfg.get("use_residual", True)),
        use_hidden_fast=bool(model_cfg.get("use_hidden_fast", True)),
        max_state_value=float(simulation_cfg["max_state_value"]),
        base_visible_noise=float(model_cfg.get("base_visible_noise", 0.015)),
        base_hidden_noise=float(model_cfg.get("base_hidden_noise", 0.012)),
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



def _diagnose(
    metrics: Dict[str, float],
    previous_reference: Dict[str, Any] | None,
) -> str:
    # Hidden recovery-centric diagnosis
    hidden_pearson = metrics.get("hidden_test_pearson", metrics.get("hidden_pearson", 0.0))
    if hidden_pearson < 0.30:
        return "hidden recovery failed"
    if hidden_pearson >= 0.85:
        return "good hidden recovery"
    if hidden_pearson >= 0.60:
        return "moderate hidden recovery"
    return "weak hidden recovery"


def _write_experiment_readme(
    run_dir: Path,
    timestamp: str,
    config: Dict[str, Any],
    summary_payload: Dict[str, Any],
) -> None:
    m = summary_payload["metrics"]
    readme_lines = [
        "# 实验 README",
        "",
        f"- 时间戳：`{timestamp}`",
        f"- 实验名：`{config['experiment']['name']}`",
        f"- 模式：**hidden recovery-centric**（不做未来预测，专注隐藏物种恢复）",
        "",
        "## 核心方法",
        "- 目标：从可见物种的已知时间序列中恢复未观测的隐藏物种",
        "- 训练：在已知数据上做滑动窗口重构，visible rollout 作为训练信号（非预测目标）",
        "- 评估：以 hidden recovery quality (RMSE/Pearson) 为核心指标",
        "- 架构：LV soft guidance + stochastic residual + hidden fast innovation",
        "",
        "## 本次结果",
        f"- Hidden Test RMSE: `{m['hidden_test_rmse']:.4f}`",
        f"- Hidden Test Pearson: `{m['hidden_test_pearson']:.4f}`",
        f"- Hidden Val Pearson: `{m['hidden_val_pearson']:.4f}`",
        f"- 数据 regime: `{summary_payload['data_regime_assessment']['label']}`",
        f"- 诊断: `{summary_payload['diagnosis']}`",
    ]
    (run_dir / "README.md").write_text("\n".join(readme_lines), encoding="utf-8")


def _plot_true_trajectories(
    visible_true: torch.Tensor,
    hidden_true: torch.Tensor,
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
    axes[1].set_title("真实隐藏物种轨迹")
    axes[1].set_ylabel("隐藏物种丰度")
    axes[1].legend(loc="upper right", fontsize=8)

    for axis in axes:
        axis.axvline(train_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
        axis.axvline(val_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
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
        f"hidden RMSE={hidden_metrics['hidden_rmse']:.4f} | Pearson={hidden_metrics['hidden_pearson']:.4f}"
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


def _plot_hidden_full_recovery(
    hidden_true_full: torch.Tensor,
    hidden_pred_full: torch.Tensor,
    split_info: Dict[str, Any],
    hidden_metrics: Dict[str, float],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(14, 5), constrained_layout=True)
    time_axis = np.arange(hidden_true_full.shape[0])
    train_end = split_info["train_range"][1]
    val_end = split_info["val_range"][1]

    axis.plot(time_axis, hidden_true_full[:, 0].numpy(), color="black", linewidth=2.0, label="真实 hidden")
    pred_numpy = hidden_pred_full[:, 0].numpy()
    valid_mask = ~np.isnan(pred_numpy)
    axis.plot(time_axis[valid_mask], pred_numpy[valid_mask], color="#ff7f0e", linewidth=1.5, label="恢复 hidden")
    axis.set_title("全序列隐藏物种恢复")
    axis.set_ylabel("丰度")
    axis.set_xlabel("时间步")
    axis.legend(loc="upper right", fontsize=9)
    metric_text = (
        f"hidden RMSE={hidden_metrics['hidden_rmse']:.4f} | "
        f"Pearson={hidden_metrics['hidden_pearson']:.4f}"
    )
    axis.text(
        0.02, 0.97, metric_text, transform=axis.transAxes, va="top", ha="left", fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#cccccc"},
    )

    axis.axvline(train_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)
    axis.axvline(val_end - 0.5, color="#999999", linestyle="--", linewidth=1.0)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_training_metrics(history: List[Dict[str, float]], output_path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), constrained_layout=True)
    epochs = [record["epoch"] for record in history]
    axes[0].plot(epochs, [record["train_total"] for record in history], color="#ff7f0e", linewidth=1.8, label="train loss")
    axes[0].plot(epochs, [record["val_score"] for record in history], color="#4c78a8", linestyle="--", linewidth=1.8, label="val score")
    axes[0].set_title("训练损失 / 验证分数")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend(fontsize=8)

    metric_specs = [
        ("val_hidden_recovery_rmse", "-", "hidden RMSE"),
        ("val_hidden_recovery_pearson", "--", "hidden Pearson"),
    ]
    for metric_key, linestyle, label in metric_specs:
        axes[1].plot(epochs, [record[metric_key] for record in history], linestyle=linestyle, linewidth=1.8, label=label)
    axes[1].set_title("验证集关键指标")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Metric")
    axes[1].legend(fontsize=8)

    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_noise_diagnostics(
    selected_noise_config: Dict[str, float],
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(1, 1, figsize=(7, 4.8), constrained_layout=True)

    noise_labels = ["input", "rollout", "latent"]
    noise_values = [
        selected_noise_config["training_input_noise"],
        selected_noise_config["training_rollout_noise"],
        selected_noise_config["training_latent_perturb"],
    ]
    axis.bar(noise_labels, noise_values, color="#59a14f")
    axis.set_title("Selected Noise Config")
    axis.set_ylabel("Noise Level")

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
        val_hidden = trainer.recover_hidden_on_split("val")
        # Pure hidden recovery noise selection — no future prediction
        score = (
            0.60 * val_hidden["metrics"]["hidden_recovery_rmse"]
            + 0.30 * (1.0 - max(0.0, min(1.0, val_hidden["metrics"]["hidden_recovery_pearson"])))
            + 0.10 * float(fit_result.best_epoch <= max(3, int(scan_config["train"]["epochs"] * 0.35)))
        )
        entry = {
            **noise_config,
            "validation_score": float(score),
            "hidden_rmse": float(val_hidden["metrics"]["hidden_recovery_rmse"]),
            "hidden_pearson": float(val_hidden["metrics"]["hidden_recovery_pearson"]),
            "best_epoch": int(fit_result.best_epoch),
        }
        tested_results.append(entry)
        if score < best_score:
            best_score = score
            best_config = noise_config

    if best_config is None:
        raise RuntimeError("Noise scan failed to select a configuration.")

    selection_reason = (
        "选中该配置是因为它在验证集上以 hidden recovery 为核心的综合评分最优，"
        "同时兼顾了 visible 重构质量。"
    )
    return tested_results, best_config, selection_reason


def run_experiment(config_path: str | Path) -> Dict[str, Any]:
    config = _load_config(config_path)
    _configure_matplotlib()
    set_random_seed(int(config["experiment"]["seed"]))
    timestamp, run_dir, results_dir = _create_run_dirs(config)

    previous_reference = _load_previous_reference()

    system = _generate_system(config)
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
    # Pure hidden recovery evaluation — no future prediction
    hidden_test = trainer.recover_hidden_on_split("test")
    hidden_val = trainer.recover_hidden_on_split("val")
    hidden_train = trainer.recover_hidden_on_split("train")
    hidden_full = trainer.recover_hidden_sequence(
        visible_series_raw=bundle.observations,
        hidden_series_true=bundle.hidden_observations,
        history_length=history_length,
        global_offset=0,
    )

    # Primary metrics: pure hidden recovery
    metrics = {
        "hidden_test_rmse": hidden_test["metrics"]["hidden_recovery_rmse"],
        "hidden_test_pearson": hidden_test["metrics"]["hidden_recovery_pearson"],
        "hidden_val_rmse": hidden_val["metrics"]["hidden_recovery_rmse"],
        "hidden_val_pearson": hidden_val["metrics"]["hidden_recovery_pearson"],
        "hidden_train_rmse": hidden_train["metrics"]["hidden_recovery_rmse"],
        "hidden_train_pearson": hidden_train["metrics"]["hidden_recovery_pearson"],
        "selected_rollout_noise": selected_noise_config["training_rollout_noise"],
    }
    training_diagnostics = {
        "best_epoch": int(fit_result.best_epoch),
        "early_best_epoch": bool(fit_result.best_epoch <= max(5, int(config["train"]["epochs"] * 0.18))),
    }

    diagnosis = _diagnose(
        metrics=metrics,
        previous_reference=previous_reference,
    )
    # Hidden recovery-centric conclusion
    explicit_conclusion = f"Hidden recovery: RMSE={metrics['hidden_test_rmse']:.4f}, Pearson={metrics['hidden_test_pearson']:.4f}. "
    if metrics["hidden_test_pearson"] >= 0.85:
        explicit_conclusion += "隐藏物种恢复质量良好。"
    elif metrics["hidden_test_pearson"] >= 0.60:
        explicit_conclusion += "隐藏物种恢复质量中等，有改善空间。"
    else:
        explicit_conclusion += "隐藏物种恢复质量不佳，需要诊断原因。"

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
        "training_diagnostics": _to_plain(training_diagnostics),
        "diagnosis": diagnosis,
        "explicit_conclusion": explicit_conclusion,
    }

    save_json(results_dir / "summary.json", _to_plain(summary_payload))

    hidden_pred_full = _fill_series_from_indices(
        length=total_steps,
        indices=hidden_full["indices"],
        values=hidden_full["hidden_pred"],
    )
    # 参数恢复所需的数据：真实参数 + 模型学到的参数
    model = trainer.model
    growth_rates_pred = model.growth_rates.detach().cpu().numpy()
    interaction_matrix_pred = model.get_interaction_matrix().detach().cpu().numpy()
    # lv_drift_scale = 0.08 + 0.20 * sigmoid(unconstrained)
    lv_drift_scale_pred = float(
        (0.08 + 0.20 * torch.sigmoid(model.lv_drift_scale_unconstrained)).detach().cpu().item()
    )
    alpha_lv_pred = float(model.alpha_lv().detach().cpu().item())
    alpha_res_pred = float(model.alpha_res().detach().cpu().item())

    np.savez(
        results_dir / "data_snapshot.npz",
        visible_true_full=bundle.observations.numpy(),
        hidden_true_full=bundle.hidden_observations.numpy(),
        hidden_pred_full=hidden_pred_full.numpy(),
        # interaction matrix
        interaction_true=system.interaction_matrix.numpy(),
        interaction_pred=interaction_matrix_pred,
        # growth rates（模型 LV 分支会用到；无 LV 先验时停留在初值 0.08）
        growth_rates_true=system.growth_rates.numpy(),
        growth_rates_pred=growth_rates_pred,
        # mixing / scale coefficients (模型特有)
        alpha_lv_pred=np.array([alpha_lv_pred]),
        alpha_res_pred=np.array([alpha_res_pred]),
        lv_drift_scale_pred=np.array([lv_drift_scale_pred]),
        # 数据生成器的其他参数（参考，模型没有对应项）
        environment_loadings_true=system.environment_loadings.numpy(),
        pulse_loadings_true=system.pulse_loadings.numpy(),
    )

    # fig1: true trajectories
    _plot_true_trajectories(
        visible_true=bundle.observations,
        hidden_true=bundle.hidden_observations,
        split_info=split_info,
        output_path=results_dir / "fig1_true_trajectories.png",
    )
    # fig2: hidden test overlay
    _plot_hidden_test_overlay(
        hidden_true=hidden_test["hidden_true"],
        hidden_pred=hidden_test["hidden_pred"],
        hidden_metrics={
            "hidden_rmse": metrics["hidden_test_rmse"],
            "hidden_pearson": metrics["hidden_test_pearson"],
        },
        output_path=results_dir / "fig2_hidden_test_overlay.png",
    )
    # fig3: hidden full recovery (全序列)
    _plot_hidden_full_recovery(
        hidden_true_full=bundle.hidden_observations,
        hidden_pred_full=hidden_pred_full,
        split_info=split_info,
        hidden_metrics={
            "hidden_rmse": metrics["hidden_test_rmse"],
            "hidden_pearson": metrics["hidden_test_pearson"],
        },
        output_path=results_dir / "fig3_hidden_full_recovery.png",
    )
    # fig4: training metrics
    _plot_training_metrics(
        history=fit_result.history,
        output_path=results_dir / "fig4_training_metrics.png",
    )
    # fig5: noise diagnostics
    _plot_noise_diagnostics(
        selected_noise_config=selected_noise_config,
        output_path=results_dir / "fig5_diagnostics.png",
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
    m = summary["metrics"]

    print("=" * 60)
    print("  Hidden Species Recovery Results")
    print("=" * 60)
    print(f"  Test  Hidden RMSE:    {m['hidden_test_rmse']:.4f}")
    print(f"  Test  Hidden Pearson: {m['hidden_test_pearson']:.4f}")
    print(f"  Val   Hidden RMSE:    {m['hidden_val_rmse']:.4f}")
    print(f"  Val   Hidden Pearson: {m['hidden_val_pearson']:.4f}")
    print(f"  Train Hidden RMSE:    {m['hidden_train_rmse']:.4f}")
    print(f"  Train Hidden Pearson: {m['hidden_train_pearson']:.4f}")
    print(f"  Noise Config: {summary['selected_noise_config']}")
    print(f"  Diagnosis: {summary['diagnosis']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

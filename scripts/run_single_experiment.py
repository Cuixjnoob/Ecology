from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from data.dataset import build_windowed_datasets
from data.lv_simulator import HiddenLotkaVolterraSimulation, generate_hidden_lv_simulation
from eval.rollout_eval import evaluate_rollout_model
from train.trainer import (
    Trainer,
    build_model_from_config,
    create_data_loaders,
    load_checkpoint,
    save_json,
    set_random_seed,
)
from viz.chinese_experiment import (
    plot_gate_heatmap,
    plot_hidden_reference,
    plot_hidden_truth_vs_prediction,
    plot_lv_interaction_matrix,
    plot_lv_system_overview,
    plot_rollout_comparison,
    plot_species_error_heatmap,
    plot_takens_delay_stack,
    plot_takens_embedding_scatter,
)


def load_config(config_path: str | Path) -> Dict[str, Any]:
    with Path(config_path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _sanitize_takens_config(config: Dict[str, Any]) -> None:
    takens_cfg = config.get("takens", {})
    if takens_cfg:
        config["data"]["delay_stride"] = int(takens_cfg.get("tau", config["data"]["delay_stride"]))
        config["data"]["delay_length"] = int(takens_cfg.get("m", config["data"]["delay_length"]))


def _build_simulation(config: Dict[str, Any]) -> HiddenLotkaVolterraSimulation:
    simulation_cfg = config["simulation"]
    return generate_hidden_lv_simulation(
        total_steps=int(simulation_cfg["total_steps"]),
        warmup_steps=int(simulation_cfg["warmup_steps"]),
        num_observed=int(simulation_cfg["observed_species"]),
        num_hidden=int(simulation_cfg["hidden_species"]),
        dt=float(simulation_cfg["dt"]),
        process_noise=float(simulation_cfg["process_noise"]),
        seed=int(config["experiment"]["seed"]),
    )


def _extract_sample_rollout(
    model: torch.nn.Module,
    dataset,
    sample_index: int,
    transform,
    horizon: int,
    device: torch.device,
    ) -> Dict[str, torch.Tensor]:
    if len(dataset) == 0:
        raise RuntimeError("测试数据集为空，无法提取分析样本。")
    sample_index = max(0, min(sample_index, len(dataset) - 1))
    sample = dataset[sample_index]
    history = sample["history"].unsqueeze(0).to(device)
    future = sample["future"][:horizon].unsqueeze(0).to(device)
    history_u = sample["history_u"].unsqueeze(0).to(device)
    future_u = sample["future_u"][:horizon].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            history_x=history,
            history_u=history_u,
            future_u=future_u,
            rollout_horizon=horizon,
            teacher_forcing_targets=None,
            teacher_forcing_ratio=0.0,
        )

    predictions = outputs["predictions"][0].cpu()
    future_truth = future[0].cpu()
    history_truth = sample["history"].cpu()
    predictions_raw = transform.inverse_transform(predictions)
    future_truth_raw = transform.inverse_transform(future_truth)
    history_truth_raw = transform.inverse_transform(history_truth)

    return {
        "sample": sample,
        "outputs": outputs,
        "history_raw": history_truth_raw,
        "future_raw": future_truth_raw,
        "predictions_raw": predictions_raw,
    }


@torch.no_grad()
def _collect_hidden_probe_batch(
    model: torch.nn.Module,
    data_loader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    features = []
    targets = []
    model.eval()
    for batch in data_loader:
        future_hidden = batch["future_hidden"]
        if future_hidden.shape[-1] == 0:
            continue
        history = batch["history"].to(device)
        history_u = batch["history_u"].to(device)
        future_u = batch["future_u"][:, :1].to(device)
        outputs = model(
            history_x=history,
            history_u=history_u,
            future_u=future_u,
            rollout_horizon=1,
            teacher_forcing_targets=None,
            teacher_forcing_ratio=0.0,
        )
        latent_summary = outputs["latent_summary"][:, 0].cpu()
        target_hidden = future_hidden[:, 0, :].cpu()
        features.append(latent_summary)
        targets.append(target_hidden)

    if not features:
        raise RuntimeError("未能收集到隐藏物种探针所需的特征。")
    return torch.cat(features, dim=0), torch.cat(targets, dim=0)


def _pearson_mean(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
    target_centered = targets - targets.mean(dim=0, keepdim=True)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)
    ).clamp_min(1e-8)
    return float(((pred_centered * target_centered).sum(dim=0) / denominator).mean().item())


def _fit_hidden_probe(
    model: torch.nn.Module,
    train_loader,
    test_loader,
    device: torch.device,
) -> Dict[str, Any]:
    train_features, train_hidden = _collect_hidden_probe_batch(model, train_loader, device)
    test_features, test_hidden = _collect_hidden_probe_batch(model, test_loader, device)

    train_hidden_log = torch.log1p(train_hidden.clamp_min(0.0))
    test_hidden_log = torch.log1p(test_hidden.clamp_min(0.0))

    train_design = torch.cat(
        [train_features, torch.ones(train_features.shape[0], 1)],
        dim=1,
    )
    test_design = torch.cat(
        [test_features, torch.ones(test_features.shape[0], 1)],
        dim=1,
    )

    solution = torch.linalg.lstsq(train_design, train_hidden_log).solution
    test_hidden_log_pred = test_design @ solution
    test_hidden_pred = torch.expm1(test_hidden_log_pred).clamp_min(0.0)

    return {
        "weights": solution,
        "test_hidden_pred": test_hidden_pred,
        "test_hidden_true": test_hidden,
        "metrics": {
            "rmse_raw": float(torch.sqrt((test_hidden_pred - test_hidden).square().mean()).item()),
            "mae_raw": float((test_hidden_pred - test_hidden).abs().mean().item()),
            "rmse_log": float(torch.sqrt((test_hidden_log_pred - test_hidden_log).square().mean()).item()),
            "pearson_mean": _pearson_mean(test_hidden_pred, test_hidden),
        },
    }


def _predict_hidden_from_sample_rollout(
    latent_summary: torch.Tensor,
    probe_weights: torch.Tensor,
) -> torch.Tensor:
    design = torch.cat(
        [latent_summary, torch.ones(latent_summary.shape[0], 1)],
        dim=1,
    )
    hidden_log_pred = design @ probe_weights
    return torch.expm1(hidden_log_pred).clamp_min(0.0)


def _build_chinese_summary(
    config: Dict[str, Any],
    simulation: HiddenLotkaVolterraSimulation,
    training_results: Dict[str, Any],
    evaluation_results: Dict[str, Any],
    hidden_probe_report: Dict[str, Any],
) -> Dict[str, Any]:
    primary_horizon = str(min(int(config["eval"]["primary_horizon"]), int(config["data"]["horizon"])))
    primary_metrics = evaluation_results["transformed"][primary_horizon]

    return {
        "实验名称": config["experiment"]["name"],
        "随机种子": int(config["experiment"]["seed"]),
        "数据设置": {
            "可见物种数": simulation.num_observed,
            "隐藏物种数": simulation.num_hidden,
            "总时间步": simulation.total_steps,
            "积分步长dt": simulation.dt,
        },
        "Takens参数": {
            "tau": int(config["data"]["delay_stride"]),
            "m": int(config["data"]["delay_length"]),
            "历史窗口长度": int(config["data"]["history_length"]),
        },
        "模型设置": {
            "潜在节点数": int(config["model"]["num_hidden_nodes"]),
            "节点嵌入维度": int(config["model"]["embedding_dim"]),
            "消息传递层数": int(config["model"]["num_message_passing_layers"]),
            "使用潜在递推记忆": bool(config["model"].get("use_latent_recurrence", True)),
        },
        "训练结果": {
            "最佳验证主指标RMSE": training_results["best_metric"],
            "最佳模型路径": training_results["best_checkpoint"],
        },
        "测试集主结果": {
            "主视野": int(primary_horizon),
            "rollout_RMSE": primary_metrics["rmse"],
            "rollout_MAE": primary_metrics["mae"],
            "平均Pearson相关": primary_metrics["pearson_mean"],
            "平均Spearman相关": primary_metrics["spearman_mean"],
            "稳定性失败率": primary_metrics["stability_failure_rate"],
        },
        "模型诊断": evaluation_results.get("diagnostics", {}),
        "隐藏物种线性探针": hidden_probe_report["metrics"],
    }


def run_single_experiment(
    config_path: str | Path,
    output_dir_override: str | None = None,
) -> Dict[str, Any]:
    config = load_config(config_path)
    _sanitize_takens_config(config)
    if output_dir_override:
        config["experiment"]["output_dir"] = output_dir_override

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(int(config["experiment"]["seed"]))

    simulation = _build_simulation(config)
    observed_bundle = simulation.to_observed_bundle()
    prepared = build_windowed_datasets(
        bundle=observed_bundle,
        history_length=int(config["data"]["history_length"]),
        horizon=int(config["data"]["horizon"]),
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
    )
    transform = prepared["transform"]
    transformed_full = transform.transform(observed_bundle.observations)

    data_loaders = create_data_loaders(
        datasets=prepared["datasets"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
    )
    model = build_model_from_config(
        config=config,
        num_observed=observed_bundle.num_observed,
        covariate_dim=observed_bundle.covariate_dim,
    )
    trainer = Trainer(
        model=model,
        config=config,
        bounds=prepared["bounds"],
        output_dir=output_dir,
        transform_state=transform.state_dict(),
        num_observed=observed_bundle.num_observed,
        covariate_dim=observed_bundle.covariate_dim,
    )

    training_results = trainer.fit(
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        transform=transform,
    )
    if training_results["best_checkpoint"]:
        trained_model, _ = load_checkpoint(training_results["best_checkpoint"], trainer.device)
    else:
        trained_model = model.to(trainer.device)
        trained_model.eval()

    evaluation_results = evaluate_rollout_model(
        trained_model,
        data_loaders["test"],
        device=trainer.device,
        horizons=config["eval"]["horizons"],
        transform=transform,
    )
    hidden_probe_report = _fit_hidden_probe(
        model=trained_model,
        train_loader=data_loaders["train"],
        test_loader=data_loaders["test"],
        device=trainer.device,
    )

    metrics_payload = {
        "experiment": config["experiment"],
        "simulation": config["simulation"],
        "takens": config["takens"],
        "training": training_results,
        "hidden_probe": hidden_probe_report["metrics"],
        "test": {
            key: value
            for key, value in evaluation_results.items()
            if key != "artifacts"
        },
    }
    save_json(output_dir / "metrics.json", metrics_payload)
    save_json(output_dir / "training_history.json", {"history": training_results["history"]})
    save_json(output_dir / "隐藏物种探针.json", hidden_probe_report["metrics"])
    save_json(
        output_dir / "实验摘要.json",
        _build_chinese_summary(
            config,
            simulation,
            training_results,
            evaluation_results,
            hidden_probe_report,
        ),
    )

    analysis_index = int(config["eval"].get("analysis_window_index", 0))
    max_horizon = min(max(int(value) for value in config["eval"]["horizons"]), int(config["data"]["horizon"]))
    sample_rollout = _extract_sample_rollout(
        model=trained_model,
        dataset=prepared["datasets"]["test"],
        sample_index=analysis_index,
        transform=transform,
        horizon=max_horizon,
        device=trainer.device,
    )

    sample = sample_rollout["sample"]
    window_end = int(sample["window_end_index"].item())
    future_hidden_truth = simulation.hidden_states[window_end + 1 : window_end + 1 + max_horizon]
    predicted_hidden_from_rollout = _predict_hidden_from_sample_rollout(
        latent_summary=sample_rollout["outputs"]["latent_summary"][0].cpu(),
        probe_weights=hidden_probe_report["weights"],
    )

    representative_index = int(config["takens"].get("representative_species_index", 0))
    representative_name = observed_bundle.observed_names[representative_index]
    representative_series = transformed_full[:, representative_index]

    figure_dir = output_dir / "figures"
    plot_lv_system_overview(
        observed_states=simulation.observed_states,
        hidden_states=simulation.hidden_states,
        observed_names=simulation.observed_names,
        hidden_names=simulation.hidden_names,
        output_path=figure_dir / "01_真实LV系统轨迹.png",
    )
    plot_lv_interaction_matrix(
        interaction_matrix=simulation.interaction_matrix,
        num_observed=simulation.num_observed,
        output_path=figure_dir / "02_真实LV相互作用矩阵.png",
    )
    plot_takens_delay_stack(
        observed_series=representative_series,
        species_name=representative_name,
        tau=int(config["data"]["delay_stride"]),
        m=int(config["data"]["delay_length"]),
        output_path=figure_dir / "03_Takens延迟窗口示意.png",
    )
    plot_takens_embedding_scatter(
        observed_series=representative_series,
        species_name=representative_name,
        tau=int(config["data"]["delay_stride"]),
        output_path=figure_dir / "04_Takens二维三维嵌入.png",
    )
    plot_rollout_comparison(
        history_truth=sample_rollout["history_raw"],
        future_truth=sample_rollout["future_raw"],
        future_prediction=sample_rollout["predictions_raw"],
        species_names=observed_bundle.observed_names,
        output_path=figure_dir / "05_正向模拟对比.png",
        max_species=int(config["viz"].get("max_species_in_rollout_plot", 4)),
    )
    plot_species_error_heatmap(
        absolute_errors=(sample_rollout["predictions_raw"] - sample_rollout["future_raw"]).abs(),
        species_names=observed_bundle.observed_names,
        output_path=figure_dir / "06_逐物种误差热图.png",
    )
    plot_gate_heatmap(
        gate_history=sample_rollout["outputs"]["gate_history"].cpu(),
        num_observed=simulation.num_observed,
        output_path=figure_dir / "07_平均边门控热图.png",
    )
    plot_hidden_reference(
        hidden_truth=future_hidden_truth,
        hidden_names=simulation.hidden_names,
        latent_activity=sample_rollout["outputs"]["hidden_activity"][0].cpu(),
        output_path=figure_dir / "08_隐藏物种与潜在活性参考图.png",
    )
    plot_hidden_truth_vs_prediction(
        hidden_truth=future_hidden_truth,
        hidden_prediction=predicted_hidden_from_rollout,
        hidden_names=simulation.hidden_names,
        output_path=figure_dir / "09_真实隐藏物种vs模型推断隐藏物种.png",
    )

    primary_horizon = str(min(int(config["eval"]["primary_horizon"]), int(config["data"]["horizon"])))
    summary = {
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "metrics.json"),
        "summary_path": str(output_dir / "实验摘要.json"),
        "primary_test_rmse": evaluation_results["transformed"][primary_horizon]["rmse"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="运行单次 LV 隐藏物种 Takens-GNN 实验。")
    parser.add_argument(
        "--config",
        default="configs/single_lv_hidden_experiment.yaml",
        help="实验配置文件路径。",
    )
    parser.add_argument("--output-dir", default=None, help="可选的输出目录覆盖。")
    args = parser.parse_args()

    summary = run_single_experiment(args.config, output_dir_override=args.output_dir)
    print(f"实验完成，结果目录：{summary['output_dir']}")
    print(f"主视野测试 RMSE：{summary['primary_test_rmse']:.4f}")
    print(f"实验摘要：{summary['summary_path']}")


if __name__ == "__main__":
    main()

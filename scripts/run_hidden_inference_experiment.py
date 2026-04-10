from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import torch
import yaml

from data.dataset import build_windowed_datasets
from data.lv_simulator import generate_rich_hidden_lv_simulation
from data.transforms import LogZScoreTransform
from models.hidden_inference_model import HiddenSpeciesInferenceModel
from train.hidden_inference_trainer import HiddenInferenceTrainer
from train.trainer import create_data_loaders, resolve_device, save_json, set_random_seed
from viz.chinese_experiment import (
    plot_gate_heatmap,
    plot_hidden_physics_comparison,
    plot_hidden_scatter,
    plot_hidden_truth_vs_prediction,
    plot_lv_interaction_matrix,
    plot_lv_system_overview,
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


def _build_model(config: Dict[str, Any], num_observed: int, covariate_dim: int) -> HiddenSpeciesInferenceModel:
    model_cfg = config["model"]
    data_cfg = config["data"]
    return HiddenSpeciesInferenceModel(
        num_observed=num_observed,
        covariate_dim=covariate_dim,
        delay_length=int(data_cfg["delay_length"]),
        delay_stride=int(data_cfg["delay_stride"]),
        embedding_dim=int(model_cfg["embedding_dim"]),
        global_dim=int(model_cfg["global_dim"]),
        edge_hidden_dim=int(model_cfg["edge_hidden_dim"]),
        num_message_passing_layers=int(model_cfg["num_message_passing_layers"]),
        num_latent_nodes=int(model_cfg["num_latent_nodes"]),
        decoder_hidden_dim=int(model_cfg["decoder_hidden_dim"]),
        dropout=float(model_cfg["dropout"]),
        use_layer_norm=bool(model_cfg["use_layer_norm"]),
        use_species_embeddings=bool(model_cfg["use_species_embeddings"]),
    )


def _load_best_model(checkpoint_path: str | Path, device: torch.device) -> Dict[str, Any]:
    return torch.load(checkpoint_path, map_location=device)


def _ordered_artifacts(artifacts: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    ordering = torch.argsort(artifacts["window_end_index"])
    return {key: value[ordering] if value.ndim > 0 and value.shape[0] == ordering.shape[0] else value for key, value in artifacts.items()}


def _build_summary(
    config: Dict[str, Any],
    evaluation_metrics: Dict[str, float],
    training_results: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "实验名称": config["experiment"]["name"],
        "数据设置": {
            "可见物种数": int(config["simulation"]["observed_species"]),
            "隐藏物种数": int(config["simulation"]["hidden_species"]),
            "时间步数": int(config["simulation"]["total_steps"]),
            "积分步长dt": float(config["simulation"]["dt"]),
        },
        "Takens参数": {
            "tau": int(config["data"]["delay_stride"]),
            "m": int(config["data"]["delay_length"]),
            "历史窗口长度": int(config["data"]["history_length"]),
        },
        "训练结果": {
            "最佳综合验证指标": training_results["best_metric"],
            "最佳模型路径": training_results["best_checkpoint"],
        },
        "测试结果": evaluation_metrics,
    }


def run_hidden_inference_experiment(
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

    simulation = generate_rich_hidden_lv_simulation(
        total_steps=int(config["simulation"]["total_steps"]),
        warmup_steps=int(config["simulation"]["warmup_steps"]),
        num_observed=int(config["simulation"]["observed_species"]),
        num_hidden=int(config["simulation"]["hidden_species"]),
        dt=float(config["simulation"]["dt"]),
        process_noise=float(config["simulation"]["process_noise"]),
        seed=int(config["experiment"]["seed"]),
    )
    observed_bundle = simulation.to_observed_bundle()
    prepared = build_windowed_datasets(
        bundle=observed_bundle,
        history_length=int(config["data"]["history_length"]),
        horizon=int(config["data"]["horizon"]),
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
    )

    splits = prepared["splits"]
    train_start, train_end = splits["train"]
    hidden_transform = LogZScoreTransform()
    hidden_transform.fit(observed_bundle.hidden_observations[train_start:train_end])

    data_loaders = create_data_loaders(
        datasets=prepared["datasets"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
    )
    device = resolve_device(str(config["train"]["device"]))
    model = _build_model(
        config=config,
        num_observed=observed_bundle.num_observed,
        covariate_dim=observed_bundle.covariate_dim,
    )
    trainer = HiddenInferenceTrainer(
        model=model,
        output_dir=output_dir,
        device=device,
        observed_transform=prepared["transform"],
        hidden_transform=hidden_transform,
        growth_rates=simulation.growth_rates,
        interaction_matrix=simulation.interaction_matrix,
        dt=simulation.dt,
        num_observed=observed_bundle.num_observed,
        config=config,
    )
    training_results = trainer.fit(
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
    )

    checkpoint = _load_best_model(training_results["best_checkpoint"], device=device)
    best_model = _build_model(
        config=checkpoint["config"],
        num_observed=int(checkpoint["num_observed"]),
        covariate_dim=0,
    )
    best_model.load_state_dict(checkpoint["model_state"])
    best_model.to(device)

    trainer_for_eval = HiddenInferenceTrainer(
        model=best_model,
        output_dir=output_dir,
        device=device,
        observed_transform=prepared["transform"],
        hidden_transform=hidden_transform,
        growth_rates=simulation.growth_rates,
        interaction_matrix=simulation.interaction_matrix,
        dt=simulation.dt,
        num_observed=observed_bundle.num_observed,
        config=config,
    )
    test_evaluation = trainer_for_eval.evaluate(data_loaders["test"])
    ordered_artifacts = _ordered_artifacts(test_evaluation.artifacts)

    metrics_payload = {
        "experiment": config["experiment"],
        "simulation": config["simulation"],
        "takens": config["takens"],
        "training": training_results,
        "test": test_evaluation.metrics,
    }
    save_json(output_dir / "metrics.json", metrics_payload)
    save_json(output_dir / "training_history.json", {"history": training_results["history"]})
    save_json(output_dir / "实验摘要.json", _build_summary(config, test_evaluation.metrics, training_results))

    representative_index = int(config["takens"]["representative_species_index"])
    representative_series = prepared["transform"].transform(observed_bundle.observations)[:, representative_index]

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
        species_name=simulation.observed_names[representative_index],
        tau=int(config["data"]["delay_stride"]),
        m=int(config["data"]["delay_length"]),
        output_path=figure_dir / "03_Takens延迟窗口示意.png",
    )
    plot_takens_embedding_scatter(
        observed_series=representative_series,
        species_name=simulation.observed_names[representative_index],
        tau=int(config["data"]["delay_stride"]),
        output_path=figure_dir / "04_Takens二维三维嵌入.png",
    )
    plot_hidden_truth_vs_prediction(
        hidden_truth=ordered_artifacts["hidden_true"],
        hidden_prediction=ordered_artifacts["hidden_pred"],
        hidden_names=simulation.hidden_names,
        output_path=figure_dir / "05_真实隐藏物种vs模型预测隐藏物种.png",
    )
    plot_hidden_scatter(
        hidden_truth=ordered_artifacts["hidden_true"].squeeze(-1),
        hidden_prediction=ordered_artifacts["hidden_pred"].squeeze(-1),
        output_path=figure_dir / "06_隐藏物种散点对比.png",
    )
    plot_hidden_physics_comparison(
        hidden_next_true=ordered_artifacts["hidden_next_true"],
        hidden_next_physics=ordered_artifacts["hidden_next_physics"],
        output_path=figure_dir / "07_隐藏物种一步物理一致性.png",
    )
    plot_gate_heatmap(
        gate_history=ordered_artifacts["gate_history"],
        num_observed=simulation.num_observed,
        output_path=figure_dir / "08_平均边门控热图.png",
    )

    return {
        "output_dir": str(output_dir),
        "summary_path": str(output_dir / "实验摘要.json"),
        "hidden_rmse": test_evaluation.metrics["hidden_rmse"],
        "physics_rmse": test_evaluation.metrics["physics_rmse"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="运行单隐藏物种反演实验。")
    parser.add_argument(
        "--config",
        default="configs/hidden_inference_experiment.yaml",
        help="配置文件路径。",
    )
    parser.add_argument("--output-dir", default=None, help="可选输出目录覆盖。")
    args = parser.parse_args()

    summary = run_hidden_inference_experiment(
        config_path=args.config,
        output_dir_override=args.output_dir,
    )
    print(f"实验完成，结果目录：{summary['output_dir']}")
    print(f"隐藏物种测试 RMSE：{summary['hidden_rmse']:.4f}")
    print(f"物理一致性 RMSE：{summary['physics_rmse']:.4f}")
    print(f"实验摘要：{summary['summary_path']}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict

import yaml

from data.dataset import build_windowed_datasets, generate_synthetic_ecosystem, load_time_series_csv
from eval.rollout_eval import evaluate_rollout_model
from train.trainer import (
    Trainer,
    build_model_from_config,
    create_data_loaders,
    load_checkpoint,
    resolve_device,
    save_json,
    set_random_seed,
)
from viz.graph_viz import plot_average_gate_heatmap
from viz.heatmaps import plot_horizon_error_curve, plot_species_horizon_heatmap
from viz.trajectories import plot_rollout_vs_truth


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if key == "extends":
            continue
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(config_path: str | Path) -> Dict[str, Any]:
    config_path = Path(config_path)
    with config_path.open("r", encoding="utf-8") as handle:
        current = yaml.safe_load(handle) or {}
    extends = current.get("extends")
    if extends:
        candidate = config_path.parent / extends
        if candidate.exists():
            parent_path = candidate.resolve()
        else:
            parent_path = (Path.cwd() / extends).resolve()
        parent = load_config(parent_path)
        return deep_merge(parent, current)
    return current


def prepare_bundle(config: Dict[str, Any]):
    data_cfg = config["data"]
    synthetic_cfg = data_cfg.get("synthetic", {})
    use_synthetic = synthetic_cfg.get("enabled", False) or not data_cfg.get("path")
    if use_synthetic:
        return generate_synthetic_ecosystem(
            total_steps=int(synthetic_cfg["total_steps"]),
            num_observed=int(synthetic_cfg["num_observed"]),
            num_hidden=int(synthetic_cfg["num_hidden"]),
            noise_scale=float(synthetic_cfg["noise_scale"]),
            seed=int(config["experiment"]["seed"]),
        )
    return load_time_series_csv(
        path=data_cfg["path"],
        observed_columns=data_cfg.get("observed_columns") or None,
        covariate_columns=data_cfg.get("covariate_columns") or None,
        time_column=data_cfg.get("time_column") or None,
    )


def _without_artifacts(results: Dict[str, Any]) -> Dict[str, Any]:
    cleaned = {}
    for key, value in results.items():
        if key == "artifacts":
            continue
        if isinstance(value, dict):
            cleaned[key] = _without_artifacts(value)
        else:
            cleaned[key] = value
    return cleaned


def maybe_write_visualizations(
    config: Dict[str, Any],
    output_dir: Path,
    bundle,
    evaluation_results: Dict[str, Any],
) -> None:
    if not config.get("viz", {}).get("enabled", True):
        return

    artifacts = evaluation_results["artifacts"]
    predictions = artifacts.get("predictions_raw", artifacts["predictions_transformed"])
    targets = artifacts.get("targets_raw", artifacts["targets_transformed"])
    absolute_errors = (predictions - targets).abs()

    plot_rollout_vs_truth(
        predictions=predictions,
        targets=targets,
        species_names=bundle.observed_names,
        output_path=output_dir / "figures" / "trajectory_rollout.png",
    )
    plot_species_horizon_heatmap(
        absolute_errors=absolute_errors,
        species_names=bundle.observed_names,
        output_path=output_dir / "figures" / "species_horizon_heatmap.png",
    )
    plot_average_gate_heatmap(
        gate_history=artifacts["gate_history"],
        num_observed=bundle.num_observed,
        output_path=output_dir / "figures" / "gate_heatmap.png",
    )


def run_experiment(config_path: str | Path, output_dir_override: str | None = None) -> Dict[str, Any]:
    config = load_config(config_path)
    if output_dir_override:
        config["experiment"]["output_dir"] = output_dir_override

    output_dir = Path(config["experiment"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_random_seed(int(config["experiment"]["seed"]))

    bundle = prepare_bundle(config)
    prepared = build_windowed_datasets(
        bundle=bundle,
        history_length=int(config["data"]["history_length"]),
        horizon=int(config["data"]["horizon"]),
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
    )
    data_loaders = create_data_loaders(
        datasets=prepared["datasets"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
    )

    model = build_model_from_config(
        config=config,
        num_observed=bundle.num_observed,
        covariate_dim=bundle.covariate_dim,
    )
    trainer = Trainer(
        model=model,
        config=config,
        bounds=prepared["bounds"],
        output_dir=output_dir,
        transform_state=prepared["transform"].state_dict(),
        num_observed=bundle.num_observed,
        covariate_dim=bundle.covariate_dim,
    )

    training_results = trainer.fit(
        train_loader=data_loaders["train"],
        val_loader=data_loaders["val"],
        transform=prepared["transform"],
    )

    if training_results["best_checkpoint"]:
        model, checkpoint = load_checkpoint(training_results["best_checkpoint"], trainer.device)
    else:
        model = model.to(trainer.device)
        model.eval()

    eval_horizons = [min(int(value), int(config["data"]["horizon"])) for value in config["eval"]["horizons"]]
    val_results = evaluate_rollout_model(
        model,
        data_loaders["val"],
        device=trainer.device,
        horizons=eval_horizons,
        transform=prepared["transform"],
    )
    test_results = evaluate_rollout_model(
        model,
        data_loaders["test"],
        device=trainer.device,
        horizons=eval_horizons,
        transform=prepared["transform"],
    )

    metrics_payload = {
        "experiment": config["experiment"],
        "training": training_results,
        "validation": _without_artifacts(val_results),
        "test": _without_artifacts(test_results),
    }
    save_json(output_dir / "metrics.json", metrics_payload)
    save_json(output_dir / "training_history.json", {"history": training_results["history"]})

    plot_horizon_error_curve(
        metrics_by_model={"test": test_results["transformed"]},
        output_path=output_dir / "figures" / "rollout_rmse_curve.png",
        metric_name="rmse",
    )
    maybe_write_visualizations(
        config=config,
        output_dir=output_dir,
        bundle=bundle,
        evaluation_results=test_results,
    )

    primary_horizon = min(int(config["eval"]["primary_horizon"]), int(config["data"]["horizon"]))
    summary = {
        "output_dir": str(output_dir),
        "metrics_path": str(output_dir / "metrics.json"),
        "primary_test_rmse": test_results["transformed"][str(primary_horizon)]["rmse"],
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an eco-dynamics model.")
    parser.add_argument("--config", default="configs/base.yaml", help="Path to the YAML config file.")
    parser.add_argument("--output-dir", default=None, help="Optional output directory override.")
    args = parser.parse_args()

    summary = run_experiment(args.config, output_dir_override=args.output_dir)
    print(f"Finished. Metrics saved to {summary['metrics_path']}")
    print(f"Primary test RMSE: {summary['primary_test_rmse']:.4f}")


if __name__ == "__main__":
    main()

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from scripts.run_train import load_config, run_experiment
from train.trainer import save_json
from viz.heatmaps import plot_horizon_error_curve


DEFAULT_QUICK_CONFIGS = [
    "configs/baseline_persistence.yaml",
    "configs/baseline_obs_mlp.yaml",
    "configs/baseline_obs_graph.yaml",
    "configs/base.yaml",
]

DEFAULT_FULL_CONFIGS = DEFAULT_QUICK_CONFIGS + [
    "configs/ablation_no_hidden.yaml",
    "configs/ablation_no_delay.yaml",
    "configs/ablation_one_step_only.yaml",
]


def collect_metrics(metrics_path: str | Path) -> Dict[str, Any]:
    with Path(metrics_path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_suite_summary(experiment_summaries: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    summary: Dict[str, Any] = {"experiments": {}, "ranking_by_primary_rmse": []}
    ranking = []
    for experiment_name, payload in experiment_summaries.items():
        metrics = collect_metrics(payload["metrics_path"])
        primary_rmse = payload["primary_test_rmse"]
        ranking.append((experiment_name, primary_rmse))
        summary["experiments"][experiment_name] = {
            "output_dir": payload["output_dir"],
            "metrics_path": payload["metrics_path"],
            "primary_test_rmse": primary_rmse,
            "test_transformed": metrics["test"]["transformed"],
            "test_raw": metrics["test"].get("raw"),
            "diagnostics": metrics["test"].get("diagnostics", {}),
        }

    summary["ranking_by_primary_rmse"] = [
        {"experiment": name, "primary_test_rmse": rmse}
        for name, rmse in sorted(ranking, key=lambda item: item[1])
    ]
    return summary


def configs_for_suite(suite: str) -> List[str]:
    if suite == "quick":
        return list(DEFAULT_QUICK_CONFIGS)
    if suite == "full":
        return list(DEFAULT_FULL_CONFIGS)
    raise ValueError(f"Unsupported suite: {suite}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full eco-dynamics experiment suite.")
    parser.add_argument(
        "--suite",
        default="full",
        choices=["quick", "full"],
        help="Which experiment suite to run.",
    )
    parser.add_argument(
        "--configs",
        nargs="*",
        default=None,
        help="Optional explicit config list. If provided, it overrides --suite.",
    )
    parser.add_argument(
        "--output-dir",
        default="runs/pipeline",
        help="Directory where all experiment outputs will be stored.",
    )
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config_paths = args.configs if args.configs else configs_for_suite(args.suite)
    experiment_summaries: Dict[str, Dict[str, Any]] = {}
    comparison_metrics: Dict[str, Dict[str, Dict[str, float]]] = {}

    for config_path in config_paths:
        resolved_config = load_config(config_path)
        experiment_name = str(resolved_config["experiment"]["name"])
        experiment_output_dir = output_dir / experiment_name
        print(f"[pipeline] running {experiment_name} -> {experiment_output_dir}")
        summary = run_experiment(
            config_path=config_path,
            output_dir_override=str(experiment_output_dir),
        )
        experiment_summaries[experiment_name] = summary
        metrics = collect_metrics(summary["metrics_path"])
        comparison_metrics[experiment_name] = metrics["test"]["transformed"]

    suite_summary = build_suite_summary(experiment_summaries)
    save_json(output_dir / "suite_summary.json", suite_summary)
    plot_horizon_error_curve(
        metrics_by_model=comparison_metrics,
        output_path=output_dir / "suite_rmse_comparison.png",
        metric_name="rmse",
    )

    print(f"[pipeline] suite summary saved to {output_dir / 'suite_summary.json'}")
    if suite_summary["ranking_by_primary_rmse"]:
        best = suite_summary["ranking_by_primary_rmse"][0]
        print(
            f"[pipeline] best primary RMSE: {best['experiment']} -> "
            f"{best['primary_test_rmse']:.4f}"
        )


if __name__ == "__main__":
    main()


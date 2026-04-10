from __future__ import annotations

import argparse
from pathlib import Path

from scripts.run_train import run_experiment
from train.trainer import save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a suite of ablation experiments.")
    parser.add_argument(
        "--configs",
        nargs="*",
        default=[
            "configs/base.yaml",
            "configs/ablation_no_hidden.yaml",
            "configs/ablation_no_delay.yaml",
            "configs/ablation_one_step_only.yaml",
        ],
        help="Config files to run.",
    )
    parser.add_argument("--output-dir", default="runs/ablation_suite")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries = {}
    for config_path in args.configs:
        config_name = Path(config_path).stem
        summary = run_experiment(
            config_path=config_path,
            output_dir_override=str(output_dir / config_name),
        )
        summaries[config_name] = summary

    save_json(output_dir / "summary.json", summaries)
    print(f"Saved ablation summary to {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()

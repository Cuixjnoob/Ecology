from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict

import yaml

from data.dataset import build_windowed_datasets
from data.transforms import LogZScoreTransform
from eval.rollout_eval import evaluate_rollout_model
from scripts.run_train import _without_artifacts, load_config, prepare_bundle
from train.trainer import load_checkpoint, resolve_device, save_json


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained eco-dynamics checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint file.")
    parser.add_argument("--config", default=None, help="Optional config override path.")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    args = parser.parse_args()

    checkpoint_path = Path(args.checkpoint)
    device = resolve_device("cpu")
    model, payload = load_checkpoint(checkpoint_path, device=device)
    config: Dict[str, Any]
    if args.config:
        config = load_config(args.config)
    else:
        config = payload["config"]

    bundle = prepare_bundle(config)
    prepared = build_windowed_datasets(
        bundle=bundle,
        history_length=int(config["data"]["history_length"]),
        horizon=int(config["data"]["horizon"]),
        train_ratio=float(config["data"]["train_ratio"]),
        val_ratio=float(config["data"]["val_ratio"]),
    )

    transform = LogZScoreTransform()
    transform.load_state_dict(payload["transform_state"])

    from train.trainer import create_data_loaders

    data_loaders = create_data_loaders(
        datasets=prepared["datasets"],
        batch_size=int(config["data"]["batch_size"]),
        num_workers=int(config["data"]["num_workers"]),
    )
    horizons = [min(int(value), int(config["data"]["horizon"])) for value in config["eval"]["horizons"]]
    results = evaluate_rollout_model(
        model,
        data_loaders[args.split],
        device=device,
        horizons=horizons,
        transform=transform,
    )

    output_path = checkpoint_path.parent / f"eval_{args.split}.json"
    save_json(output_path, _without_artifacts(results))
    print(f"Saved evaluation metrics to {output_path}")


if __name__ == "__main__":
    main()


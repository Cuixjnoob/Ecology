from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from eval.baselines import ObservedMLPBaseline, PersistenceBaseline
from eval.rollout_eval import evaluate_rollout_model
from losses.ecological import (
    direct_latent_balance_penalty,
    metabolic_prior_loss,
    range_penalty,
    smoothness_penalty,
    sparsity_penalty,
)
from losses.prediction import one_step_loss
from losses.rollout import rollout_loss
from models.full_model import EcoDynamicsModel


@dataclass
class CurriculumState:
    rollout_horizon: int
    teacher_forcing_ratio: float


def resolve_device(device_name: str) -> torch.device:
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(
    datasets: Dict[str, torch.utils.data.Dataset],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def build_model_from_config(
    config: Dict[str, object],
    num_observed: int,
    covariate_dim: int,
) -> nn.Module:
    model_type = config["experiment"].get("model_type", "full")
    model_cfg = config["model"]
    data_cfg = config["data"]

    if model_type == "persistence":
        return PersistenceBaseline()
    if model_type == "obs_mlp":
        return ObservedMLPBaseline(
            num_observed=num_observed,
            covariate_dim=covariate_dim,
            delay_length=int(data_cfg["delay_length"]),
            delay_stride=int(data_cfg["delay_stride"]),
            hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
        )
    if model_type in {"full", "obs_graph"}:
        num_hidden_nodes = int(model_cfg["num_hidden_nodes"])
        if model_type == "obs_graph":
            num_hidden_nodes = 0
        return EcoDynamicsModel(
            num_observed=num_observed,
            covariate_dim=covariate_dim,
            delay_length=int(data_cfg["delay_length"]),
            delay_stride=int(data_cfg["delay_stride"]),
            embedding_dim=int(model_cfg["embedding_dim"]),
            global_dim=int(model_cfg["global_dim"]),
            edge_hidden_dim=int(model_cfg["edge_hidden_dim"]),
            num_message_passing_layers=int(model_cfg["num_message_passing_layers"]),
            num_hidden_nodes=num_hidden_nodes,
            decoder_hidden_dim=int(model_cfg["decoder_hidden_dim"]),
            dropout=float(model_cfg["dropout"]),
            use_layer_norm=bool(model_cfg["use_layer_norm"]),
            use_latent_recurrence=bool(model_cfg.get("use_latent_recurrence", True)),
            use_species_embeddings=bool(model_cfg.get("use_species_embeddings", True)),
            max_log_delta=float(model_cfg.get("max_log_delta", 0.35)),
        )
    raise ValueError(f"Unsupported model type: {model_type}")


def build_optimizer(model: nn.Module, train_config: Dict[str, object]) -> torch.optim.Optimizer | None:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not parameters:
        return None
    return AdamW(
        parameters,
        lr=float(train_config["learning_rate"]),
        weight_decay=float(train_config["weight_decay"]),
    )


def save_json(path: str | Path, payload: Dict[str, object]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        config: Dict[str, object],
        bounds: Dict[str, torch.Tensor],
        output_dir: str | Path,
        transform_state: Dict[str, object],
        num_observed: int,
        covariate_dim: int,
    ) -> None:
        self.model = model
        self.config = config
        self.loss_cfg = config["loss"]
        self.train_cfg = config["train"]
        self.eval_cfg = config["eval"]
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = resolve_device(str(self.train_cfg["device"]))
        self.model.to(self.device)
        self.optimizer = build_optimizer(model, self.train_cfg)
        self.bounds = {
            "min": bounds["min"].to(self.device),
            "max": bounds["max"].to(self.device),
        }
        self.transform_state = transform_state
        self.num_observed = num_observed
        self.covariate_dim = covariate_dim
        self.checkpoint_path = self.output_dir / str(self.train_cfg["checkpoint_name"])
        if hasattr(self.model, "set_transform_stats"):
            self.model.set_transform_stats(
                mean=self.transform_state["mean"].to(self.device),
                std=self.transform_state["std"].to(self.device),
            )

    def _curriculum_stage_from_epoch(self, epoch: int) -> tuple[int, Dict[str, object]] | None:
        curriculum_cfg = self.train_cfg.get("curriculum")
        if not curriculum_cfg:
            return None
        stage_epochs = [int(value) for value in curriculum_cfg["stage_epochs"]]
        cumulative = 0
        for stage_index, num_epochs in enumerate(stage_epochs):
            cumulative += num_epochs
            if epoch < cumulative:
                return stage_index, curriculum_cfg
        return len(stage_epochs) - 1, curriculum_cfg

    def curriculum_for_epoch(self, epoch: int, dataset_horizon: int) -> CurriculumState:
        curriculum_stage = self._curriculum_stage_from_epoch(epoch)
        if curriculum_stage is not None:
            stage_index, curriculum_cfg = curriculum_stage
            horizons = [min(int(value), dataset_horizon) for value in curriculum_cfg["horizons"]]
            teacher_forcing = [float(value) for value in curriculum_cfg["teacher_forcing"]]
            return CurriculumState(
                rollout_horizon=horizons[stage_index],
                teacher_forcing_ratio=teacher_forcing[stage_index],
            )

        stage_a_epochs = int(self.train_cfg["stage_a_epochs"])
        stage_b_epochs = int(self.train_cfg["stage_b_epochs"])
        stage_b_horizon = min(int(self.train_cfg["stage_b_horizon"]), dataset_horizon)
        max_rollout_horizon = min(int(self.train_cfg["max_rollout_horizon"]), dataset_horizon)

        if epoch < stage_a_epochs:
            return CurriculumState(rollout_horizon=1, teacher_forcing_ratio=1.0)

        if epoch < stage_a_epochs + stage_b_epochs and stage_b_epochs > 0:
            progress_epoch = epoch - stage_a_epochs
            if stage_b_epochs == 1:
                rollout_horizon = stage_b_horizon
            else:
                progress = progress_epoch / max(stage_b_epochs - 1, 1)
                rollout_horizon = round(2 + progress * max(stage_b_horizon - 2, 0))
            rollout_horizon = max(2, min(rollout_horizon, stage_b_horizon))
            return CurriculumState(
                rollout_horizon=rollout_horizon,
                teacher_forcing_ratio=float(self.train_cfg["teacher_forcing_ratio_stage_b"]),
            )

        return CurriculumState(rollout_horizon=max_rollout_horizon, teacher_forcing_ratio=0.0)

    def update_learning_rate_for_epoch(self, epoch: int) -> None:
        if self.optimizer is None:
            return
        curriculum_stage = self._curriculum_stage_from_epoch(epoch)
        if curriculum_stage is None:
            return
        stage_index, curriculum_cfg = curriculum_stage
        lr_multipliers = curriculum_cfg.get("lr_multipliers")
        if not lr_multipliers:
            return
        base_lr = float(self.train_cfg["learning_rate"])
        new_lr = base_lr * float(lr_multipliers[stage_index])
        for parameter_group in self.optimizer.param_groups:
            parameter_group["lr"] = new_lr

    def compute_loss_bundle(
        self,
        outputs: Dict[str, torch.Tensor],
        future_targets: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        one_step = one_step_loss(
            outputs["predictions"][:, 0, :],
            future_targets[:, 0, :],
            loss_type=str(self.loss_cfg["one_step"]),
        )
        rollout_value = rollout_loss(
            outputs["predictions"],
            future_targets,
            loss_type=str(self.loss_cfg["one_step"]),
        )
        range_value = range_penalty(
            outputs["predictions"],
            min_values=self.bounds["min"],
            max_values=self.bounds["max"],
            margin=float(self.loss_cfg["range_margin"]),
        )
        sparse_value = sparsity_penalty(outputs["gate_history"])
        smooth_value = smoothness_penalty(outputs["deltas"])
        meta_value = metabolic_prior_loss(outputs["hidden_activity"])
        balance_value = direct_latent_balance_penalty(
            outputs["direct_deltas"],
            outputs["latent_deltas"],
            target_min_ratio=float(self.loss_cfg.get("balance_target_min_ratio", 0.15)),
        )

        total = (
            float(self.loss_cfg["lambda_1step"]) * one_step
            + float(self.loss_cfg["lambda_rollout"]) * rollout_value
            + float(self.loss_cfg["lambda_range"]) * range_value
            + float(self.loss_cfg["lambda_sparse"]) * sparse_value
            + float(self.loss_cfg["lambda_meta"]) * meta_value
            + float(self.loss_cfg["lambda_smooth"]) * smooth_value
            + float(self.loss_cfg.get("lambda_balance", 0.0)) * balance_value
        )

        return {
            "total": total,
            "one_step": one_step,
            "rollout": rollout_value,
            "range": range_value,
            "sparse": sparse_value,
            "smooth": smooth_value,
            "meta": meta_value,
            "balance": balance_value,
        }

    def _run_epoch(
        self,
        data_loader: DataLoader,
        curriculum: CurriculumState,
        training: bool,
    ) -> Dict[str, float]:
        if training:
            self.model.train()
        else:
            self.model.eval()

        aggregates = {
            "total": 0.0,
            "one_step": 0.0,
            "rollout": 0.0,
            "range": 0.0,
            "sparse": 0.0,
            "smooth": 0.0,
            "meta": 0.0,
            "balance": 0.0,
        }
        num_batches = 0

        for batch in data_loader:
            history = batch["history"].to(self.device)
            future = batch["future"][:, : curriculum.rollout_horizon].to(self.device)
            history_u = batch["history_u"].to(self.device)
            future_u = batch["future_u"][:, : curriculum.rollout_horizon].to(self.device)

            with torch.set_grad_enabled(training):
                outputs = self.model(
                    history_x=history,
                    history_u=history_u,
                    future_u=future_u,
                    rollout_horizon=curriculum.rollout_horizon,
                    teacher_forcing_targets=future if curriculum.teacher_forcing_ratio > 0.0 else None,
                    teacher_forcing_ratio=curriculum.teacher_forcing_ratio,
                )
                losses = self.compute_loss_bundle(outputs, future)

                if training and self.optimizer is not None:
                    self.optimizer.zero_grad()
                    losses["total"].backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        float(self.train_cfg["grad_clip"]),
                    )
                    self.optimizer.step()

            for key in aggregates:
                aggregates[key] += float(losses[key].detach().cpu().item())
            num_batches += 1

        if num_batches == 0:
            raise RuntimeError("Encountered an empty dataloader during training.")
        return {key: value / num_batches for key, value in aggregates.items()}

    def save_checkpoint(self, epoch: int, metric: float) -> None:
        optimizer_state = self.optimizer.state_dict() if self.optimizer is not None else None
        torch.save(
            {
                "epoch": epoch,
                "metric": metric,
                "model_state": self.model.state_dict(),
                "optimizer_state": optimizer_state,
                "config": self.config,
                "transform_state": self.transform_state,
                "metadata": {
                    "num_observed": self.num_observed,
                    "covariate_dim": self.covariate_dim,
                },
            },
            self.checkpoint_path,
        )

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        transform,
    ) -> Dict[str, object]:
        if self.optimizer is None:
            return {
                "history": [],
                "best_metric": None,
                "best_checkpoint": None,
            }

        dataset_horizon = int(self.config["data"]["horizon"])
        primary_horizon = min(int(self.eval_cfg["primary_horizon"]), dataset_horizon)
        best_metric = float("inf")
        patience = int(self.train_cfg["early_stopping_patience"])
        patience_counter = 0
        history = []

        for epoch in range(int(self.train_cfg["epochs"])):
            self.update_learning_rate_for_epoch(epoch)
            curriculum = self.curriculum_for_epoch(epoch, dataset_horizon)
            train_metrics = self._run_epoch(train_loader, curriculum, training=True)
            val_results = evaluate_rollout_model(
                self.model,
                val_loader,
                device=self.device,
                horizons=[primary_horizon],
                transform=transform,
            )
            val_metric = float(val_results["transformed"][str(primary_horizon)]["rmse"])

            epoch_record = {
                "epoch": epoch,
                "train": train_metrics,
                "val_primary_rmse": val_metric,
                "rollout_horizon": curriculum.rollout_horizon,
                "teacher_forcing_ratio": curriculum.teacher_forcing_ratio,
                "learning_rate": self.optimizer.param_groups[0]["lr"],
            }
            history.append(epoch_record)

            if val_metric < best_metric:
                best_metric = val_metric
                patience_counter = 0
                self.save_checkpoint(epoch=epoch, metric=val_metric)
            else:
                patience_counter += 1

            if patience_counter >= patience:
                break

        return {
            "history": history,
            "best_metric": best_metric,
            "best_checkpoint": str(self.checkpoint_path),
        }


def load_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Tuple[nn.Module, Dict[str, object]]:
    payload = torch.load(checkpoint_path, map_location=device)
    metadata = payload["metadata"]
    model = build_model_from_config(
        config=payload["config"],
        num_observed=int(metadata["num_observed"]),
        covariate_dim=int(metadata["covariate_dim"]),
    )
    model.load_state_dict(payload["model_state"])
    model.to(device)
    if hasattr(model, "set_transform_stats"):
        model.set_transform_stats(
            mean=payload["transform_state"]["mean"].to(device),
            std=payload["transform_state"]["std"].to(device),
        )
    model.eval()
    return model, payload

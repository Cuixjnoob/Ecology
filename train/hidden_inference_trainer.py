from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
from torch import nn
from torch.optim import AdamW

from data.transforms import LogZScoreTransform
from models.hidden_inference_model import HiddenSpeciesInferenceModel


@dataclass
class HiddenInferenceEvaluation:
    metrics: Dict[str, float]
    artifacts: Dict[str, torch.Tensor]


def _pearson_mean(predictions: torch.Tensor, targets: torch.Tensor) -> float:
    pred_centered = predictions - predictions.mean(dim=0, keepdim=True)
    target_centered = targets - targets.mean(dim=0, keepdim=True)
    denominator = torch.sqrt(
        pred_centered.square().sum(dim=0) * target_centered.square().sum(dim=0)
    ).clamp_min(1e-8)
    values = (pred_centered * target_centered).sum(dim=0) / denominator
    return float(values.mean().item())


class HiddenInferenceTrainer:
    def __init__(
        self,
        model: HiddenSpeciesInferenceModel,
        output_dir: str | Path,
        device: torch.device,
        observed_transform: LogZScoreTransform,
        hidden_transform: LogZScoreTransform,
        growth_rates: torch.Tensor,
        interaction_matrix: torch.Tensor,
        dt: float,
        num_observed: int,
        config: Dict[str, object],
    ) -> None:
        self.model = model.to(device)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.device = device
        self.observed_transform = observed_transform
        self.hidden_transform = hidden_transform
        self.hidden_mean = hidden_transform.mean.to(device)
        self.hidden_std = hidden_transform.std.to(device)
        self.growth_rates = growth_rates.to(device)
        self.interaction_matrix = interaction_matrix.to(device)
        self.dt = float(dt)
        self.num_observed = num_observed
        self.config = config
        self.train_cfg = config["train"]
        self.loss_cfg = config["loss"]
        self.checkpoint_path = self.output_dir / str(self.train_cfg["checkpoint_name"])
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=float(self.train_cfg["learning_rate"]),
            weight_decay=float(self.train_cfg["weight_decay"]),
        )

    def hidden_to_standardized(self, hidden_raw: torch.Tensor) -> torch.Tensor:
        log_values = torch.log1p(hidden_raw.clamp_min(0.0))
        return (log_values - self.hidden_mean) / (self.hidden_std + self.hidden_transform.eps)

    def hidden_to_raw(self, hidden_standardized: torch.Tensor) -> torch.Tensor:
        log_values = hidden_standardized * (self.hidden_std + self.hidden_transform.eps) + self.hidden_mean
        return torch.expm1(log_values).clamp_min(0.0)

    def hidden_physics_step(
        self,
        observed_current_raw: torch.Tensor,
        hidden_current_raw: torch.Tensor,
    ) -> torch.Tensor:
        hidden_index = self.num_observed
        hidden_growth = self.growth_rates[hidden_index]
        observed_weights = self.interaction_matrix[hidden_index, : self.num_observed]
        hidden_weight = self.interaction_matrix[hidden_index, hidden_index]

        def rhs(hidden_value: torch.Tensor) -> torch.Tensor:
            drive = hidden_growth + observed_current_raw @ observed_weights.unsqueeze(-1)
            drive = drive.squeeze(-1) + hidden_weight * hidden_value.squeeze(-1)
            return hidden_value.squeeze(-1) * drive

        h0 = hidden_current_raw.squeeze(-1)
        k1 = rhs(hidden_current_raw)
        k2 = rhs((h0 + 0.5 * self.dt * k1).unsqueeze(-1))
        k3 = rhs((h0 + 0.5 * self.dt * k2).unsqueeze(-1))
        k4 = rhs((h0 + self.dt * k3).unsqueeze(-1))
        next_hidden = h0 + (self.dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
        return next_hidden.unsqueeze(-1).clamp_min(1e-5)

    def compute_loss(
        self,
        batch: Dict[str, torch.Tensor],
        outputs: Dict[str, torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        hidden_current_raw = batch["history_hidden"][:, -1, :].to(self.device)
        hidden_next_raw = batch["future_hidden"][:, 0, :].to(self.device)
        observed_current_raw = batch["history_raw"][:, -1, :].to(self.device)

        hidden_current_target = self.hidden_to_standardized(hidden_current_raw)
        hidden_pred_standardized = outputs["hidden_standardized"]
        hidden_pred_raw = self.hidden_to_raw(hidden_pred_standardized)
        hidden_next_physics = self.hidden_physics_step(observed_current_raw, hidden_pred_raw)

        hidden_state_loss = torch.nn.functional.mse_loss(
            hidden_pred_standardized,
            hidden_current_target,
        )
        physics_loss = torch.nn.functional.mse_loss(
            torch.log1p(hidden_next_physics),
            torch.log1p(hidden_next_raw),
        )
        sparsity_loss = outputs["gates"].mean()

        total = (
            float(self.loss_cfg["lambda_hidden_state"]) * hidden_state_loss
            + float(self.loss_cfg["lambda_physics"]) * physics_loss
            + float(self.loss_cfg["lambda_sparse"]) * sparsity_loss
        )

        return {
            "total": total,
            "hidden_state": hidden_state_loss,
            "physics": physics_loss,
            "sparse": sparsity_loss,
            "hidden_pred_raw": hidden_pred_raw,
            "hidden_next_physics": hidden_next_physics,
        }

    def run_epoch(self, data_loader, training: bool) -> Dict[str, float]:
        self.model.train(training)
        totals = {"total": 0.0, "hidden_state": 0.0, "physics": 0.0, "sparse": 0.0}
        num_batches = 0

        for batch in data_loader:
            history = batch["history"].to(self.device)
            history_u = batch["history_u"].to(self.device)
            outputs = self.model(history_x=history, history_u=history_u)
            losses = self.compute_loss(batch, outputs)

            if training:
                self.optimizer.zero_grad()
                losses["total"].backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    float(self.train_cfg["grad_clip"]),
                )
                self.optimizer.step()

            for key in totals:
                totals[key] += float(losses[key].detach().cpu().item())
            num_batches += 1

        return {key: value / max(num_batches, 1) for key, value in totals.items()}

    @torch.no_grad()
    def evaluate(self, data_loader) -> HiddenInferenceEvaluation:
        self.model.eval()
        hidden_true = []
        hidden_pred = []
        hidden_next_true = []
        hidden_next_physics = []
        gate_history = []
        window_end = []

        for batch in data_loader:
            history = batch["history"].to(self.device)
            history_u = batch["history_u"].to(self.device)
            outputs = self.model(history_x=history, history_u=history_u)
            losses = self.compute_loss(batch, outputs)

            hidden_true.append(batch["history_hidden"][:, -1, :].cpu())
            hidden_pred.append(losses["hidden_pred_raw"].cpu())
            hidden_next_true.append(batch["future_hidden"][:, 0, :].cpu())
            hidden_next_physics.append(losses["hidden_next_physics"].cpu())
            gate_history.append(outputs["gates"].cpu())
            window_end.append(batch["window_end_index"].cpu())

        hidden_true_tensor = torch.cat(hidden_true, dim=0)
        hidden_pred_tensor = torch.cat(hidden_pred, dim=0)
        hidden_next_true_tensor = torch.cat(hidden_next_true, dim=0)
        hidden_next_phys_tensor = torch.cat(hidden_next_physics, dim=0)
        gate_tensor = torch.cat(gate_history, dim=0)
        window_end_tensor = torch.cat(window_end, dim=0)

        metrics = {
            "hidden_rmse": float(torch.sqrt((hidden_pred_tensor - hidden_true_tensor).square().mean()).item()),
            "hidden_mae": float((hidden_pred_tensor - hidden_true_tensor).abs().mean().item()),
            "hidden_pearson": _pearson_mean(hidden_pred_tensor, hidden_true_tensor),
            "physics_rmse": float(
                torch.sqrt((hidden_next_phys_tensor - hidden_next_true_tensor).square().mean()).item()
            ),
            "physics_mae": float((hidden_next_phys_tensor - hidden_next_true_tensor).abs().mean().item()),
            "physics_pearson": _pearson_mean(hidden_next_phys_tensor, hidden_next_true_tensor),
            "mean_gate_strength": float(gate_tensor.mean().item()),
        }
        artifacts = {
            "hidden_true": hidden_true_tensor,
            "hidden_pred": hidden_pred_tensor,
            "hidden_next_true": hidden_next_true_tensor,
            "hidden_next_physics": hidden_next_phys_tensor,
            "gate_history": gate_tensor,
            "window_end_index": window_end_tensor,
        }
        return HiddenInferenceEvaluation(metrics=metrics, artifacts=artifacts)

    def save_checkpoint(self, epoch: int, metric: float) -> None:
        torch.save(
            {
                "epoch": epoch,
                "metric": metric,
                "model_state": self.model.state_dict(),
                "optimizer_state": self.optimizer.state_dict(),
                "config": self.config,
                "observed_transform": self.observed_transform.state_dict(),
                "hidden_transform": self.hidden_transform.state_dict(),
                "growth_rates": self.growth_rates.cpu(),
                "interaction_matrix": self.interaction_matrix.cpu(),
                "dt": self.dt,
                "num_observed": self.num_observed,
            },
            self.checkpoint_path,
        )

    def fit(self, train_loader, val_loader) -> Dict[str, object]:
        best_metric = float("inf")
        patience = int(self.train_cfg["early_stopping_patience"])
        patience_counter = 0
        history: List[Dict[str, object]] = []

        for epoch in range(int(self.train_cfg["epochs"])):
            train_losses = self.run_epoch(train_loader, training=True)
            val_eval = self.evaluate(val_loader)
            val_metric = val_eval.metrics["hidden_rmse"] + 0.7 * val_eval.metrics["physics_rmse"]

            epoch_record = {
                "epoch": epoch,
                "train": train_losses,
                "val_hidden_rmse": val_eval.metrics["hidden_rmse"],
                "val_physics_rmse": val_eval.metrics["physics_rmse"],
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


def load_hidden_inference_checkpoint(checkpoint_path: str | Path, device: torch.device) -> Dict[str, object]:
    return torch.load(checkpoint_path, map_location=device)

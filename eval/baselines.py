from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.encoders import MLP


class PersistenceBaseline(nn.Module):
    def forward(
        self,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
        future_u: torch.Tensor | None = None,
        rollout_horizon: int = 1,
        teacher_forcing_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        last_value = history_x[:, -1, :]
        predictions = last_value.unsqueeze(1).repeat(1, rollout_horizon, 1)
        deltas = predictions - history_x[:, -1:, :]
        batch_size, _, num_observed = predictions.shape
        gate_history = history_x.new_zeros(batch_size, rollout_horizon, 1, num_observed, num_observed)
        hidden_activity = history_x.new_zeros(batch_size, rollout_horizon)
        return {
            "predictions": predictions,
            "deltas": deltas,
            "direct_deltas": deltas,
            "latent_deltas": history_x.new_zeros(batch_size, rollout_horizon, num_observed),
            "gate_history": gate_history,
            "hidden_activity": hidden_activity,
            "latent_summary": history_x.new_zeros(batch_size, rollout_horizon, 0),
        }


class ObservedMLPBaseline(nn.Module):
    def __init__(
        self,
        num_observed: int,
        covariate_dim: int,
        delay_length: int,
        delay_stride: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_observed = num_observed
        self.covariate_dim = covariate_dim
        self.delay_length = delay_length
        self.delay_stride = delay_stride
        input_dim = num_observed * delay_length + covariate_dim
        self.mlp = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=num_observed,
            num_layers=3,
            dropout=dropout,
        )

    def _required_history_steps(self) -> int:
        return 1 + (self.delay_length - 1) * self.delay_stride

    def _build_delay_features(self, history_x: torch.Tensor) -> torch.Tensor:
        if history_x.shape[1] < self._required_history_steps():
            raise ValueError("history_x is too short for the configured delay embedding")
        time_indices = [
            history_x.shape[1] - 1 - step * self.delay_stride
            for step in range(self.delay_length)
        ]
        selected = history_x[:, time_indices, :]
        return selected.permute(0, 2, 1).reshape(history_x.shape[0], -1)

    def forward(
        self,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
        future_u: torch.Tensor | None = None,
        rollout_horizon: int = 1,
        teacher_forcing_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        batch_size, history_length, num_observed = history_x.shape
        x_buffer = history_x
        if history_u is None:
            history_u = history_x.new_zeros(batch_size, history_length, 0)
        if future_u is None:
            future_u = history_x.new_zeros(batch_size, rollout_horizon, 0)
        u_buffer = history_u

        predictions = []
        deltas = []
        for step_index in range(rollout_horizon):
            features = self._build_delay_features(x_buffer)
            current_u = u_buffer[:, -1, :] if u_buffer.shape[-1] > 0 else None
            if current_u is not None:
                features = torch.cat([features, current_u], dim=-1)
            delta_x = self.mlp(features)
            next_x = x_buffer[:, -1, :] + delta_x

            predictions.append(next_x)
            deltas.append(delta_x)

            buffer_value = next_x
            if (
                teacher_forcing_targets is not None
                and step_index < teacher_forcing_targets.shape[1]
                and teacher_forcing_ratio > 0.0
            ):
                target_value = teacher_forcing_targets[:, step_index, :]
                if teacher_forcing_ratio >= 1.0:
                    buffer_value = target_value
                else:
                    teacher_mask = (
                        torch.rand(batch_size, 1, device=history_x.device)
                        < teacher_forcing_ratio
                    )
                    buffer_value = torch.where(teacher_mask, target_value, next_x)

            x_buffer = torch.cat([x_buffer[:, 1:, :], buffer_value.unsqueeze(1)], dim=1)
            if u_buffer.shape[-1] > 0:
                next_u = future_u[:, step_index, :] if step_index < future_u.shape[1] else u_buffer[:, -1, :]
                u_buffer = torch.cat([u_buffer[:, 1:, :], next_u.unsqueeze(1)], dim=1)

        predictions_tensor = torch.stack(predictions, dim=1)
        deltas_tensor = torch.stack(deltas, dim=1)
        gate_history = history_x.new_zeros(batch_size, rollout_horizon, 1, num_observed, num_observed)
        hidden_activity = history_x.new_zeros(batch_size, rollout_horizon)
        return {
            "predictions": predictions_tensor,
            "deltas": deltas_tensor,
            "direct_deltas": deltas_tensor,
            "latent_deltas": history_x.new_zeros(batch_size, rollout_horizon, num_observed),
            "gate_history": gate_history,
            "hidden_activity": hidden_activity,
            "latent_summary": history_x.new_zeros(batch_size, rollout_horizon, 0),
        }

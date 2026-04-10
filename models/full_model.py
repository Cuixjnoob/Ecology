from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.decoder import LogGrowthDecoder
from models.encoders import GlobalContextEncoder, LatentNodeEncoder, ObservedDelayEncoder
from models.gnn import DenseMessagePassingStack
from models.graph_builder import DenseGraphBuilder


class EcoDynamicsModel(nn.Module):
    def __init__(
        self,
        num_observed: int,
        covariate_dim: int,
        delay_length: int,
        delay_stride: int,
        embedding_dim: int,
        global_dim: int,
        edge_hidden_dim: int,
        num_message_passing_layers: int,
        num_hidden_nodes: int,
        decoder_hidden_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_latent_recurrence: bool = True,
        use_species_embeddings: bool = True,
        max_log_delta: float = 0.35,
    ) -> None:
        super().__init__()
        self.num_observed = num_observed
        self.covariate_dim = covariate_dim
        self.delay_length = delay_length
        self.delay_stride = delay_stride
        self.embedding_dim = embedding_dim
        self.num_hidden_nodes = num_hidden_nodes
        self.use_latent_recurrence = use_latent_recurrence
        self.use_species_embeddings = use_species_embeddings
        self.register_buffer("transform_mean", torch.zeros(num_observed))
        self.register_buffer("transform_std", torch.ones(num_observed))

        self.observed_encoder = ObservedDelayEncoder(
            delay_length=delay_length,
            embedding_dim=embedding_dim,
            hidden_dim=embedding_dim,
            dropout=dropout,
        )
        self.global_encoder = GlobalContextEncoder(
            num_observed=num_observed,
            delay_length=delay_length,
            covariate_dim=covariate_dim,
            global_dim=global_dim,
            hidden_dim=global_dim,
            dropout=dropout,
        )
        self.latent_encoder = LatentNodeEncoder(
            global_dim=global_dim,
            embedding_dim=embedding_dim,
            num_hidden_nodes=num_hidden_nodes,
            hidden_dim=embedding_dim,
            dropout=dropout,
        )
        if use_species_embeddings:
            self.observed_species_embeddings = nn.Parameter(
                torch.zeros(num_observed, embedding_dim)
            )
        else:
            self.register_parameter("observed_species_embeddings", None)
        if num_hidden_nodes > 0 and use_latent_recurrence:
            self.latent_transition = nn.GRUCell(
                input_size=global_dim,
                hidden_size=num_hidden_nodes * embedding_dim,
            )
        else:
            self.latent_transition = None
        self.graph_builder = DenseGraphBuilder()
        self.gnn = DenseMessagePassingStack(
            num_layers=num_message_passing_layers,
            embedding_dim=embedding_dim,
            global_dim=global_dim,
            edge_hidden_dim=edge_hidden_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.decoder = LogGrowthDecoder(
            embedding_dim=embedding_dim,
            global_dim=global_dim,
            hidden_dim=decoder_hidden_dim,
            max_log_delta=max_log_delta,
            dropout=dropout,
        )

    def set_transform_stats(self, mean: torch.Tensor, std: torch.Tensor) -> None:
        if mean.shape[-1] != self.num_observed or std.shape[-1] != self.num_observed:
            raise ValueError("Transform statistics must match the observed dimension.")
        self.transform_mean.copy_(mean.detach().to(self.transform_mean.device))
        self.transform_std.copy_(std.detach().to(self.transform_std.device))

    def _required_history_steps(self) -> int:
        return 1 + (self.delay_length - 1) * self.delay_stride

    def _build_delay_features(self, history_x: torch.Tensor) -> torch.Tensor:
        if history_x.ndim != 3:
            raise ValueError("history_x must have shape [B, W, N_obs]")
        required_steps = self._required_history_steps()
        if history_x.shape[1] < required_steps:
            raise ValueError(
                f"history length {history_x.shape[1]} is too short for delay embedding "
                f"with required length {required_steps}"
            )
        time_indices = [
            history_x.shape[1] - 1 - step * self.delay_stride
            for step in range(self.delay_length)
        ]
        selected = history_x[:, time_indices, :]
        return selected.permute(0, 2, 1).contiguous()

    def step(
        self,
        history_x: torch.Tensor,
        current_u: torch.Tensor | None = None,
        latent_state: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        delay_features = self._build_delay_features(history_x)
        observed_nodes = self.observed_encoder(delay_features)
        if self.observed_species_embeddings is not None:
            observed_nodes = observed_nodes + self.observed_species_embeddings.unsqueeze(0)
        global_context = self.global_encoder(delay_features, current_u)

        if self.num_hidden_nodes > 0:
            if latent_state is None:
                hidden_nodes = self.latent_encoder(global_context)
                latent_state = hidden_nodes.reshape(history_x.shape[0], -1)
            elif self.latent_transition is not None:
                latent_state = self.latent_transition(global_context, latent_state)
                hidden_nodes = latent_state.view(
                    history_x.shape[0],
                    self.num_hidden_nodes,
                    self.embedding_dim,
                )
            else:
                hidden_nodes = latent_state.view(
                    history_x.shape[0],
                    self.num_hidden_nodes,
                    self.embedding_dim,
                )
        else:
            hidden_nodes = self.latent_encoder(global_context)

        graph_state = self.graph_builder(observed_nodes, hidden_nodes, global_context)
        graph_state, gate_history = self.gnn(graph_state)
        decoder_outputs = self.decoder(graph_state)

        current_x = history_x[:, -1, :]
        current_log = current_x * self.transform_std.unsqueeze(0) + self.transform_mean.unsqueeze(0)
        next_log = current_log + decoder_outputs["delta_log"]
        next_x = (next_log - self.transform_mean.unsqueeze(0)) / self.transform_std.unsqueeze(0)

        if self.num_hidden_nodes > 0:
            latent_nodes = graph_state.nodes[:, self.num_observed :]
            hidden_activity = latent_nodes.norm(dim=-1).mean(dim=-1)
            latent_summary = latent_nodes.reshape(history_x.shape[0], -1)
            latent_state = latent_summary
        else:
            hidden_activity = history_x.new_zeros(history_x.shape[0])
            latent_nodes = history_x.new_zeros(history_x.shape[0], 0, self.embedding_dim)
            latent_summary = history_x.new_zeros(history_x.shape[0], 0)

        return {
            "next_x": next_x,
            "delta_x": decoder_outputs["delta_log"],
            "direct_delta_x": decoder_outputs["direct_delta_log"],
            "latent_delta_x": decoder_outputs["latent_delta_log"],
            "gates": gate_history,
            "hidden_activity": hidden_activity,
            "latent_state": latent_state,
            "delay_features": delay_features,
            "latent_nodes": latent_nodes,
            "latent_summary": latent_summary,
            "latent_attention": decoder_outputs["latent_attention"],
        }

    def forward(
        self,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
        future_u: torch.Tensor | None = None,
        rollout_horizon: int = 1,
        teacher_forcing_targets: torch.Tensor | None = None,
        teacher_forcing_ratio: float = 0.0,
    ) -> Dict[str, torch.Tensor]:
        batch_size, history_length, _ = history_x.shape
        x_buffer = history_x

        if history_u is None:
            history_u = history_x.new_zeros(batch_size, history_length, 0)
        if future_u is None:
            future_u = history_x.new_zeros(batch_size, rollout_horizon, 0)
        u_buffer = history_u

        predictions = []
        deltas = []
        gate_history = []
        hidden_activity = []
        delay_features_history = []
        direct_delta_history = []
        latent_delta_history = []
        latent_nodes_history = []
        latent_summary_history = []
        latent_attention_history = []
        latent_state = None

        for step_index in range(rollout_horizon):
            current_u = u_buffer[:, -1, :] if u_buffer.shape[-1] > 0 else None
            step_outputs = self.step(
                x_buffer,
                current_u=current_u,
                latent_state=latent_state,
            )
            latent_state = step_outputs["latent_state"]
            next_x = step_outputs["next_x"]
            predictions.append(next_x)
            deltas.append(step_outputs["delta_x"])
            direct_delta_history.append(step_outputs["direct_delta_x"])
            latent_delta_history.append(step_outputs["latent_delta_x"])
            gate_history.append(step_outputs["gates"])
            hidden_activity.append(step_outputs["hidden_activity"])
            delay_features_history.append(step_outputs["delay_features"])
            latent_nodes_history.append(step_outputs["latent_nodes"])
            latent_summary_history.append(step_outputs["latent_summary"])
            latent_attention_history.append(step_outputs["latent_attention"])

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
                if step_index < future_u.shape[1]:
                    next_u = future_u[:, step_index, :]
                else:
                    next_u = u_buffer[:, -1, :]
                u_buffer = torch.cat([u_buffer[:, 1:, :], next_u.unsqueeze(1)], dim=1)

        return {
            "predictions": torch.stack(predictions, dim=1),
            "deltas": torch.stack(deltas, dim=1),
            "direct_deltas": torch.stack(direct_delta_history, dim=1),
            "latent_deltas": torch.stack(latent_delta_history, dim=1),
            "gate_history": torch.stack(gate_history, dim=1),
            "hidden_activity": torch.stack(hidden_activity, dim=1),
            "delay_features": torch.stack(delay_features_history, dim=1),
            "latent_nodes": torch.stack(latent_nodes_history, dim=1),
            "latent_summary": torch.stack(latent_summary_history, dim=1),
            "latent_attention": torch.stack(latent_attention_history, dim=1),
        }

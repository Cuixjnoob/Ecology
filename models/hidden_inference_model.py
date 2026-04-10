from __future__ import annotations

from typing import Dict

import torch
from torch import nn

from models.encoders import GlobalContextEncoder, LatentNodeEncoder, MLP, ObservedDelayEncoder
from models.gnn import DenseMessagePassingStack
from models.graph_builder import DenseGraphBuilder


class HiddenSpeciesInferenceModel(nn.Module):
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
        num_latent_nodes: int,
        decoder_hidden_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
        use_species_embeddings: bool = True,
    ) -> None:
        super().__init__()
        self.num_observed = num_observed
        self.delay_length = delay_length
        self.delay_stride = delay_stride
        self.embedding_dim = embedding_dim
        self.num_latent_nodes = num_latent_nodes

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
            num_hidden_nodes=num_latent_nodes,
            hidden_dim=embedding_dim,
            dropout=dropout,
        )
        if use_species_embeddings:
            self.observed_species_embeddings = nn.Parameter(
                torch.zeros(num_observed, embedding_dim)
            )
        else:
            self.register_parameter("observed_species_embeddings", None)

        self.graph_builder = DenseGraphBuilder()
        self.gnn = DenseMessagePassingStack(
            num_layers=num_message_passing_layers,
            embedding_dim=embedding_dim,
            global_dim=global_dim,
            edge_hidden_dim=edge_hidden_dim,
            dropout=dropout,
            use_layer_norm=use_layer_norm,
        )
        self.hidden_decoder = MLP(
            input_dim=num_latent_nodes * embedding_dim + embedding_dim + global_dim,
            hidden_dim=decoder_hidden_dim,
            output_dim=1,
            num_layers=3,
            dropout=dropout,
        )

    def _required_history_steps(self) -> int:
        return 1 + (self.delay_length - 1) * self.delay_stride

    def _build_delay_features(self, history_x: torch.Tensor) -> torch.Tensor:
        required_steps = self._required_history_steps()
        if history_x.shape[1] < required_steps:
            raise ValueError(
                f"history length {history_x.shape[1]} is too short for Takens embedding "
                f"with required length {required_steps}"
            )
        time_indices = [
            history_x.shape[1] - 1 - step * self.delay_stride
            for step in range(self.delay_length)
        ]
        selected = history_x[:, time_indices, :]
        return selected.permute(0, 2, 1).contiguous()

    def forward(
        self,
        history_x: torch.Tensor,
        history_u: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, history_length, _ = history_x.shape
        if history_u is None:
            history_u = history_x.new_zeros(batch_size, history_length, 0)

        delay_features = self._build_delay_features(history_x)
        observed_nodes = self.observed_encoder(delay_features)
        if self.observed_species_embeddings is not None:
            observed_nodes = observed_nodes + self.observed_species_embeddings.unsqueeze(0)

        global_context = self.global_encoder(delay_features, history_u[:, -1, :] if history_u.shape[-1] > 0 else None)
        latent_nodes = self.latent_encoder(global_context)

        graph_state = self.graph_builder(observed_nodes, latent_nodes, global_context)
        graph_state, gate_history = self.gnn(graph_state)

        updated_observed = graph_state.nodes[:, : self.num_observed]
        updated_latent = graph_state.nodes[:, self.num_observed :]
        observed_summary = updated_observed.mean(dim=1)
        latent_summary = updated_latent.reshape(batch_size, -1)

        decoder_inputs = torch.cat([latent_summary, observed_summary, global_context], dim=-1)
        hidden_standardized = self.hidden_decoder(decoder_inputs)

        return {
            "hidden_standardized": hidden_standardized,
            "gates": gate_history,
            "delay_features": delay_features,
            "observed_nodes": updated_observed,
            "latent_nodes": updated_latent,
            "latent_summary": latent_summary,
            "global_context": global_context,
        }

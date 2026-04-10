from __future__ import annotations

import math
from typing import Dict

import torch
from torch import nn

from models.encoders import MLP
from models.graph_builder import DenseGraphState


class LogGrowthDecoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        global_dim: int,
        hidden_dim: int,
        max_log_delta: float = 0.35,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.max_log_delta = max_log_delta

        self.direct_head = MLP(
            input_dim=embedding_dim + global_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=2,
            dropout=dropout,
        )
        self.query_proj = nn.Linear(embedding_dim, hidden_dim)
        self.key_proj = nn.Linear(embedding_dim, hidden_dim)
        self.value_proj = nn.Linear(embedding_dim, embedding_dim)
        self.latent_gate = nn.Linear(2 * embedding_dim + global_dim, 1)
        self.latent_head = MLP(
            input_dim=2 * embedding_dim + global_dim,
            hidden_dim=hidden_dim,
            output_dim=1,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, graph_state: DenseGraphState) -> Dict[str, torch.Tensor]:
        observed_nodes = graph_state.nodes[:, : graph_state.num_observed]
        batch_size = observed_nodes.shape[0]
        global_context = graph_state.global_context.unsqueeze(1).expand(
            -1,
            graph_state.num_observed,
            -1,
        )

        direct_inputs = torch.cat([observed_nodes, global_context], dim=-1)
        direct_raw = self.direct_head(direct_inputs).squeeze(-1)
        direct_delta_log = 0.5 * self.max_log_delta * torch.tanh(direct_raw)

        if graph_state.num_hidden > 0:
            hidden_nodes = graph_state.nodes[:, graph_state.num_observed :]
            queries = self.query_proj(observed_nodes)
            keys = self.key_proj(hidden_nodes)
            values = self.value_proj(hidden_nodes)

            attention_logits = torch.matmul(queries, keys.transpose(1, 2)) / math.sqrt(keys.shape[-1])
            attention_weights = torch.softmax(attention_logits, dim=-1)
            attended_hidden = torch.matmul(attention_weights, values)

            latent_inputs = torch.cat(
                [observed_nodes, attended_hidden, global_context],
                dim=-1,
            )
            latent_gate = torch.sigmoid(self.latent_gate(latent_inputs)).squeeze(-1)
            latent_raw = self.latent_head(latent_inputs).squeeze(-1)
            latent_delta_log = 0.5 * self.max_log_delta * torch.tanh(latent_raw) * latent_gate
        else:
            attended_hidden = observed_nodes.new_zeros(batch_size, graph_state.num_observed, observed_nodes.shape[-1])
            attention_weights = observed_nodes.new_zeros(batch_size, graph_state.num_observed, 0)
            latent_delta_log = observed_nodes.new_zeros(batch_size, graph_state.num_observed)

        total_delta_log = torch.clamp(
            direct_delta_log + latent_delta_log,
            min=-self.max_log_delta,
            max=self.max_log_delta,
        )
        return {
            "delta_log": total_delta_log,
            "direct_delta_log": direct_delta_log,
            "latent_delta_log": latent_delta_log,
            "latent_attention": attention_weights,
            "attended_hidden": attended_hidden,
        }

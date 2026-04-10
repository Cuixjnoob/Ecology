from __future__ import annotations

from typing import List, Tuple

import torch
from torch import nn

from models.encoders import MLP
from models.graph_builder import DenseGraphState


class DenseTypedMessagePassingLayer(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        global_dim: int,
        edge_hidden_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.edge_type_embedding = nn.Embedding(4, edge_hidden_dim)
        self.edge_mlp = MLP(
            input_dim=2 * embedding_dim + global_dim + edge_hidden_dim,
            hidden_dim=edge_hidden_dim,
            output_dim=edge_hidden_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.gate_head = nn.Linear(edge_hidden_dim, 1)
        self.message_head = nn.Linear(edge_hidden_dim, embedding_dim)
        self.update_mlp = MLP(
            input_dim=2 * embedding_dim + global_dim,
            hidden_dim=edge_hidden_dim,
            output_dim=embedding_dim,
            num_layers=2,
            dropout=dropout,
        )
        self.layer_norm = nn.LayerNorm(embedding_dim) if use_layer_norm else None

    def forward(self, graph_state: DenseGraphState) -> Tuple[DenseGraphState, torch.Tensor]:
        nodes = graph_state.nodes
        batch_size, total_nodes, _ = nodes.shape

        source_nodes = nodes.unsqueeze(2).expand(-1, -1, total_nodes, -1)
        target_nodes = nodes.unsqueeze(1).expand(-1, total_nodes, -1, -1)
        global_context = graph_state.global_context.unsqueeze(1).unsqueeze(1).expand(
            -1,
            total_nodes,
            total_nodes,
            -1,
        )
        edge_type_embeddings = self.edge_type_embedding(graph_state.type_ids)
        edge_type_embeddings = edge_type_embeddings.unsqueeze(0).expand(batch_size, -1, -1, -1)

        edge_inputs = torch.cat(
            [source_nodes, target_nodes, global_context, edge_type_embeddings],
            dim=-1,
        )
        edge_hidden = self.edge_mlp(edge_inputs)
        gates = torch.sigmoid(self.gate_head(edge_hidden)).squeeze(-1)

        self_mask = torch.eye(total_nodes, device=nodes.device, dtype=torch.bool).unsqueeze(0)
        gates = gates.masked_fill(self_mask, 0.0)

        messages = self.message_head(edge_hidden) * gates.unsqueeze(-1)
        messages = messages.masked_fill(self_mask.unsqueeze(-1), 0.0)
        aggregated = messages.sum(dim=1)

        update_inputs = torch.cat(
            [
                nodes,
                aggregated,
                graph_state.global_context.unsqueeze(1).expand(-1, total_nodes, -1),
            ],
            dim=-1,
        )
        updated_nodes = nodes + self.update_mlp(update_inputs)
        if self.layer_norm is not None:
            updated_nodes = self.layer_norm(updated_nodes)

        return (
            DenseGraphState(
                nodes=updated_nodes,
                global_context=graph_state.global_context,
                type_ids=graph_state.type_ids,
                num_observed=graph_state.num_observed,
                num_hidden=graph_state.num_hidden,
            ),
            gates,
        )


class DenseMessagePassingStack(nn.Module):
    def __init__(
        self,
        num_layers: int,
        embedding_dim: int,
        global_dim: int,
        edge_hidden_dim: int,
        dropout: float = 0.0,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.layers = nn.ModuleList(
            [
                DenseTypedMessagePassingLayer(
                    embedding_dim=embedding_dim,
                    global_dim=global_dim,
                    edge_hidden_dim=edge_hidden_dim,
                    dropout=dropout,
                    use_layer_norm=use_layer_norm,
                )
                for _ in range(num_layers)
            ]
        )

    def forward(self, graph_state: DenseGraphState) -> Tuple[DenseGraphState, torch.Tensor]:
        gate_history: List[torch.Tensor] = []
        updated_state = graph_state
        for layer in self.layers:
            updated_state, gates = layer(updated_state)
            gate_history.append(gates)
        return updated_state, torch.stack(gate_history, dim=1)


from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DenseGraphState:
    nodes: torch.Tensor
    global_context: torch.Tensor
    type_ids: torch.Tensor
    num_observed: int
    num_hidden: int


class DenseGraphBuilder(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def build_type_ids(
        num_observed: int,
        num_hidden: int,
        device: torch.device,
    ) -> torch.Tensor:
        total_nodes = num_observed + num_hidden
        node_types = torch.zeros(total_nodes, dtype=torch.long, device=device)
        if num_hidden > 0:
            node_types[num_observed:] = 1
        source_types = node_types.unsqueeze(1)
        target_types = node_types.unsqueeze(0)
        return source_types * 2 + target_types

    def forward(
        self,
        observed_nodes: torch.Tensor,
        hidden_nodes: torch.Tensor,
        global_context: torch.Tensor,
    ) -> DenseGraphState:
        num_observed = observed_nodes.shape[1]
        num_hidden = hidden_nodes.shape[1]
        nodes = torch.cat([observed_nodes, hidden_nodes], dim=1)
        type_ids = self.build_type_ids(
            num_observed=num_observed,
            num_hidden=num_hidden,
            device=nodes.device,
        )
        return DenseGraphState(
            nodes=nodes,
            global_context=global_context,
            type_ids=type_ids,
            num_observed=num_observed,
            num_hidden=num_hidden,
        )


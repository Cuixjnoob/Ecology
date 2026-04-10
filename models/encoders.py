from __future__ import annotations

from typing import List

import torch
from torch import nn


class MLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        layers: List[nn.Module] = []
        in_dim = input_dim
        for _ in range(max(num_layers - 1, 0)):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.GELU())
            if dropout > 0.0:
                layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.network(inputs)


class ObservedDelayEncoder(nn.Module):
    def __init__(
        self,
        delay_length: int,
        embedding_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.encoder = MLP(
            input_dim=delay_length,
            hidden_dim=hidden_dim,
            output_dim=embedding_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward(self, delay_features: torch.Tensor) -> torch.Tensor:
        batch_size, num_observed, delay_length = delay_features.shape
        flattened = delay_features.reshape(batch_size * num_observed, delay_length)
        encoded = self.encoder(flattened)
        return encoded.reshape(batch_size, num_observed, -1)


class GlobalContextEncoder(nn.Module):
    def __init__(
        self,
        num_observed: int,
        delay_length: int,
        covariate_dim: int,
        global_dim: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        input_dim = num_observed * delay_length + covariate_dim
        self.encoder = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=global_dim,
            num_layers=2,
            dropout=dropout,
        )

    def forward(
        self,
        delay_features: torch.Tensor,
        current_covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        flattened = delay_features.reshape(delay_features.shape[0], -1)
        if current_covariates is not None and current_covariates.shape[-1] > 0:
            flattened = torch.cat([flattened, current_covariates], dim=-1)
        return self.encoder(flattened)


class LatentNodeEncoder(nn.Module):
    def __init__(
        self,
        global_dim: int,
        embedding_dim: int,
        num_hidden_nodes: int,
        hidden_dim: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.num_hidden_nodes = num_hidden_nodes
        self.embedding_dim = embedding_dim
        if num_hidden_nodes > 0:
            self.encoder = MLP(
                input_dim=global_dim,
                hidden_dim=hidden_dim,
                output_dim=num_hidden_nodes * embedding_dim,
                num_layers=2,
                dropout=dropout,
            )
            self.identity_embeddings = nn.Parameter(
                torch.zeros(num_hidden_nodes, embedding_dim)
            )
        else:
            self.encoder = None
            self.register_parameter("identity_embeddings", None)

    def forward(self, global_context: torch.Tensor) -> torch.Tensor:
        batch_size = global_context.shape[0]
        if self.num_hidden_nodes == 0 or self.encoder is None:
            return global_context.new_zeros(batch_size, 0, self.embedding_dim)
        encoded = self.encoder(global_context).view(
            batch_size,
            self.num_hidden_nodes,
            self.embedding_dim,
        )
        return encoded + self.identity_embeddings.unsqueeze(0)


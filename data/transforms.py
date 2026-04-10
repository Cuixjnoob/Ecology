from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch


@dataclass
class TransformState:
    mean: torch.Tensor
    std: torch.Tensor
    eps: float


class LogZScoreTransform:
    def __init__(self, eps: float = 1e-6) -> None:
        self.eps = eps
        self.mean: torch.Tensor | None = None
        self.std: torch.Tensor | None = None

    @property
    def fitted(self) -> bool:
        return self.mean is not None and self.std is not None

    def fit(self, observations: torch.Tensor) -> "LogZScoreTransform":
        if observations.ndim != 2:
            raise ValueError("observations must have shape [T, N_obs]")
        log_values = self.to_log_space(observations)
        self.mean = log_values.mean(dim=0)
        self.std = log_values.std(dim=0, unbiased=False).clamp_min(self.eps)
        return self

    def transform(self, observations: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("transform must be fitted before calling transform()")
        log_values = self.to_log_space(observations)
        return self.standardize_log(log_values)

    def inverse_transform(self, transformed: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("transform must be fitted before calling inverse_transform()")
        log_values = self.destandardize(transformed)
        return self.from_log_space(log_values)

    def to_log_space(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.log1p(torch.clamp(observations, min=0.0))

    def from_log_space(self, log_values: torch.Tensor) -> torch.Tensor:
        return torch.expm1(log_values).clamp_min(0.0)

    def standardize_log(self, log_values: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("transform must be fitted before calling standardize_log()")
        return (log_values - self.mean) / (self.std + self.eps)

    def destandardize(self, transformed: torch.Tensor) -> torch.Tensor:
        if not self.fitted:
            raise RuntimeError("transform must be fitted before calling destandardize()")
        return transformed * (self.std + self.eps) + self.mean

    def state_dict(self) -> Dict[str, torch.Tensor | float]:
        if not self.fitted:
            raise RuntimeError("transform must be fitted before exporting state")
        return {"mean": self.mean, "std": self.std, "eps": self.eps}

    def load_state_dict(self, state: Dict[str, torch.Tensor | float]) -> None:
        self.mean = state["mean"]
        self.std = state["std"]
        self.eps = float(state["eps"])

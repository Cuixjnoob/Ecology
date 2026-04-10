from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch
from torch.utils.data import Dataset

from data.transforms import LogZScoreTransform
from data.window_sampler import compute_window_end_indices


@dataclass
class TimeSeriesBundle:
    observations: torch.Tensor
    covariates: torch.Tensor
    observed_names: List[str]
    covariate_names: List[str]
    timestamps: List[str]
    hidden_observations: torch.Tensor | None = None
    hidden_names: List[str] | None = None

    @property
    def num_observed(self) -> int:
        return int(self.observations.shape[1])

    @property
    def covariate_dim(self) -> int:
        return int(self.covariates.shape[1])

    @property
    def total_steps(self) -> int:
        return int(self.observations.shape[0])

    @property
    def num_hidden(self) -> int:
        if self.hidden_observations is None:
            return 0
        return int(self.hidden_observations.shape[1])


class WindowedTimeSeriesDataset(Dataset):
    def __init__(
        self,
        raw_observations: torch.Tensor,
        transformed_observations: torch.Tensor,
        covariates: torch.Tensor,
        hidden_observations: torch.Tensor | None,
        history_length: int,
        horizon: int,
        split_start: int,
        split_end: int,
    ) -> None:
        self.raw_observations = raw_observations
        self.transformed_observations = transformed_observations
        self.covariates = covariates
        self.hidden_observations = hidden_observations
        self.history_length = history_length
        self.horizon = horizon
        self.indices = compute_window_end_indices(
            split_start=split_start,
            split_end=split_end,
            history_length=history_length,
            horizon=horizon,
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        window_end = self.indices[index]
        history_start = window_end - self.history_length + 1
        future_end = window_end + self.horizon + 1

        history = self.transformed_observations[history_start : window_end + 1]
        future = self.transformed_observations[window_end + 1 : future_end]
        history_raw = self.raw_observations[history_start : window_end + 1]
        future_raw = self.raw_observations[window_end + 1 : future_end]
        history_u = self.covariates[history_start : window_end + 1]
        future_u = self.covariates[window_end + 1 : future_end]
        if self.hidden_observations is not None:
            history_hidden = self.hidden_observations[history_start : window_end + 1]
            future_hidden = self.hidden_observations[window_end + 1 : future_end]
        else:
            history_hidden = torch.zeros(self.history_length, 0, dtype=torch.float32)
            future_hidden = torch.zeros(self.horizon, 0, dtype=torch.float32)

        return {
            "history": history,
            "future": future,
            "history_raw": history_raw,
            "future_raw": future_raw,
            "history_u": history_u,
            "future_u": future_u,
            "history_hidden": history_hidden,
            "future_hidden": future_hidden,
            "window_end_index": torch.tensor(window_end, dtype=torch.long),
        }


def _infer_columns(
    header: Sequence[str],
    time_column: str | None,
    observed_columns: Sequence[str] | None,
    covariate_columns: Sequence[str] | None,
) -> tuple[List[str], List[str]]:
    covariates = list(covariate_columns or [])
    if observed_columns:
        observed = list(observed_columns)
    else:
        blocked = set(covariates)
        if time_column:
            blocked.add(time_column)
        observed = [column for column in header if column not in blocked]
    if not observed:
        raise ValueError("No observed columns were found in the input CSV.")
    return observed, covariates


def load_time_series_csv(
    path: str | Path,
    observed_columns: Sequence[str] | None = None,
    covariate_columns: Sequence[str] | None = None,
    time_column: str | None = None,
) -> TimeSeriesBundle:
    path = Path(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError("CSV file must include a header row.")
        observed_names, covariate_names = _infer_columns(
            reader.fieldnames,
            time_column=time_column,
            observed_columns=observed_columns,
            covariate_columns=covariate_columns,
        )

        observed_rows: List[List[float]] = []
        covariate_rows: List[List[float]] = []
        timestamps: List[str] = []

        for row in reader:
            if time_column:
                timestamps.append(row[time_column])
            else:
                timestamps.append(str(len(timestamps)))
            observed_rows.append([float(row[column]) for column in observed_names])
            covariate_rows.append([float(row[column]) for column in covariate_names])

    observations = torch.tensor(observed_rows, dtype=torch.float32)
    if covariate_rows and covariate_names:
        covariates = torch.tensor(covariate_rows, dtype=torch.float32)
    else:
        covariates = torch.zeros(observations.shape[0], 0, dtype=torch.float32)
    return TimeSeriesBundle(
        observations=observations,
        covariates=covariates,
        observed_names=observed_names,
        covariate_names=covariate_names,
        timestamps=timestamps,
        hidden_observations=None,
        hidden_names=None,
    )


def generate_synthetic_ecosystem(
    total_steps: int,
    num_observed: int,
    num_hidden: int = 1,
    noise_scale: float = 0.02,
    seed: int = 42,
) -> TimeSeriesBundle:
    generator = torch.Generator().manual_seed(seed)
    observations = torch.zeros(total_steps, num_observed, dtype=torch.float32)
    hidden = torch.zeros(total_steps, num_hidden, dtype=torch.float32)

    observations[0] = 0.6 + torch.rand(num_observed, generator=generator)
    hidden[0] = 0.4 + 0.2 * torch.rand(num_hidden, generator=generator)

    obs_growth = torch.linspace(0.02, 0.06, num_observed)
    obs_decay = torch.linspace(0.03, 0.08, num_observed)
    obs_to_obs = 0.08 * torch.randn(num_observed, num_observed, generator=generator)
    obs_to_obs.fill_diagonal_(-0.05)
    hidden_to_obs = 0.12 * torch.randn(num_hidden, num_observed, generator=generator)
    obs_to_hidden = 0.10 * torch.randn(num_observed, num_hidden, generator=generator)
    hidden_to_hidden = -0.05 * torch.eye(num_hidden)

    for time_index in range(total_steps - 1):
        x_t = observations[time_index]
        h_t = hidden[time_index]

        obs_drive = x_t @ obs_to_obs.t() + h_t @ hidden_to_obs
        hidden_drive = x_t @ obs_to_hidden + h_t @ hidden_to_hidden.t()

        dx = x_t * (obs_growth + obs_drive) - obs_decay * x_t.square()
        dh = 0.08 * (0.5 - h_t) + 0.1 * hidden_drive

        x_noise = noise_scale * torch.randn(num_observed, generator=generator)
        h_noise = noise_scale * 0.5 * torch.randn(num_hidden, generator=generator)

        observations[time_index + 1] = torch.clamp(x_t + dx + x_noise, min=1e-4)
        hidden[time_index + 1] = torch.clamp(h_t + dh + h_noise, min=1e-4)

    return TimeSeriesBundle(
        observations=observations,
        covariates=torch.zeros(total_steps, 0, dtype=torch.float32),
        observed_names=[f"species_{idx}" for idx in range(num_observed)],
        covariate_names=[],
        timestamps=[str(idx) for idx in range(total_steps)],
        hidden_observations=hidden,
        hidden_names=[f"hidden_{idx}" for idx in range(num_hidden)],
    )


def split_time_series(
    total_steps: int,
    train_ratio: float,
    val_ratio: float,
) -> Dict[str, tuple[int, int]]:
    if not 0.0 < train_ratio < 1.0:
        raise ValueError("train_ratio must be between 0 and 1.")
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError("val_ratio must be between 0 and 1.")

    train_end = int(total_steps * train_ratio)
    val_end = int(total_steps * (train_ratio + val_ratio))
    val_end = min(max(val_end, train_end + 1), total_steps)
    train_end = max(train_end, 1)

    return {
        "train": (0, train_end),
        "val": (train_end, val_end),
        "test": (val_end, total_steps),
    }


def build_windowed_datasets(
    bundle: TimeSeriesBundle,
    history_length: int,
    horizon: int,
    train_ratio: float,
    val_ratio: float,
    transform: LogZScoreTransform | None = None,
) -> Dict[str, object]:
    transform = transform or LogZScoreTransform()
    splits = split_time_series(
        total_steps=bundle.total_steps,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
    )
    train_start, train_end = splits["train"]
    transform.fit(bundle.observations[train_start:train_end])
    transformed = transform.transform(bundle.observations)

    datasets = {}
    for split_name, (split_start, split_end) in splits.items():
        datasets[split_name] = WindowedTimeSeriesDataset(
            raw_observations=bundle.observations,
            transformed_observations=transformed,
            covariates=bundle.covariates,
            hidden_observations=bundle.hidden_observations,
            history_length=history_length,
            horizon=horizon,
            split_start=split_start,
            split_end=split_end,
        )

    transformed_train = transformed[train_start:train_end]
    bounds = {
        "min": transformed_train.min(dim=0).values,
        "max": transformed_train.max(dim=0).values,
    }

    return {
        "bundle": bundle,
        "transform": transform,
        "splits": splits,
        "datasets": datasets,
        "bounds": bounds,
    }

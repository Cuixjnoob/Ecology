from __future__ import annotations

from copy import deepcopy
from typing import Dict


def make_rollout_config(config: Dict[str, object], max_rollout_horizon: int | None = None) -> Dict[str, object]:
    updated = deepcopy(config)
    if max_rollout_horizon is not None:
        updated["train"]["max_rollout_horizon"] = max_rollout_horizon
    updated["loss"]["lambda_rollout"] = max(float(updated["loss"]["lambda_rollout"]), 0.5)
    return updated

from __future__ import annotations

from copy import deepcopy
from typing import Dict


def make_one_step_config(config: Dict[str, object]) -> Dict[str, object]:
    updated = deepcopy(config)
    updated["loss"]["lambda_rollout"] = 0.0
    updated["train"]["stage_a_epochs"] = int(updated["train"]["epochs"])
    updated["train"]["stage_b_epochs"] = 0
    updated["train"]["max_rollout_horizon"] = 1
    return updated


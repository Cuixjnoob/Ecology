from __future__ import annotations

from typing import List


def compute_window_end_indices(
    split_start: int,
    split_end: int,
    history_length: int,
    horizon: int,
) -> List[int]:
    if split_end <= split_start:
        return []
    first_end = split_start + history_length - 1
    last_end = split_end - horizon - 1
    if last_end < first_end:
        return []
    return list(range(first_end, last_end + 1))


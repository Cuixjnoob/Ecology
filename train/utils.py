"""训练相关的工具函数。

从旧版 train/trainer.py 中提取的公共函数，供主线代码使用。
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader


def resolve_device(device_name: str) -> torch.device:
    """根据名称解析设备，若 CUDA 不可用则自动回退到 CPU。"""
    if device_name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_name)


def set_random_seed(seed: int) -> None:
    """设置 PyTorch 全局随机种子（CPU + CUDA）。"""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def create_data_loaders(
    datasets: Dict[str, torch.utils.data.Dataset],
    batch_size: int,
    num_workers: int,
) -> Dict[str, DataLoader]:
    """为 train / val / test 三个子集分别构建 DataLoader。"""
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        ),
    }


def save_json(path: str | Path, payload: Dict[str, object]) -> None:
    """将字典以 JSON 格式保存到文件，自动创建父目录。"""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)

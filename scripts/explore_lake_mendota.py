"""Lake Mendota Phytoplankton 数据探索 + 月度聚合 + top-K 物种选择.

目的: 决定 hidden 物种与 visible 组合, 类似 Portal 的 top-12 + OT 做法.
"""
from __future__ import annotations

import csv
from collections import defaultdict
from datetime import datetime

import numpy as np


def load_and_aggregate():
    """返回 (月度丰度矩阵, 所选 top-K 物种, 月份列表)."""
    # 按 (年, 月, 物种) 累积 biomass 值
    records = defaultdict(lambda: defaultdict(float))
    all_taxa = defaultdict(float)  # 整体丰度排序用

    with open("data/real_datasets/lake_mendota_phytoplankton.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["lakeid"] != "ME":
                continue
            # 规范化物种名（去头尾空格 + 去多余空格）
            taxa = " ".join(row["taxa_name"].split())
            try:
                year = int(row["year4"])
                date = row["sampledate"]
                month = int(date.split("-")[1])
                biomass = float(row["biomass_conc"]) if row["biomass_conc"] else 0
            except (ValueError, IndexError, KeyError):
                continue
            if biomass <= 0:
                continue
            records[(year, month)][taxa] += biomass
            all_taxa[taxa] += biomass

    # 累计覆盖度排序
    total = sum(all_taxa.values())
    sorted_taxa = sorted(all_taxa.items(), key=lambda x: -x[1])
    print(f"总 biomass: {total:.2e}")
    print(f"\n累计覆盖度 top-30:")
    cum = 0
    for i, (tx, v) in enumerate(sorted_taxa[:30]):
        cum += v
        print(f"  {i+1:3d}. {tx[:50]:<50s}  pct={100*v/total:5.2f}%  cum={100*cum/total:5.2f}%")

    # 找到累计覆盖 ≥ 95% 需要多少个物种
    cum = 0
    n_95 = 0
    for v in sorted(all_taxa.values(), reverse=True):
        cum += v
        n_95 += 1
        if cum >= 0.95 * total:
            break
    print(f"\n累计覆盖 95% 需要 top-{n_95} 个物种")

    cum = 0
    n_90 = 0
    for v in sorted(all_taxa.values(), reverse=True):
        cum += v
        n_90 += 1
        if cum >= 0.90 * total:
            break
    print(f"累计覆盖 90% 需要 top-{n_90} 个物种")

    # 按月聚合
    all_ym = sorted(records.keys())
    ym_span = (all_ym[0], all_ym[-1])
    n_months_actual = len(all_ym)
    print(f"\n采样月度跨度: {ym_span[0]} → {ym_span[1]}, 实际 {n_months_actual} 个月")

    # 选择 top-12 物种（与 Portal 对齐）
    top_k = 12
    top_taxa = [tx for tx, _ in sorted_taxa[:top_k]]
    print(f"\n选定 top-{top_k} 物种作为近似群落:")
    for i, tx in enumerate(top_taxa):
        print(f"  {i+1:2d}. {tx}")

    # 构造 (T, top_k) 矩阵 - 按年月顺序
    T = len(all_ym)
    matrix = np.zeros((T, top_k), dtype=np.float32)
    for t, ym in enumerate(all_ym):
        for j, tx in enumerate(top_taxa):
            matrix[t, j] = records[ym].get(tx, 0.0)

    # 统计每物种的非零月份比例 + 平均值
    print(f"\n每个 top-{top_k} 物种的出现率与规模:")
    for j, tx in enumerate(top_taxa):
        nonzero = (matrix[:, j] > 0).sum()
        mean_val = matrix[:, j].mean()
        max_val = matrix[:, j].max()
        print(f"  {tx[:40]:<40s}  出现月={nonzero}/{T}  mean={mean_val:.4f}  max={max_val:.4f}")

    return matrix, top_taxa, all_ym


if __name__ == "__main__":
    matrix, taxa, ym = load_and_aggregate()
    print(f"\n最终矩阵 shape: {matrix.shape}")

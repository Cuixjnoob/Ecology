"""5 物种 vs 6 物种正向模拟对比。

目的: 量化"那一个额外物种（原 hidden）"对其他 5 个物种的动力学影响有多大。

实验设计:
  A 系统: 5 物种纯 LV，参数取自 6 物种系统的前 5 维，不包含 hidden 的交互
  B 系统: 6 物种完整 LV（当前数据生成器的默认）

两个系统使用:
  - 完全相同的 growth rates（前 5 维）
  - 完全相同的 interaction matrix（A 是 B 的 [:5, :5] 子矩阵）
  - 完全相同的 env/pulse 时间序列
  - 完全相同的初始条件（v1-v5）
  - 完全相同的 process noise seed

唯一区别: B 多了 1 个物种，其与 v1-v5 有交互。

对比: v1-v5 的轨迹差异（RMSE、相对误差、相关性）。
"""
from __future__ import annotations

import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from data.partial_lv_mvp import generate_partial_lv_mvp_system, _build_candidate_parameters


def _configure_matplotlib() -> None:
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Hiragino Sans GB", "Heiti TC", "STHeiti",
        "Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def simulate_reduced_5species(
    growth_rates_5: torch.Tensor,          # (5,) 前 5 个物种的 growth
    interaction_5x5: torch.Tensor,          # (5, 5) 纯 5 物种的交互
    environment_loadings_5: torch.Tensor,   # (5,) 前 5 物种的 env 响应
    pulse_loadings_5: torch.Tensor,         # (5,) 前 5 物种的 pulse 响应
    initial_state_5: torch.Tensor,          # (5,) v1-v5 的初始值
    environment_driver: torch.Tensor,        # (T, 1) 完整环境序列
    pulse_driver: torch.Tensor,              # (T, 1) 完整脉冲序列
    noise_seed: int,
    process_noise: float = 0.006,
    max_state_value: float = 5.5,
) -> torch.Tensor:
    """只演化 5 物种，使用和 6 物种系统一致的 env/pulse/noise。"""
    T = int(environment_driver.shape[0])
    states = torch.zeros(T, 5, dtype=torch.float32)
    states[0] = initial_state_5.clone()

    gen = torch.Generator().manual_seed(noise_seed)
    for t in range(T - 1):
        current = states[t]
        drive = growth_rates_5 + interaction_5x5 @ current
        drive = drive + environment_loadings_5 * environment_driver[t, 0]
        drive = drive + pulse_loadings_5 * pulse_driver[t, 0]
        drive = drive + process_noise * 0.65 * torch.randn(5, generator=gen)

        next_state = current * torch.exp(torch.clamp(drive, min=-1.12, max=0.92))
        next_state = next_state + process_noise * 0.35 * torch.randn(5, generator=gen)
        states[t + 1] = torch.clamp(next_state, min=1e-4, max=max_state_value)

    return states


def main() -> None:
    _configure_matplotlib()
    out_dir = Path("runs/analysis_5vs6_species")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ----------------- Step 1: 生成 6 物种 baseline 系统 -----------------
    print("生成 6 物种系统（与之前实验完全相同）...")
    system_B = generate_partial_lv_mvp_system(seed=42)
    states_B = system_B.full_states  # (T, 6)
    visible_B = states_B[:, :5]        # v1-v5 的轨迹（在 6 物种系统中）
    hidden_B = states_B[:, 5]          # 第 6 物种在 6 物种系统中的轨迹
    env_driver = system_B.environment_driver
    pulse_driver = system_B.pulse_driver
    T = int(states_B.shape[0])
    print(f"  T = {T} 步")
    print(f"  growth_rates (6 物种): {system_B.growth_rates.tolist()}")

    # ----------------- Step 2: 提取前 5 物种的参数 -----------------
    growth_5 = system_B.growth_rates[:5].clone()
    interaction_6x6 = system_B.interaction_matrix.clone()
    interaction_5x5 = interaction_6x6[:5, :5].clone()   # 纯 5 物种子矩阵
    env_loadings_5 = system_B.environment_loadings[:5].clone()
    pulse_loadings_5 = system_B.pulse_loadings[:5].clone()

    # 初始条件：和 6 物种系统的起点完全一致
    # 注意：generate_partial_lv_mvp_system 内部有 warmup，我们取 warmup 后的 states
    # states_B[0] 已经是 warmup 之后的，所以直接取 states_B[0, :5]
    initial_5 = states_B[0, :5].clone()

    # ----------------- Step 3: 演化 5 物种系统 -----------------
    print("\n演化 5 物种系统（无第 6 物种影响）...")
    # 生成噪声 seed: 和 generate_partial_lv_mvp_system 里 simulate 用的 seed 保持差异但可复现
    # 原始系统内部 seed 是 seed + 1000 * (attempt + 1)。我们用一个固定 seed
    states_A = simulate_reduced_5species(
        growth_rates_5=growth_5,
        interaction_5x5=interaction_5x5,
        environment_loadings_5=env_loadings_5,
        pulse_loadings_5=pulse_loadings_5,
        initial_state_5=initial_5,
        environment_driver=env_driver,
        pulse_driver=pulse_driver,
        noise_seed=42 + 1000,
    )
    visible_A = states_A  # (T, 5)

    # ----------------- Step 4: 量化差异 -----------------
    print("\n量化 v1-v5 的轨迹差异:\n")

    # 每个物种的 RMSE / 相对 L2 / Pearson
    per_species_metrics = []
    for i in range(5):
        a = visible_A[:, i].numpy()
        b = visible_B[:, i].numpy()
        rmse = float(np.sqrt(((a - b) ** 2).mean()))
        rel_l2 = float(np.linalg.norm(a - b) / (np.linalg.norm(b) + 1e-8))
        pearson = float(np.corrcoef(a, b)[0, 1])
        abs_diff_mean = float(np.abs(a - b).mean())
        abs_diff_max = float(np.abs(a - b).max())
        per_species_metrics.append({
            "species": f"v{i+1}",
            "rmse": rmse,
            "rel_l2": rel_l2,
            "pearson": pearson,
            "abs_diff_mean": abs_diff_mean,
            "abs_diff_max": abs_diff_max,
            "a_std": float(a.std()),
            "b_std": float(b.std()),
        })
        print(f"  v{i+1}: RMSE={rmse:.4f}, rel_L2={rel_l2:.3f}, Pearson={pearson:+.3f}, "
              f"mean|diff|={abs_diff_mean:.4f}, std(A)={a.std():.3f}, std(B)={b.std():.3f}")

    # 全体（5 物种打包）指标
    all_rmse = float(np.sqrt(((visible_A.numpy() - visible_B.numpy()) ** 2).mean()))
    all_rel_l2 = float(np.linalg.norm(visible_A.numpy() - visible_B.numpy()) / np.linalg.norm(visible_B.numpy()))
    # 相对 B 的 std 标准化的差异（"影响相对信号的比例"）
    relative_to_signal = float(np.abs(visible_A.numpy() - visible_B.numpy()).mean() / visible_B.numpy().std())

    print()
    print(f"  ==== 整体 ====")
    print(f"  全 5 物种 RMSE: {all_rmse:.4f}")
    print(f"  全 5 物种 相对 L2: {all_rel_l2:.3f}")
    print(f"  mean|diff| / std(B): {relative_to_signal:.3f}    <- 差异占 visible 信号尺度的比例")

    # ----------------- Step 5: 画轨迹对比图 -----------------
    fig, axes = plt.subplots(5, 1, figsize=(14, 14), sharex=True, constrained_layout=True)
    time_axis = np.arange(T)
    train_end, val_end = 492, 656
    for i, ax in enumerate(axes):
        a = visible_A[:, i].numpy()
        b = visible_B[:, i].numpy()
        ax.plot(time_axis, b, color="black", linewidth=1.4, label=f"实验 B: 6物种系统中的 v{i+1}")
        ax.plot(time_axis, a, color="#ff7f0e", linewidth=1.1, alpha=0.85, linestyle="--", label=f"实验 A: 纯 5 物种系统中的 v{i+1}")
        ax.axvline(train_end - 0.5, color="#bbb", linestyle=":", linewidth=0.8)
        ax.axvline(val_end - 0.5, color="#bbb", linestyle=":", linewidth=0.8)
        m = per_species_metrics[i]
        ax.set_title(f"v{i+1}: Pearson={m['pearson']:+.3f}, RMSE={m['rmse']:.3f}, rel_L2={m['rel_l2']:.2f}",
                     fontsize=11)
        ax.set_ylabel("丰度")
        ax.legend(loc="upper right", fontsize=9)
        ax.grid(alpha=0.25)
    axes[-1].set_xlabel("时间步")
    fig.suptitle(
        "5 物种 vs 6 物种系统 v1-v5 轨迹对比\n"
        "（橙色虚线 = 没有第 6 物种；黑色实线 = 有第 6 物种）",
        fontsize=13, fontweight="bold", y=1.01,
    )
    fig.savefig(out_dir / "fig_trajectories_compare.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[OK] saved: {out_dir / 'fig_trajectories_compare.png'}")

    # ----------------- Step 6: 画影响强度柱状图 + 第 6 物种轨迹 -----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)

    # Panel 1: 每物种 RMSE
    axes[0].bar(range(5), [m["rmse"] for m in per_species_metrics],
                color=["#1565c0", "#e53935", "#43a047", "#fb8c00", "#8e24aa"],
                edgecolor="black")
    axes[0].set_xticks(range(5))
    axes[0].set_xticklabels([f"v{i+1}" for i in range(5)])
    axes[0].set_ylabel("RMSE (A vs B)")
    axes[0].set_title("每物种轨迹差异（RMSE）")
    for i, m in enumerate(per_species_metrics):
        axes[0].text(i, m["rmse"] + 0.005, f"{m['rmse']:.3f}", ha="center", fontsize=9)
    axes[0].grid(alpha=0.25, axis="y")

    # Panel 2: 每物种 Pearson
    axes[1].bar(range(5), [m["pearson"] for m in per_species_metrics],
                color=["#1565c0", "#e53935", "#43a047", "#fb8c00", "#8e24aa"],
                edgecolor="black")
    axes[1].set_xticks(range(5))
    axes[1].set_xticklabels([f"v{i+1}" for i in range(5)])
    axes[1].set_ylabel("Pearson (A vs B)")
    axes[1].set_title("每物种轨迹相似度（Pearson）")
    axes[1].axhline(1.0, color="grey", linestyle="--", linewidth=0.8, alpha=0.5)
    axes[1].axhline(0.5, color="red", linestyle="--", linewidth=0.8, alpha=0.5, label="0.5 线")
    axes[1].set_ylim([min(0, min(m["pearson"] for m in per_species_metrics) - 0.1), 1.05])
    for i, m in enumerate(per_species_metrics):
        axes[1].text(i, m["pearson"] - 0.08 if m["pearson"] > 0 else m["pearson"] + 0.03,
                     f"{m['pearson']:.2f}", ha="center", fontsize=9)
    axes[1].grid(alpha=0.25, axis="y")

    # Panel 3: 第 6 物种的轨迹（在 B 系统中）
    axes[2].plot(time_axis, hidden_B.numpy(), color="#222", linewidth=1.3)
    axes[2].axvline(train_end - 0.5, color="#bbb", linestyle=":", linewidth=0.8)
    axes[2].axvline(val_end - 0.5, color="#bbb", linestyle=":", linewidth=0.8)
    axes[2].set_title(f"第 6 物种（被 hide 的）在 B 系统中的轨迹\nstd={hidden_B.std():.3f}, range=[{hidden_B.min():.2f}, {hidden_B.max():.2f}]")
    axes[2].set_xlabel("时间步")
    axes[2].set_ylabel("丰度")
    axes[2].grid(alpha=0.25)

    fig.suptitle("「第 6 物种」对前 5 物种动力学的影响量化", fontsize=13, fontweight="bold", y=1.02)
    fig.savefig(out_dir / "fig_impact_summary.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[OK] saved: {out_dir / 'fig_impact_summary.png'}")

    # ----------------- Step 7: 生成文字报告 -----------------
    # 判断影响大小
    max_pearson = max(m["pearson"] for m in per_species_metrics)
    min_pearson = min(m["pearson"] for m in per_species_metrics)
    mean_pearson = np.mean([m["pearson"] for m in per_species_metrics])
    mean_rmse = np.mean([m["rmse"] for m in per_species_metrics])

    interpretation = []
    if min_pearson > 0.95:
        interpretation.append("- **结论：那一个物种的影响非常小**。v1-v5 在有无它的情况下轨迹几乎一致（Pearson > 0.95），说明它对系统动力学几乎是'被动'存在。")
    elif min_pearson > 0.8:
        interpretation.append("- **结论：中等影响**。v1-v5 的整体形状保持相似，但有可辨识的偏差。")
    elif min_pearson > 0.5:
        interpretation.append("- **结论：显著影响**。v1-v5 的轨迹在中后期开始出现实质差异。")
    else:
        interpretation.append("- **结论：决定性影响**。那一个物种不存在时，系统的演化轨迹发生了根本改变。")

    if max_pearson - min_pearson > 0.3:
        interpretation.append(f"- 影响不均匀：最强影响的物种 Pearson 仅 {min_pearson:.2f}，最弱影响的 {max_pearson:.2f}。说明 hidden 物种对不同 visible 物种的影响差异很大。")
    interpretation.append(f"- 整体差异幅度 mean|diff|/std(B) = {relative_to_signal:.2f}（差异 ≈ visible 信号标准差的 {relative_to_signal*100:.0f}%）")

    report = [
        "# 5 物种 vs 6 物种动力学对比报告",
        "",
        "## 实验目的",
        "量化'第 6 物种（即原先实验里的 hidden）'对前 5 物种动力学的实际影响。通过对比：",
        "- **A 系统**：纯 5 物种演化（没有第 6 物种存在）",
        "- **B 系统**：6 物种完整系统",
        "",
        "两个系统使用完全相同的前 5 维参数、完全相同的环境/脉冲序列、完全相同的初始条件。",
        "唯一变量：第 6 物种是否存在。",
        "",
        "## 每物种的轨迹差异",
        "",
        "| 物种 | RMSE | 相对 L2 | Pearson | mean\\|diff\\| |",
        "|------|------|---------|---------|-------------|",
    ]
    for m in per_species_metrics:
        report.append(f"| {m['species']} | {m['rmse']:.4f} | {m['rel_l2']:.3f} | {m['pearson']:+.3f} | {m['abs_diff_mean']:.4f} |")

    report += [
        "",
        "## 整体指标",
        "",
        f"- 全 5 物种 RMSE: **{all_rmse:.4f}**",
        f"- 全 5 物种 相对 L2: **{all_rel_l2:.3f}**",
        f"- 平均 Pearson: **{mean_pearson:.3f}**",
        f"- 差异占信号尺度比例: **{relative_to_signal:.3f}**",
        "",
        "## 第 6 物种（原 hidden）的自身轨迹",
        "",
        f"- std = {hidden_B.std():.3f}",
        f"- range = [{hidden_B.min():.3f}, {hidden_B.max():.3f}]",
        f"- mean = {hidden_B.mean():.3f}",
        "",
        "## 解读",
        "",
    ] + interpretation + [
        "",
        "## 对 hidden recovery 研究的意义",
        "",
        "- 如果影响小（Pearson > 0.95）：hidden 物种在 visible 动力学中几乎无迹可寻，**'从 visible 恢复 hidden'在信息论上是不可能的**。之前的 Pearson 0.87 恢复能力可能来自 encoder 过拟合，而非真实 partial observation。",
        "- 如果影响显著（Pearson < 0.85）：hidden 物种确实在 visible 动力学中留下了可识别的'signature'，**partial observation recovery 是有信息基础的**。",
        "",
    ]

    (out_dir / "report.md").write_text("\n".join(report), encoding="utf-8")
    print(f"[OK] saved: {out_dir / 'report.md'}")

    # 最后：保存 npz 便于后续分析
    np.savez(
        out_dir / "trajectories.npz",
        states_A_5species=visible_A.numpy(),
        states_B_5species=visible_B.numpy(),
        hidden_B=hidden_B.numpy(),
        environment_driver=env_driver.numpy(),
        pulse_driver=pulse_driver.numpy(),
        growth_rates_full=system_B.growth_rates.numpy(),
        interaction_matrix_full=interaction_6x6.numpy(),
    )
    print(f"[OK] saved: {out_dir / 'trajectories.npz'}")

    print(f"\n完整报告: {out_dir}/report.md")


if __name__ == "__main__":
    main()

"""Lake Mendota 水生浮游生物群落 hidden recovery.

与 Portal 对应的真实数据第二案例 (水生 vs 陆地).

预处理:
- 筛选 ME (Mendota)
- 按 (年, 月, 物种) 聚合 biomass_conc
- 从出现率 >= 40% 且 biomass 累积 top 的物种中选 top-12
- 缺失月份填 0 (浮游群落确实可在某些月几近消失)
- 在 top-12 中选 hidden (倾向中等耦合强度物种)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Tuple, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual


def load_lake_mendota(
    csv_path: str = "data/real_datasets/lake_mendota_phytoplankton.csv",
    lake_id: str = "ME",
    top_k: int = 12,
    min_occurrence: float = 0.40,
):
    """返回 (matrix T×K, 物种名列表, 年月列表, 全部候选物种的排序).

    top_k 物种选择: 要求出现率 >= min_occurrence, 然后在其中按总 biomass 取 top-k.
    """
    records = defaultdict(lambda: defaultdict(float))
    taxa_total_biomass = defaultdict(float)

    with open(csv_path) as f:
        r = csv.DictReader(f)
        for row in r:
            if row["lakeid"] != lake_id:
                continue
            taxa = " ".join(row["taxa_name"].split())
            try:
                year = int(row["year4"])
                month = int(row["sampledate"].split("-")[1])
                biomass = float(row["biomass_conc"]) if row["biomass_conc"] else 0.0
            except (ValueError, IndexError, KeyError):
                continue
            if biomass <= 0:
                continue
            records[(year, month)][taxa] += biomass
            taxa_total_biomass[taxa] += biomass

    all_ym = sorted(records.keys())
    T = len(all_ym)

    # 构造完整 (year, month) 网格填补缺失
    year_min, year_max = all_ym[0][0], all_ym[-1][0]
    full_ym = [(y, m) for y in range(year_min, year_max + 1) for m in range(1, 13)]
    # 取两者交集顺序 (完整网格)
    ym_full = full_ym  # 使用完整 12 月网格
    T_full = len(ym_full)

    # 每物种出现率 (按完整月网格算)
    taxa_occurrence = {tx: 0 for tx in taxa_total_biomass}
    for ym in ym_full:
        if ym in records:
            for tx in records[ym]:
                taxa_occurrence[tx] += 1
    taxa_occurrence = {tx: n / T_full for tx, n in taxa_occurrence.items()}

    # 筛选 + 排序
    eligible = {tx: bio for tx, bio in taxa_total_biomass.items()
                if taxa_occurrence.get(tx, 0) >= min_occurrence}
    sorted_taxa = sorted(eligible.items(), key=lambda x: -x[1])
    top_taxa = [tx for tx, _ in sorted_taxa[:top_k]]

    # 构造矩阵 (T_full × top_k), 缺月填 0
    matrix = np.zeros((T_full, top_k), dtype=np.float32)
    for t, ym in enumerate(ym_full):
        if ym in records:
            for j, tx in enumerate(top_taxa):
                matrix[t, j] = records[ym].get(tx, 0.0)
        # else 保持 0

    return matrix, top_taxa, ym_full, taxa_occurrence, taxa_total_biomass


def print_data_summary(matrix, top_taxa, ym_full, taxa_occurrence, taxa_total_biomass):
    T = matrix.shape[0]
    total_all_taxa = sum(taxa_total_biomass.values())
    top_total = sum(taxa_total_biomass[tx] for tx in top_taxa)

    print(f"=== Lake Mendota 数据汇总 ===")
    print(f"完整月网格: {T} 个月 ({ym_full[0]} → {ym_full[-1]})")
    print(f"候选物种总数: {len(taxa_total_biomass)}")
    print(f"筛选后 top-{len(top_taxa)} 物种覆盖 biomass: {100*top_total/total_all_taxa:.1f}%")
    print()
    print(f"top-{len(top_taxa)} 物种详情:")
    print(f"  {'rank':<5}{'taxa':<40s}{'occur':<10}{'mean':<10}{'max':<10}{'std':<10}")
    for j, tx in enumerate(top_taxa):
        v = matrix[:, j]
        print(f"  {j+1:<5d}{tx[:38]:<40s}"
              f"{taxa_occurrence[tx]:<10.3f}{v.mean():<10.3f}{v.max():<10.3f}{v.std():<10.3f}")


def train_once(visible, hidden_eval, device, seed=42, epochs=300, hidden_name=""):
    """单 seed 训练, 返回 Pearson 与诊断."""
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0,
        gnn_backbone="mlp",
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(0.2 * epochs)
    ramp_epochs = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            h_weight, rollout_K, lam_rollout = 0.0, 0, 0.0
        else:
            post = epoch - warmup_epochs
            h_weight = min(1.0, post / ramp_epochs)
            k_ramp = min(1.0, post / (epochs - warmup_epochs) * 2)
            rollout_K = max(1 if h_weight > 0 else 0, int(round(k_ramp * 3)))
            lam_rollout = 0.5 * h_weight

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=rollout_K)
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out, beta_kl=0.03, free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=0.05,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_weight, lam_rollout=lam_rollout,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=0.05,
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup_epochs + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

    # Pearson
    L = min(len(h_mean), len(hidden_eval))
    h_mean = h_mean[:L]; hidden_eval = hidden_eval[:L]
    X = np.concatenate([h_mean.reshape(-1, 1), np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_eval, rcond=None)
    h_scaled = X @ coef
    pear = float(np.corrcoef(h_scaled, hidden_eval)[0, 1])

    # d_ratio
    with torch.no_grad():
        h_true_t = torch.tensor(hidden_eval, dtype=torch.float32, device=device).unsqueeze(0)
        h_true_c = h_true_t - h_true_t.mean()
        mu_k, _ = model.encoder(x)
        encoder_h = mu_k[..., 0]
        encoder_h_c = encoder_h - encoder_h.mean()
        encoder_std = encoder_h_c.std()
        h_true_scaled = h_true_c * (encoder_std / (h_true_c.std() + 1e-6))

        safe = torch.clamp(x, min=1e-6)
        actual = torch.log(safe[:, 1:] / safe[:, :-1])
        actual = torch.clamp(actual, -2.5, 2.5)

        base = model.compute_f_visible(x)
        G = model.compute_G(x)
        pred_encoder = base + encoder_h.unsqueeze(-1) * G
        pred_true = base + h_true_scaled.unsqueeze(-1) * G
        recon_encoder = F.mse_loss(pred_encoder[:, :-1], actual).item()
        recon_true = F.mse_loss(pred_true[:, :-1], actual).item()
    d_ratio = recon_true / recon_encoder

    return {
        "seed": seed,
        "pearson": pear,
        "h_mean": h_mean,
        "h_scaled": h_scaled,
        "val_recon": best_val,
        "d_ratio": d_ratio,
        "best_epoch": best_epoch,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--hidden_idx", type=int, default=None,
                         help="指定 hidden 物种 index (0-based, 不指定则遍历探索)")
    parser.add_argument("--explore", action="store_true",
                         help="只探索不同 hidden 选择, 每个 1 seed")
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_mendota_cvhi")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    matrix, top_taxa, ym_full, taxa_occurrence, taxa_total_biomass = load_lake_mendota()
    print_data_summary(matrix, top_taxa, ym_full, taxa_occurrence, taxa_total_biomass)

    # smooth (3-mo moving avg, 与 Portal 一致)
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w//2, w//2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w//2:w//2+x.shape[0]]
    matrix_s = smooth(matrix, w=3)

    # 过滤几乎全零的月份 (前几个月常数据稀疏)
    valid = matrix_s.sum(axis=1) > 0
    matrix_s = matrix_s[valid]
    ym_valid = [ym for ym, v in zip(ym_full, valid) if v]
    T_final = len(ym_valid)
    print(f"\n过滤后 T = {T_final} 月")

    if args.explore:
        # 以每个物种作为 hidden 跑 1 seed, 看哪些可恢复
        print(f"\n{'='*70}\nExplore: 遍历每个 top-{len(top_taxa)} 物种作为 hidden (1 seed each)\n{'='*70}")
        results = []
        for h_idx in range(len(top_taxa)):
            keep = [i for i in range(len(top_taxa)) if i != h_idx]
            visible = matrix_s[:, keep] + 0.5
            hidden = matrix_s[:, h_idx] + 0.5
            r = train_once(visible, hidden, device, seed=42, epochs=args.epochs,
                            hidden_name=top_taxa[h_idx])
            r["hidden_name"] = top_taxa[h_idx]
            r["hidden_idx"] = h_idx
            results.append(r)
            print(f"  idx {h_idx:2d} {top_taxa[h_idx][:35]:<35s}  P={r['pearson']:+.4f}  "
                  f"d_ratio={r['d_ratio']:.3f}  val={r['val_recon']:.4f}")

        # 按 |Pearson| 排序
        results_sorted = sorted(results, key=lambda r: -abs(r["pearson"]))
        print(f"\n按 |Pearson| 排序:")
        for i, r in enumerate(results_sorted):
            print(f"  {i+1:2d}. {r['hidden_name'][:35]:<35s}  P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}")
        return

    # 单一 hidden 多 seed 实验
    h_idx = args.hidden_idx if args.hidden_idx is not None else 0
    hidden_name = top_taxa[h_idx]
    print(f"\n{'='*70}\nHidden = {hidden_name} (idx {h_idx})\n{'='*70}")

    keep = [i for i in range(len(top_taxa)) if i != h_idx]
    visible = matrix_s[:, keep] + 0.5
    hidden = matrix_s[:, h_idx] + 0.5
    print(f"Visible N={visible.shape[1]}, T={visible.shape[0]}\n")

    seeds = [42, 123, 456, 789, 2024][:args.n_seeds]
    results = []
    for seed in seeds:
        print(f"\n--- seed {seed} ---")
        r = train_once(visible, hidden, device, seed=seed, epochs=args.epochs,
                        hidden_name=hidden_name)
        r["hidden_name"] = hidden_name
        results.append(r)
        print(f"  Pearson={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}  "
              f"val_recon={r['val_recon']:.4f}  best_ep={r['best_epoch']}")

    pearsons = np.array([r["pearson"] for r in results])
    print(f"\n{'='*70}")
    print(f"Lake Mendota / hidden={hidden_name} / {len(seeds)} seeds")
    print(f"{'='*70}")
    print(f"mean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}")
    print(f"max  = {pearsons.max():+.4f}")
    print(f"median = {np.median(pearsons):+.4f}")

    # 保存
    np.savez(out_dir / "results.npz",
              pearsons=pearsons,
              seeds=np.array(seeds),
              hidden_name=hidden_name,
              matrix_shape=np.array(matrix_s.shape))

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# Lake Mendota 第二个真实数据集实验\n\n")
        f.write(f"hidden = {hidden_name} (top-12 中 idx {h_idx})\n\n")
        f.write(f"visible N = {visible.shape[1]}, T = {visible.shape[0]}\n\n")
        f.write(f"| seed | Pearson | d_ratio | val_recon |\n|---|---|---|---|\n")
        for r in results:
            f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['d_ratio']:.3f} | {r['val_recon']:.4f} |\n")
        f.write(f"\nmean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}\n")
        f.write(f"max  = {pearsons.max():+.4f}\n")

    print(f"\n[OK] saved to {out_dir}")


if __name__ == "__main__":
    main()

"""Lake Mendota 跨营养级数据集: phytoplankton (生产者) + zooplankton (消费者).

意义: 真正跨营养级的"较完整子系统", 突破 Portal/Mendota-only 的子群落限制.
Visible 同时包含生产者与消费者, hidden 也可在跨营养级语境下选择.
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual


def load_phyto_zoo_combined(
    n_phyto: int = 7,
    n_zoo: int = 5,
    min_occurrence: float = 0.30,
):
    """加载并合并 phytoplankton + zooplankton 数据.

    返回:
      matrix (T, n_phyto + n_zoo): 月度丰度矩阵 (log1p 单位统一前)
      taxa_list: 物种名列表 (前 n_phyto 是 phyto, 后 n_zoo 是 zoo)
      ym_full: 完整年月列表
      occurrence: dict 物种 → 出现率
    """
    # 1) phytoplankton: biomass_conc 累积
    phyto_records = defaultdict(lambda: defaultdict(float))
    phyto_total = defaultdict(float)
    with open("data/real_datasets/lake_mendota_phytoplankton.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["lakeid"] != "ME":
                continue
            tx = " ".join(row["taxa_name"].split())
            try:
                year = int(row["year4"])
                month = int(row["sampledate"].split("-")[1])
                bio = float(row["biomass_conc"]) if row["biomass_conc"] else 0
            except (ValueError, IndexError, KeyError):
                continue
            if bio <= 0:
                continue
            phyto_records[(year, month)][tx] += bio
            phyto_total[tx] += bio

    # 2) zooplankton: density 累积
    zoo_records = defaultdict(lambda: defaultdict(float))
    zoo_total = defaultdict(float)
    with open("data/real_datasets/lake_mendota_zooplankton.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            if row["lakeid"] != "ME":
                continue
            tx = " ".join(row["species_name"].split())
            try:
                year = int(row["year4"])
                month = int(row["sample_date"].split("-")[1])
                density = float(row["density"]) if row["density"] else 0
            except (ValueError, IndexError, KeyError):
                continue
            if density <= 0:
                continue
            zoo_records[(year, month)][tx] += density
            zoo_total[tx] += density

    # 3) 共同时间网格 (取两者的并集年范围 + 完整月)
    all_years_phyto = set(y for y, _ in phyto_records)
    all_years_zoo = set(y for y, _ in zoo_records)
    common_years = sorted(all_years_phyto & all_years_zoo)
    print(f"Phyto 年范围: {sorted(all_years_phyto)[0]}-{sorted(all_years_phyto)[-1]}")
    print(f"Zoo 年范围:   {sorted(all_years_zoo)[0]}-{sorted(all_years_zoo)[-1]}")
    print(f"共同年范围:   {common_years[0]}-{common_years[-1]} ({len(common_years)} 年)")

    ym_full = [(y, m) for y in common_years for m in range(1, 13)]
    T_full = len(ym_full)

    # 4) 物种出现率 (按完整网格)
    def calc_occurrence(records, ym_list):
        out = defaultdict(int)
        for ym in ym_list:
            if ym in records:
                for tx in records[ym]:
                    out[tx] += 1
        return {tx: n / len(ym_list) for tx, n in out.items()}

    phyto_occur = calc_occurrence(phyto_records, ym_full)
    zoo_occur = calc_occurrence(zoo_records, ym_full)

    # 5) 选 top 物种 (出现率达标 + biomass/density 最大)
    phyto_eligible = {tx: bio for tx, bio in phyto_total.items()
                      if phyto_occur.get(tx, 0) >= min_occurrence}
    zoo_eligible = {tx: d for tx, d in zoo_total.items()
                    if zoo_occur.get(tx, 0) >= min_occurrence}

    top_phyto = [tx for tx, _ in sorted(phyto_eligible.items(), key=lambda x: -x[1])[:n_phyto]]
    top_zoo = [tx for tx, _ in sorted(zoo_eligible.items(), key=lambda x: -x[1])[:n_zoo]]
    taxa_list = top_phyto + top_zoo

    print(f"\n选中 top-{n_phyto} 浮游植物 (生产者):")
    for i, tx in enumerate(top_phyto):
        print(f"  P{i+1}. {tx[:50]:<50s}  occur={phyto_occur[tx]:.2f}")
    print(f"\n选中 top-{n_zoo} 浮游动物 (消费者):")
    for i, tx in enumerate(top_zoo):
        print(f"  Z{i+1}. {tx[:50]:<50s}  occur={zoo_occur[tx]:.2f}")

    # 6) 构造矩阵
    K = len(taxa_list)
    matrix = np.zeros((T_full, K), dtype=np.float32)
    for t, ym in enumerate(ym_full):
        # phytoplankton
        if ym in phyto_records:
            for j, tx in enumerate(top_phyto):
                matrix[t, j] = phyto_records[ym].get(tx, 0.0)
        # zooplankton (后 n_zoo 列)
        if ym in zoo_records:
            for j, tx in enumerate(top_zoo):
                matrix[t, n_phyto + j] = zoo_records[ym].get(tx, 0.0)

    # 7) 单位归一化: 各列除以其非零值的中位数 (使 phyto 和 zoo 量级相近)
    matrix_norm = matrix.copy()
    for j in range(K):
        col = matrix[:, j]
        nz = col[col > 0]
        if len(nz) > 0:
            med = np.median(nz)
            matrix_norm[:, j] = col / med  # 各列中位数标准化

    return matrix_norm, taxa_list, ym_full, phyto_occur, zoo_occur, n_phyto


def train_one(visible, hidden_eval, device, seed=42, epochs=300):
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

    return {"seed": seed, "pearson": pear, "h_mean": h_mean, "h_scaled": h_scaled,
            "val_recon": best_val, "d_ratio": d_ratio, "best_epoch": best_epoch}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--explore", action="store_true")
    parser.add_argument("--hidden_idx", type=int, default=None)
    parser.add_argument("--n_seeds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=300)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_mendota_xtroph")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    matrix, taxa_list, ym_full, phyto_occur, zoo_occur, n_phyto = load_phyto_zoo_combined()
    print(f"\n最终矩阵: {matrix.shape}  (前 {n_phyto} 列 phyto, 后 {len(taxa_list)-n_phyto} 列 zoo)")

    # smooth
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w//2, w//2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w//2:w//2+x.shape[0]]
    matrix_s = smooth(matrix, w=3)

    # 过滤几乎全零月份
    valid = matrix_s.sum(axis=1) > 0
    matrix_s = matrix_s[valid]
    print(f"过滤后 T = {matrix_s.shape[0]}\n")

    if args.explore:
        print(f"{'='*72}\nExplore: 遍历 12 个物种作 hidden (1 seed each)\n{'='*72}")
        results = []
        for h_idx in range(len(taxa_list)):
            keep = [i for i in range(len(taxa_list)) if i != h_idx]
            visible = matrix_s[:, keep] + 0.5
            hidden = matrix_s[:, h_idx] + 0.5
            r = train_one(visible, hidden, device, seed=42, epochs=args.epochs)
            r["hidden_name"] = taxa_list[h_idx]
            r["hidden_idx"] = h_idx
            r["hidden_type"] = "phyto" if h_idx < n_phyto else "zoo"
            results.append(r)
            print(f"  idx {h_idx:2d} [{r['hidden_type']:5s}] {taxa_list[h_idx][:35]:<35s}  "
                  f"P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}")

        results_sorted = sorted(results, key=lambda r: -abs(r["pearson"]))
        print(f"\n按 |Pearson| 排序:")
        for i, r in enumerate(results_sorted):
            print(f"  {i+1:2d}. [{r['hidden_type']:5s}] {r['hidden_name'][:35]:<35s}  "
                  f"P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}")
        return

    # 单 hidden 多 seed
    h_idx = args.hidden_idx if args.hidden_idx is not None else 0
    hidden_name = taxa_list[h_idx]
    h_type = "phyto" if h_idx < n_phyto else "zoo"
    print(f"{'='*72}\nHidden = [{h_type}] {hidden_name}\n{'='*72}")

    keep = [i for i in range(len(taxa_list)) if i != h_idx]
    visible = matrix_s[:, keep] + 0.5
    hidden = matrix_s[:, h_idx] + 0.5
    print(f"Visible N={visible.shape[1]}, T={visible.shape[0]}\n")

    seeds = [42, 123, 456, 789, 2024][:args.n_seeds]
    results = []
    for seed in seeds:
        print(f"--- seed {seed} ---")
        r = train_one(visible, hidden, device, seed=seed, epochs=args.epochs)
        results.append(r)
        print(f"  P={r['pearson']:+.4f}  d_ratio={r['d_ratio']:.3f}  val={r['val_recon']:.4f}")

    pearsons = np.array([r["pearson"] for r in results])
    print(f"\n{'='*72}")
    print(f"Lake Mendota 跨营养级 / hidden = [{h_type}] {hidden_name}")
    print(f"{'='*72}")
    print(f"mean = {pearsons.mean():+.4f} ± {pearsons.std():.4f}")
    print(f"max  = {pearsons.max():+.4f}")
    print(f"median = {np.median(pearsons):+.4f}")

    np.savez(out_dir / "results.npz",
              pearsons=pearsons, seeds=np.array(seeds),
              hidden_name=hidden_name, hidden_type=h_type)
    print(f"\n[OK] saved to {out_dir}")


if __name__ == "__main__":
    main()

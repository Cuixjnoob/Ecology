"""Path B: Multi-hidden joint training.

每训练 step 随机选一个物种作 hidden, 其它 11 作 visible.
共享 encoder (按 species ID 索引 emb), 不同 hidden choice 用不同的 f_visible / G.
评测时 fix hidden = OT.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib, hidden_true_substitution, TOP12,
)


SEEDS_20 = [42, 123, 456, 789, 2024, 31415, 27182, 65537,
             7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47]


def load_portal_full():
    """加载 Portal 12 species 原始 matrix (T, 12)."""
    # reuse load_portal with arbitrary hidden, rebuild full
    vis, hid = load_portal("OT")  # 11 + 1
    ot_idx = TOP12.index("OT")
    # reconstruct: combined = insert hid at position ot_idx
    T = vis.shape[0]
    full = np.zeros((T, 12), dtype=np.float32)
    other_ids = [i for i in range(12) if i != ot_idx]
    full[:, other_ids] = vis
    full[:, ot_idx] = hid
    return full, TOP12


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_multi_hidden(full_data, eval_hidden_idx, device, seed, epochs=300,
                       sample_schedule="random"):
    """full_data: (T, 12), eval_hidden_idx: which species to evaluate on (= OT idx)."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    T, N_total = full_data.shape
    N_visible = N_total - 1
    full = torch.tensor(full_data, dtype=torch.float32, device=device)  # (T, 12)

    model = CVHI_Residual(
        num_visible=N_visible,
        num_total_species=N_total,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val_eval = float("inf"); best_state = None
    m_null, m_shuf, min_e = 0.002, 0.001, 0.05

    def make_visible(h_idx):
        """Return (visible_tensor (1,T,11), species_ids (11,)) given which species is hidden."""
        other_ids = [i for i in range(N_total) if i != h_idx]
        sp_ids = torch.tensor(other_ids, dtype=torch.long, device=device)
        vis = full[:, sp_ids].unsqueeze(0)  # (1, T, 11)
        return vis, sp_ids

    for epoch in range(epochs):
        model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / (epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        # 选 hidden idx: sample_schedule 决定
        if sample_schedule == "random":
            h_idx_train = np.random.randint(0, N_total)
        elif sample_schedule == "cycle":
            h_idx_train = epoch % N_total
        elif sample_schedule == "eval_only":
            h_idx_train = eval_hidden_idx
        else:
            raise ValueError(sample_schedule)

        vis_tr, sp_ids_tr = make_visible(h_idx_train)

        model.train(); opt.zero_grad()
        out = model(vis_tr, n_samples=2, rollout_K=K_r, species_ids=sp_ids_tr)
        tr_out = model.slice_out(out, 0, train_end)
        losses = model.loss(tr_out, beta_kl=0.03, free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_e,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        # val_recon tracked on EVAL hidden (= OT), not training-step hidden
        with torch.no_grad():
            vis_eval, sp_ids_eval = make_visible(eval_hidden_idx)
            out_eval = model(vis_eval, n_samples=1, rollout_K=0, species_ids=sp_ids_eval)
            val_out = model.slice_out(out_eval, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=5.0, lam_shuffle=3.0,
                lam_energy=2.0, min_energy=min_e,
                lam_rollout=0.0, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0)
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val_eval:
            best_val_eval = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()

    # eval on OT
    vis_eval, sp_ids_eval = make_visible(eval_hidden_idx)
    with torch.no_grad():
        out_final = model(vis_eval, n_samples=30, rollout_K=3, species_ids=sp_ids_eval)
        h_mean = out_final["h_samples"].mean(dim=0)[0].cpu().numpy()
    hidden_true = full_data[:, eval_hidden_idx]
    pear, _ = evaluate(h_mean, hidden_true)
    # d_ratio 只在 eval hidden 下算, 但 substitution helper 需要 model.encoder(x) 接口兼容
    # 这里跳过 d_ratio (新接口未实现), 返回 pearson + val
    return {"seed": seed, "pearson": pear, "val_recon": best_val_eval}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_seeds", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--schedules", nargs="+", default=["eval_only", "random", "cycle"])
    args = ap.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_multi_hidden")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species_names = load_portal_full()
    print(f"Portal full shape: {full.shape}  species: {species_names}")
    ot_idx = species_names.index("OT")
    print(f"OT index: {ot_idx}")
    seeds = SEEDS_20[:args.n_seeds]

    results = {s: [] for s in args.schedules}
    for sched in args.schedules:
        print(f"\n{'='*70}\nSchedule: {sched}\n{'='*70}")
        for s in seeds:
            t0 = datetime.now()
            r = train_multi_hidden(full, ot_idx, device, s, epochs=args.epochs, sample_schedule=sched)
            dt = (datetime.now() - t0).total_seconds()
            results[sched].append(r)
            print(f"  seed={s}  P={r['pearson']:+.3f}  val={r['val_recon']:.4f}  ({dt:.1f}s)")

    from scipy import stats
    summary = {}
    for sched in args.schedules:
        P = np.array([r["pearson"] for r in results[sched]])
        summary[sched] = {"mean": float(P.mean()), "std": float(P.std(ddof=1) if len(P)>1 else 0),
                           "max": float(P.max())}

    print(f"\n{'='*70}\nSUMMARY\n{'='*70}")
    print(f"{'schedule':<15s}{'mean P':<12s}{'std':<10s}{'max':<8s}")
    for sched in args.schedules:
        s = summary[sched]
        print(f"{sched:<15s}{s['mean']:+.4f}      {s['std']:.4f}    {s['max']:+.3f}")

    # paired-t tests
    if "random" in summary and "eval_only" in summary:
        P_rand = np.array([r["pearson"] for r in results["random"]])
        P_eval = np.array([r["pearson"] for r in results["eval_only"]])
        t, p = stats.ttest_rel(P_rand, P_eval)
        print(f"\npaired: random vs eval_only  Δ={P_rand.mean()-P_eval.mean():+.4f}  t={t:+.3f}  p={p:.4g}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Multi-Hidden Joint Training\n\n")
        f.write(f"- seeds: {len(seeds)}  epochs: {args.epochs}\n")
        f.write(f"- eval hidden: OT (species_idx={ot_idx})\n\n")
        f.write("| schedule | mean P | std | max |\n|---|---|---|---|\n")
        for sched in args.schedules:
            s = summary[sched]
            f.write(f"| {sched} | {s['mean']:+.4f} | {s['std']:.4f} | {s['max']:+.3f} |\n")

    with open(out_dir / "raw_results.json", "w") as f:
        json.dump({s: [{k: float(v) if isinstance(v,(int,float,np.floating)) else v for k,v in r.items()} for r in rs]
                    for s, rs in results.items()}, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

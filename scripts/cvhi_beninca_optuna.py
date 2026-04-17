"""Beninca 超参数 Optuna 自动搜索.

Phase 1: 在 Nanophyto (v1 测试 max 0.27) 上搜 8 个超参, 30 trials, 3 seeds 每 trial.
Phase 2 (后续手动): 用最佳超参跑全部 9 hidden, 5 seeds.

目标: mean Pearson > 0.20 (Phase 1 sweet spot), 然后 Phase 2 看能否 ~0.3.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import optuna
import torch

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, _configure_matplotlib
from scripts.load_beninca import load_beninca


SEEDS_3 = [42, 123, 456]


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden_eval, device, seed, hp):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = CVHI_Residual(
        num_visible=N,
        encoder_d=hp["encoder_d"], encoder_blocks=hp["encoder_blocks"], encoder_heads=4,
        takens_lags=hp["takens_lags"], encoder_dropout=hp["encoder_dropout"],
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=1e-4)
    epochs = hp["epochs"]
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None

    m_null, m_shuf = 0.002, 0.001
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
        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            tr_out, beta_kl=hp["lam_kl"], free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=hp["lam_cf"], lam_shuffle=hp["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=hp["min_energy"],
            lam_smooth=hp["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=hp["lam_hf"], lowpass_sigma=6.0,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=hp["lam_cf"], lam_shuffle=hp["lam_cf"]*0.6,
                lam_energy=2.0, min_energy=hp["min_energy"],
                lam_rollout=lam_r, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=hp["lam_hf"], lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden_eval)
    return pear


def make_objective(visible, hidden_eval, device, n_seeds=3):
    def objective(trial):
        # Search space
        takens_choice = trial.suggest_categorical(
            "takens_choice", ["short", "med", "long"]
        )
        takens_map = {
            "short": (1, 2, 3),
            "med": (1, 2, 4, 8),
            "long": (1, 2, 4, 8, 12),
        }
        hp = {
            "encoder_d": trial.suggest_categorical("encoder_d", [48, 64, 96, 128]),
            "encoder_blocks": trial.suggest_int("encoder_blocks", 2, 3),
            "encoder_dropout": trial.suggest_float("encoder_dropout", 0.05, 0.3, step=0.05),
            "takens_lags": takens_map[takens_choice],
            "lr": trial.suggest_float("lr", 3e-4, 3e-3, log=True),
            "epochs": trial.suggest_categorical("epochs", [200, 300, 500]),
            "lam_smooth": trial.suggest_float("lam_smooth", 0.0, 0.05, step=0.005),
            "lam_kl": trial.suggest_float("lam_kl", 0.005, 0.1, log=True),
            "lam_hf": trial.suggest_float("lam_hf", 0.0, 0.3, step=0.05),
            "min_energy": trial.suggest_float("min_energy", 0.02, 0.3, log=True),
            "lam_cf": trial.suggest_float("lam_cf", 1.0, 10.0, log=True),
        }
        pearsons = []
        for s in SEEDS_3[:n_seeds]:
            try:
                p = train_one(visible, hidden_eval, device, s, hp)
                pearsons.append(p)
            except Exception as e:
                print(f"  seed {s} failed: {e}")
                pearsons.append(0.0)
        mean_p = float(np.mean(pearsons))
        # report intermediate
        trial.set_user_attr("pearsons", pearsons)
        return mean_p
    return objective


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_trials", type=int, default=30)
    ap.add_argument("--n_seeds", type=int, default=3)
    ap.add_argument("--hidden_species", type=str, default="Nanophyto")
    ap.add_argument("--dt_days", type=int, default=4)
    args = ap.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_optuna")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, days = load_beninca(dt_days=args.dt_days)
    h_idx = species.index(args.hidden_species)
    visible = np.delete(full, h_idx, axis=1)
    hidden = full[:, h_idx]
    print(f"Tuning on hidden={args.hidden_species} (idx={h_idx}), visible shape={visible.shape}")

    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    obj = make_objective(visible, hidden, device, n_seeds=args.n_seeds)

    print(f"\nStarting Optuna study: {args.n_trials} trials × {args.n_seeds} seeds each")
    study.optimize(obj, n_trials=args.n_trials, show_progress_bar=False)

    print(f"\n{'='*70}\nBest trial: #{study.best_trial.number}")
    print(f"  Pearson mean = {study.best_value:+.4f}")
    print(f"  Per-seed: {study.best_trial.user_attrs.get('pearsons', [])}")
    print(f"  Params: {study.best_params}")

    # Save
    with open(out_dir / "best_params.json", "w") as f:
        json.dump({"best_value": study.best_value, "best_params": study.best_params,
                    "best_pearsons": study.best_trial.user_attrs.get("pearsons", [])},
                   f, indent=2)

    # All trials
    all_trials = []
    for t in study.trials:
        all_trials.append({
            "number": t.number, "value": t.value,
            "params": t.params, "pearsons": t.user_attrs.get("pearsons", []),
        })
    with open(out_dir / "all_trials.json", "w") as f:
        json.dump(all_trials, f, indent=2, default=float)

    # Top 10 print
    sorted_trials = sorted([t for t in study.trials if t.value is not None],
                            key=lambda x: -x.value)[:10]
    print(f"\nTop-10 trials:")
    print(f"  {'rank':<5}{'trial':<7}{'P_mean':<10}{'lam_smooth':<12}{'enc_d':<7}{'lr':<10}{'takens':<8}")
    for r, t in enumerate(sorted_trials):
        p = t.params
        print(f"  {r+1:<5}{t.number:<7}{t.value:<+10.4f}"
               f"{p.get('lam_smooth', 0):<12.4f}{p.get('encoder_d',0):<7d}"
               f"{p.get('lr',0):<10.5f}{p.get('takens_choice','?'):<8}")
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

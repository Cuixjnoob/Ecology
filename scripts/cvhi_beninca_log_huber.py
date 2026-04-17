"""Fix 1 + Fix 4: log-transform input + Huber loss on Beninca.

Diagnostic hypothesis: real plankton data is heavily bursty (CV=3.17, kurt=62),
violating VAE's Gaussian assumptions and MSE's outlier sensitivity.

Two fixes (orthogonal, complementary):
  Fix 1: encoder/f/G input = log(x + ε)  → smoother representation
  Fix 4: MSE → Huber(δ=0.1)              → robust to bloom spikes

Test grid:
  (raw, MSE) — baseline Stage 1b
  (log, MSE) — Fix 1 only
  (raw, Huber) — Fix 4 only
  (log, Huber) — Fix 1+4
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456]
EPOCHS = 500

BEST_HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_smooth=0.02, lam_kl=0.017251789430967935,
    lam_hf=0.2, min_energy=0.14353013693386804,
    lam_cf=9.517725868477207,
)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

# Ablation configs to compare
CONFIGS = [
    # (name, use_log, use_huber)
    ("baseline (raw, MSE)",  False, False),
    ("log only (Fix1)",       True,  False),
    ("huber only (Fix4)",     False, True),
    ("log+huber (Fix1+4)",    True,  True),
]


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=BEST_HP["encoder_d"], encoder_blocks=BEST_HP["encoder_blocks"],
        encoder_heads=4,
        takens_lags=BEST_HP["takens_lags"], encoder_dropout=BEST_HP["encoder_dropout"],
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible_raw, hidden_raw, seed, device, use_log, use_huber, epochs=EPOCHS):
    """
    visible_raw: (T, N) raw mean-normalized data
    hidden_raw: (T,) true hidden (for eval only)
    use_log: if True, transform input to log-space
    use_huber: if True, Huber recon
    """
    torch.manual_seed(seed)
    T, N = visible_raw.shape

    # Fix 1: log-transform if enabled
    if use_log:
        visible_proc = np.log(visible_raw + 1e-6)
        # Re-center to mean=0 per channel (helps network init)
        visible_proc = visible_proc - visible_proc.mean(axis=0, keepdims=True)
    else:
        visible_proc = visible_raw.astype(np.float32)

    x_full = torch.tensor(visible_proc, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    opt = torch.optim.AdamW(model.parameters(), lr=BEST_HP["lr"], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)

        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]

        losses = model.loss(
            tr_out,
            beta_kl=BEST_HP["lam_kl"], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=BEST_HP["lam_cf"],
            lam_shuffle=BEST_HP["lam_cf"]*0.6,
            lam_energy=2.0, min_energy=BEST_HP["min_energy"],
            lam_smooth=BEST_HP["lam_smooth"], lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=BEST_HP["lam_hf"], lowpass_sigma=6.0,
            lam_rmse_log=0.1,
            use_huber_recon=use_huber, huber_delta=0.1,
        )
        total = losses["total"]
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_out["visible"] = out["visible"][:, train_end:T]
            val_out["G"] = out["G"][:, train_end:T]
            val_losses = model.loss(
                val_out, h_weight=1.0,
                margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=BEST_HP["min_energy"],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=BEST_HP["lam_hf"], lowpass_sigma=6.0,
                use_huber_recon=use_huber, huber_delta=0.1,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

    # Evaluate Pearson against raw hidden (in original space)
    pear, _ = evaluate(h_mean, hidden_raw)
    d = hidden_true_substitution(model, visible_proc, hidden_raw, device)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "h_mean": h_mean,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_log_huber")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== Beninca burst-fix ablation ===")
    print(f"Configs: {[c[0] for c in CONFIGS]}")
    print(f"{len(SPECIES_ORDER)} species × {len(CONFIGS)} configs × {len(SEEDS)} seeds")
    print(f"Total: {len(SPECIES_ORDER)*len(CONFIGS)*len(SEEDS)} runs\n")

    # results[config_name][species_name] = list of seed runs
    results = {cfg_name: {sp: [] for sp in SPECIES_ORDER}
               for cfg_name, _, _ in CONFIGS}

    total_runs = len(SPECIES_ORDER) * len(CONFIGS) * len(SEEDS)
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible_raw = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)

        print(f"\n--- hidden={h_name} ---")
        for cfg_name, use_log, use_huber in CONFIGS:
            for seed in SEEDS:
                run_i += 1
                t0 = datetime.now()
                try:
                    r = train_one(visible_raw, hidden_raw, seed, device,
                                    use_log, use_huber)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  [{run_i}/{total_runs}] {cfg_name:<22}  seed={seed}  "
                          f"P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "d_ratio": float("nan"),
                         "val_recon": float("nan"), "h_mean": None}
                r["seed"] = seed; r["config"] = cfg_name
                results[cfg_name][h_name].append(r)

    # Summary
    stage1b_ref = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
                    "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
                    "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}

    print(f"\n{'='*100}")
    print("BURST-FIX ABLATION RESULTS")
    print('='*100)
    header = f"{'Species':<18}"
    for cfg_name, _, _ in CONFIGS:
        header += f"{cfg_name:<22}"
    print(header)
    print('-' * 100)

    config_means = {cfg_name: [] for cfg_name, _, _ in CONFIGS}
    for h in SPECIES_ORDER:
        line = f"{h:<18}"
        for cfg_name, _, _ in CONFIGS:
            rs = results[cfg_name][h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            m = float(P.mean()) if len(P) else float("nan")
            if not np.isnan(m):
                config_means[cfg_name].append(m)
            line += f"{m:<+22.3f}"
        print(line)

    print('-' * 100)
    avg_line = f"{'Overall':<18}"
    for cfg_name, _, _ in CONFIGS:
        avg = np.mean(config_means[cfg_name])
        avg_line += f"{avg:<+22.4f}"
    print(avg_line)
    print('='*100)

    base_avg = np.mean(config_means["baseline (raw, MSE)"])
    best_cfg = max(CONFIGS, key=lambda c: np.mean(config_means[c[0]]))
    best_avg = np.mean(config_means[best_cfg[0]])
    print(f"\nStage 1b official ref: 0.132")
    print(f"Baseline (this run):   {base_avg:+.4f}")
    print(f"Best config:           {best_cfg[0]}  mean={best_avg:+.4f}  Δ={best_avg-base_avg:+.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
    x_pos = np.arange(len(SPECIES_ORDER))
    width = 0.2
    colors = ["#90a4ae", "#4fc3f7", "#ff8a65", "#c62828"]
    for i, (cfg_name, _, _) in enumerate(CONFIGS):
        vals = [np.mean([r["pearson"] for r in results[cfg_name][h]
                          if not np.isnan(r["pearson"])])
                for h in SPECIES_ORDER]
        ax.bar(x_pos + (i - 1.5) * width, vals, width,
               label=cfg_name, color=colors[i])
    # Stage 1b reference
    ax.plot(x_pos, [stage1b_ref[h] for h in SPECIES_ORDER], 'k*', markersize=10,
            label="Stage 1b ref (official)")
    ax.set_xticks(x_pos); ax.set_xticklabels(SPECIES_ORDER, rotation=25, fontsize=9)
    ax.set_ylabel("Pearson")
    ax.set_title(f"Beninca burst-fix ablation: baseline={base_avg:+.3f} → "
                 f"{best_cfg[0]}={best_avg:+.3f}", fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.25, axis="y")
    fig.savefig(out_dir / "fig_ablation.png", dpi=130, bbox_inches="tight")
    plt.close(fig)

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca burst-fix ablation\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("| Species | " + " | ".join(c[0] for c in CONFIGS) + " |\n")
        f.write("|---|" + "---|" * len(CONFIGS) + "\n")
        for h in SPECIES_ORDER:
            row = f"| {h}"
            for cfg_name, _, _ in CONFIGS:
                rs = results[cfg_name][h]
                P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
                m = float(P.mean()) if len(P) else float("nan")
                row += f" | {m:+.3f}"
            f.write(row + " |\n")
        f.write("\n**Overall means**:\n")
        for cfg_name, _, _ in CONFIGS:
            f.write(f"- {cfg_name}: {np.mean(config_means[cfg_name]):+.4f}\n")
        f.write(f"\nBest: **{best_cfg[0]}** = {best_avg:+.4f} "
                f"(Δ vs baseline = {best_avg-base_avg:+.4f})\n")

    def to_ser(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {cfg: {h: [{k: to_ser(v) for k, v in r.items() if k != "h_mean"}
                       for r in rs] for h, rs in d.items()}
            for cfg, d in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

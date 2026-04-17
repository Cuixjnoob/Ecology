"""MSP (Masked Species Prediction) + Alternating training.

Core idea: during Phase B, randomly mask 1 visible species and train
encoder to predict it. This gives h a 100% signal (not 0.2% residual).

Configs:
  A. baseline: joint training, no masking
  B. alt_5_1: alternating 5:1, no masking (previous best)
  C. msp_joint: joint training + random species masking every step
  D. msp_alt: alternating 5:1 + masking in Phase B only
  E. msp_alt_strong: alternating 5:1 + masking in Phase B + higher mask rate (2 species)
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

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate


SEEDS = [42, 123, 456]
EPOCHS = 500

HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_kl=0.017251789430967935,
    lam_cf=9.517725868477207,
    min_energy=0.14353013693386804,
)

SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

# (name, use_alt, pa, pb, mask_in_B, n_mask)
CONFIGS = [
    ("baseline",       False, 0, 0, False, 0),
    ("alt_5_1",        True,  5, 1, False, 0),
    ("msp_joint",      False, 0, 0, True,  1),   # mask every step
    ("msp_alt",        True,  5, 1, True,  1),   # mask only in Phase B
    ("msp_alt_2",      True,  5, 1, True,  2),   # mask 2 species in Phase B
]


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N,
        encoder_d=HP["encoder_d"], encoder_blocks=HP["encoder_blocks"],
        encoder_heads=4,
        takens_lags=HP["takens_lags"], encoder_dropout=HP["encoder_dropout"],
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
        hierarchical_h=False,
    ).to(device)


def get_param_groups(model):
    enc_ids = set()
    for name, p in model.named_parameters():
        if 'encoder' in name or 'readout' in name:
            enc_ids.add(id(p))
    fvis = [p for p in model.parameters() if id(p) not in enc_ids]
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    return fvis, enc


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def mask_species(x, n_mask, N_bio=9):
    """Randomly zero out n_mask biological species channels.
    x: (B, T, N_total). Only mask among first N_bio channels.
    Returns masked x and indices of masked species.
    """
    B, T, N = x.shape
    mask_idx = torch.randperm(N_bio)[:n_mask]
    x_masked = x.clone()
    for idx in mask_idx:
        x_masked[:, :, idx] = 0.0
    return x_masked, mask_idx


def train_one(visible, hidden, seed, device, cfg, epochs=EPOCHS):
    cfg_name, use_alt, pa, pb, do_mask, n_mask = cfg
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    fvis_params, enc_params = get_param_groups(model)
    opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=1e-4)

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
            h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))

        # Determine phase
        in_phase_B = False
        if use_alt and epoch >= warmup:
            cyc = (epoch - warmup) % (pa + pb)
            if cyc < pa:
                for p in enc_params: p.requires_grad_(False)
                for p in fvis_params: p.requires_grad_(True)
            else:
                for p in fvis_params: p.requires_grad_(False)
                for p in enc_params: p.requires_grad_(True)
                in_phase_B = True
        else:
            for p in fvis_params: p.requires_grad_(True)
            for p in enc_params: p.requires_grad_(True)

        # Input augmentation
        if epoch > warmup:
            dropout_mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * dropout_mask + (1 - dropout_mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full.clone()

        # MSP: mask species
        should_mask = False
        if do_mask and epoch >= warmup and n_mask > 0:
            if use_alt:
                should_mask = in_phase_B  # only mask in Phase B
            else:
                should_mask = True  # mask every step

        if should_mask:
            x_train, masked_idx = mask_species(x_train, n_mask, N_bio=min(8, N))
        else:
            masked_idx = None

        model.train(); opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=K_r)

        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]

        losses = model.loss(
            tr_out, beta_kl=HP["lam_kl"], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP["lam_cf"], lam_shuffle=HP["lam_cf"] * 0.6,
            lam_energy=2.0, min_energy=HP["min_energy"],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.2, lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )
        total = losses["total"]

        # MSP extra loss: reconstruction of masked species using ORIGINAL data as target
        if masked_idx is not None and len(masked_idx) > 0:
            # pred_full uses the masked input but targets should be from original
            pred = out["pred_full"]  # (S, B, T-1, N)
            safe_orig = torch.clamp(x_full, min=1e-6)
            actual_dlog = torch.log(safe_orig[:, 1:] / safe_orig[:, :-1])  # (B, T-1, N)
            # MSP loss: only on masked channels, train portion
            msp_loss = 0.0
            for idx in masked_idx:
                msp_loss = msp_loss + ((pred[:, :, :train_end, idx] -
                                         actual_dlog[:, :train_end, idx].unsqueeze(0)) ** 2).mean()
            msp_loss = msp_loss / len(masked_idx)
            total = total + 2.0 * msp_loss  # weight MSP loss

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
                lam_energy=2.0, min_energy=HP["min_energy"],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.2, lowpass_sigma=6.0,
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    for p in model.parameters(): p.requires_grad_(True)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pearson": pear, "val_recon": best_val, "h_mean": h_mean.tolist()}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_msp")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== MSP + Alternating training ===")
    for c in CONFIGS:
        print(f"  {c[0]}: alt={c[1]}, pa={c[2]}, pb={c[3]}, mask={c[4]}, n_mask={c[5]}")
    total_runs = len(SPECIES_ORDER) * len(CONFIGS) * len(SEEDS)
    print(f"Total: {total_runs} runs\n")

    results = {c[0]: {sp: [] for sp in SPECIES_ORDER} for c in CONFIGS}
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        print(f"\n--- hidden={h_name} ---")
        for cfg in CONFIGS:
            for seed in SEEDS:
                run_i += 1
                t0 = datetime.now()
                try:
                    r = train_one(visible, hidden_raw, seed, device, cfg)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  [{run_i}/{total_runs}] {cfg[0]:<16} seed={seed}  "
                          f"P={r['pearson']:+.3f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "val_recon": float("nan"),
                         "h_mean": None}
                r["seed"] = seed; r["config"] = cfg[0]
                results[cfg[0]][h_name].append(r)

    # Summary
    print(f"\n{'='*100}")
    print("MSP RESULTS")
    print('='*100)
    header = f"{'Species':<16}"
    for c in CONFIGS:
        header += f"{c[0]:<16}"
    print(header)
    print('-'*100)

    config_means = {c[0]: [] for c in CONFIGS}
    for sp in SPECIES_ORDER:
        line = f"{sp:<16}"
        for c in CONFIGS:
            rs = results[c[0]][sp]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            m = float(P.mean()) if len(P) else float("nan")
            if not np.isnan(m):
                config_means[c[0]].append(m)
            line += f"{m:<+16.3f}"
        print(line)

    print('-'*100)
    avg_line = f"{'Overall':<16}"
    for c in CONFIGS:
        avg = np.mean(config_means[c[0]]) if config_means[c[0]] else float("nan")
        avg_line += f"{avg:<+16.4f}"
    print(avg_line)

    best_cfg = max(CONFIGS, key=lambda c: np.mean(config_means[c[0]]) if config_means[c[0]] else -999)
    best_avg = np.mean(config_means[best_cfg[0]])
    print(f"\nBest: {best_cfg[0]} = {best_avg:+.4f} (S1b ref: +0.132)")

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# MSP + Alternating training experiment\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("| Config | alt | pa | pb | mask | n_mask |\n|---|---|---|---|---|---|\n")
        for c in CONFIGS:
            f.write(f"| {c[0]} | {c[1]} | {c[2]} | {c[3]} | {c[4]} | {c[5]} |\n")
        f.write(f"\n| Species |" + "|".join(f" {c[0]} " for c in CONFIGS) + "|\n")
        f.write("|---|" + "---|" * len(CONFIGS) + "\n")
        for sp in SPECIES_ORDER:
            row = f"| {sp}"
            for c in CONFIGS:
                rs = results[c[0]][sp]
                P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
                m = float(P.mean()) if len(P) else float("nan")
                row += f" | {m:+.3f}"
            f.write(row + " |\n")
        f.write(f"\n**Overall**: " + ", ".join(
            f"{c[0]}={np.mean(config_means[c[0]]):+.4f}" for c in CONFIGS) + "\n")
        f.write(f"\nBest: **{best_cfg[0]}** = {best_avg:+.4f}\n")

    # Save raw with h_mean
    def to_ser(v):
        if isinstance(v, (int, float, np.floating)):
            return float(v)
        if isinstance(v, np.ndarray):
            return v.tolist()
        return v
    dump = {cfg: {sp: [{k: to_ser(v) for k, v in r.items()}
                        for r in rs] for sp, rs in d.items()}
            for cfg, d in results.items()}
    with open(out_dir / "raw.json", "w") as f:
        json.dump(dump, f, default=float)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

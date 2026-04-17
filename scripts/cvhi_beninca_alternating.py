"""Alternating training: fix f_visible, train encoder; then vice versa.

Root cause: joint training gives h near-zero gradient (hidden explains <0.2% of visible).
Fix: alternating optimization so residual signal flows directly to encoder.

Phase A (3 ep): train f_visible + G_field, freeze encoder. h_weight per schedule.
Phase B (1 ep): freeze f_visible + G_field, train encoder. h_weight=1.

Compare:
  - baseline: joint training (Stage 1b)
  - alt_3_1: alternate 3:1
  - alt_5_1: alternate 5:1
  - alt_pretrain: pretrain f_visible 200ep alone, then alternate 3:1
"""
from __future__ import annotations
import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
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

# (name, pretrain_fvis_epochs, phase_a_epochs, phase_b_epochs)
CONFIGS = [
    ("baseline",      0, 0, 0),       # joint training (no alternating)
    ("alt_3_1",       0, 3, 1),
    ("alt_5_1",       0, 5, 1),
    ("pretrain+alt",  150, 3, 1),     # pretrain f_visible 150ep, then alternate
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
    """Split params into f_visible/G vs encoder groups."""
    encoder_params = set()
    for name, p in model.named_parameters():
        if 'encoder' in name or 'readout' in name:
            encoder_params.add(id(p))

    fvis_params = [p for p in model.parameters() if id(p) not in encoder_params]
    enc_params = [p for p in model.parameters() if id(p) in encoder_params]
    return fvis_params, enc_params


def freeze(params):
    for p in params:
        p.requires_grad_(False)

def unfreeze(params):
    for p in params:
        p.requires_grad_(True)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def compute_loss(model, x_full, train_end, T, h_w, K_r, epoch, warmup, epochs, device):
    if epoch > warmup:
        mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
        x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
    else:
        x_train = x_full

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
    # Compute val
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
    return losses, val_losses["recon_full"].item()


def train_one(visible, hidden, seed, device, cfg_name, pretrain_ep, pa, pb, epochs=EPOCHS):
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

    is_alternating = (pa > 0 and pb > 0)

    # Phase 0: optional f_visible pretraining
    if pretrain_ep > 0:
        freeze(enc_params)
        for epoch in range(pretrain_ep):
            model.train(); opt.zero_grad()
            out = model(x_full, n_samples=1, rollout_K=0)
            tr = model.slice_out(out, 0, train_end)
            tr["visible"] = out["visible"][:, :train_end]
            tr["G"] = out["G"][:, :train_end]
            losses = model.loss(tr, h_weight=0.0, beta_kl=0.0, lam_smooth=0.0,
                                 lam_energy=0, min_energy=0, lam_hf=0,
                                 lam_necessary=0, lam_shuffle=0,
                                 lam_sparse=0.01, lam_rmse_log=0.1)
            losses["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        unfreeze(enc_params)
        # Reset optimizer state after pretraining
        opt = torch.optim.AdamW(model.parameters(), lr=HP["lr"], weight_decay=1e-4)
        sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    # Main training loop
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

        # Determine phase for alternating
        if is_alternating and epoch >= warmup:
            cycle_pos = (epoch - warmup) % (pa + pb)
            if cycle_pos < pa:
                # Phase A: train f_visible, freeze encoder
                freeze(enc_params)
                unfreeze(fvis_params)
            else:
                # Phase B: freeze f_visible, train encoder
                freeze(fvis_params)
                unfreeze(enc_params)
        else:
            # Joint training (baseline or warmup)
            unfreeze(fvis_params)
            unfreeze(enc_params)

        model.train(); opt.zero_grad()
        losses, val_recon = compute_loss(model, x_full, train_end, T,
                                          h_w, K_r, epoch, warmup, epochs, device)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Restore all params for eval
    unfreeze(fvis_params)
    unfreeze(enc_params)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

        # Check margin
        recon_full_val = ((out_eval["pred_full"] -
            torch.log(torch.clamp(out_eval["visible"][:,1:]/
                      torch.clamp(out_eval["visible"][:,:-1], min=1e-6), min=1e-6)
            ).unsqueeze(0))**2).mean().item()
        recon_null_val = ((out_eval["pred_null"] -
            torch.log(torch.clamp(out_eval["visible"][:,1:]/
                      torch.clamp(out_eval["visible"][:,:-1], min=1e-6), min=1e-6)
            ).unsqueeze(0))**2).mean().item()

    pear, _ = evaluate(h_mean, hidden)
    margin = recon_null_val - recon_full_val
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pearson": pear, "val_recon": best_val,
            "margin": margin, "recon_full": recon_full_val, "recon_null": recon_null_val}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_alternating")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, _ = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== Alternating training experiment ===")
    for name, pre, a, b in CONFIGS:
        if a == 0:
            print(f"  {name}: joint training (no alternating)")
        else:
            print(f"  {name}: pretrain={pre}ep, alternate {a}:{b}")
    total_runs = len(SPECIES_ORDER) * len(CONFIGS) * len(SEEDS)
    print(f"Total: {total_runs} runs\n")

    results = {c[0]: {sp: [] for sp in SPECIES_ORDER} for c in CONFIGS}
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        print(f"\n--- hidden={h_name} ---")
        for cfg_name, pretrain, pa, pb in CONFIGS:
            for seed in SEEDS:
                run_i += 1
                t0 = datetime.now()
                try:
                    r = train_one(visible, hidden_raw, seed, device,
                                  cfg_name, pretrain, pa, pb)
                    dt = (datetime.now() - t0).total_seconds()
                    print(f"  [{run_i}/{total_runs}] {cfg_name:<16} seed={seed}  "
                          f"P={r['pearson']:+.3f}  margin={r['margin']:.4f}  ({dt:.1f}s)")
                except Exception as e:
                    print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                    import traceback; traceback.print_exc()
                    r = {"pearson": float("nan"), "val_recon": float("nan"),
                         "margin": float("nan"), "recon_full": float("nan"),
                         "recon_null": float("nan")}
                r["seed"] = seed; r["config"] = cfg_name
                results[cfg_name][h_name].append(r)

    # Summary
    print(f"\n{'='*100}")
    print("ALTERNATING TRAINING RESULTS")
    print('='*100)
    header = f"{'Species':<16}"
    for c in CONFIGS:
        header += f"{c[0]:<16}"
    print(header)
    print('-'*100)

    config_means = {c[0]: [] for c in CONFIGS}
    config_margins = {c[0]: [] for c in CONFIGS}
    for sp in SPECIES_ORDER:
        line = f"{sp:<16}"
        for c in CONFIGS:
            rs = results[c[0]][sp]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            m = float(P.mean()) if len(P) else float("nan")
            margins = [r["margin"] for r in rs if not np.isnan(r.get("margin", float("nan")))]
            if not np.isnan(m):
                config_means[c[0]].append(m)
            if margins:
                config_margins[c[0]].extend(margins)
            line += f"{m:<+16.3f}"
        print(line)

    print('-'*100)
    avg_line = f"{'Overall':<16}"
    for c in CONFIGS:
        avg = np.mean(config_means[c[0]]) if config_means[c[0]] else float("nan")
        avg_line += f"{avg:<+16.4f}"
    print(avg_line)

    margin_line = f"{'Avg margin':<16}"
    for c in CONFIGS:
        mg = np.mean(config_margins[c[0]]) if config_margins[c[0]] else float("nan")
        margin_line += f"{mg:<16.4f}"
    print(margin_line)

    best_cfg = max(CONFIGS, key=lambda c: np.mean(config_means[c[0]]) if config_means[c[0]] else -999)
    best_avg = np.mean(config_means[best_cfg[0]])
    print(f"\nBest: {best_cfg[0]} = {best_avg:+.4f} (S1b ref: +0.132)")

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Alternating training experiment\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("| Config | pretrain | phase_A | phase_B |\n|---|---|---|---|\n")
        for name, pre, a, b in CONFIGS:
            f.write(f"| {name} | {pre} | {a} | {b} |\n")
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
        f.write(f"\n**Avg margin**: " + ", ".join(
            f"{c[0]}={np.mean(config_margins[c[0]]):.4f}" for c in CONFIGS) + "\n")
        f.write(f"\nBest: **{best_cfg[0]}** = {best_avg:+.4f}\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

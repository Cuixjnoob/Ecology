"""Alternating training (5:1) with nutrients-as-input + burst P/R/F evaluation.

Changes vs cvhi_beninca_alternating.py:
  1. Nutrients are input-only: recon loss only on species channels, not nutrients
  2. Burst binary classification metrics: Precision / Recall / F-score
     on hidden species recovery (evaluation only, no supervision violation)
  3. Uses alt_5_1 (current best alternating config)
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

# 9 biological species (nutrients excluded from rotation)
SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]

# Alternating 5:1 (current best)
PA, PB = 5, 1


# --------------- Burst evaluation (Racca & Magri inspired) ---------------

def find_burst_mask(series, pct=10, eps=1e-6):
    """Binary mask: 1 at timesteps with |d log x| in top `pct`%."""
    safe = np.maximum(np.abs(series), eps)
    log_x = np.log(safe)
    dlog = np.abs(np.diff(log_x))        # (T-1,)
    if len(dlog) == 0:
        return np.zeros(len(series), dtype=bool)
    threshold = np.percentile(dlog, 100 - pct)
    burst = np.concatenate([dlog > threshold, [False]])  # align to T
    return burst


def burst_precision_recall(h_recovered, hidden_true, pct=10):
    """Compute burst P/R/F between recovered and true hidden species.

    - burst_true: timesteps where true hidden has |d log x| in top pct%
    - burst_pred: timesteps where recovered h has |d log x| in top pct%
    Returns dict with precision, recall, f_score.
    """
    burst_true = find_burst_mask(hidden_true, pct=pct)
    burst_pred = find_burst_mask(h_recovered, pct=pct)

    tp = int((burst_true & burst_pred).sum())
    fp = int((~burst_true & burst_pred).sum())
    fn = int((burst_true & ~burst_pred).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f_score = 2 * precision * recall / max(precision + recall, 1e-8)

    return {
        "burst_precision": precision,
        "burst_recall": recall,
        "burst_f_score": f_score,
        "burst_tp": tp, "burst_fp": fp, "burst_fn": fn,
        "n_true_bursts": int(burst_true.sum()),
        "n_pred_bursts": int(burst_pred.sum()),
    }


# --------------- Model & training ---------------

def make_model(N, n_species, device):
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


def compute_loss(model, x_full, train_end, T, h_w, K_r, epoch, warmup, epochs,
                 device, n_species):
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
        n_recon_channels=n_species,       # <-- only species in recon loss
    )
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
            n_recon_channels=n_species,   # <-- same
        )
    return losses, val_losses["recon_full"].item()


def train_one(visible, hidden, seed, device, n_species, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, n_species, device)
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

        # Alternating phase
        if epoch >= warmup:
            cycle_pos = (epoch - warmup) % (PA + PB)
            if cycle_pos < PA:
                freeze(enc_params); unfreeze(fvis_params)
            else:
                freeze(fvis_params); unfreeze(enc_params)
        else:
            unfreeze(fvis_params); unfreeze(enc_params)

        model.train(); opt.zero_grad()
        losses, val_recon = compute_loss(
            model, x_full, train_end, T, h_w, K_r, epoch, warmup, epochs,
            device, n_species)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()

        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    # Eval
    unfreeze(fvis_params); unfreeze(enc_params)
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

        vis = out_eval["visible"]
        safe_v = torch.clamp(vis, min=1e-6)
        log_ratio = torch.log(safe_v[:, 1:] / safe_v[:, :-1]).unsqueeze(0)
        recon_full_val = ((out_eval["pred_full"][..., :n_species] -
                           log_ratio[..., :n_species]) ** 2).mean().item()
        recon_null_val = ((out_eval["pred_null"][..., :n_species] -
                           log_ratio[0, ..., :n_species]) ** 2).mean().item()

    pear, h_scaled = evaluate(h_mean, hidden)
    margin = recon_null_val - recon_full_val

    # Burst evaluation
    burst_metrics = burst_precision_recall(h_scaled.flatten(), hidden, pct=10)

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {"pearson": pear, "val_recon": best_val,
            "margin": margin, **burst_metrics}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_nutrient_input")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, _ = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]

    # Identify species vs nutrient channels
    n_species = len([s for s in species if s in SPECIES_ORDER])
    nutrient_names = [s for s in species if s not in SPECIES_ORDER]
    print(f"Species: {n_species}, Nutrients: {nutrient_names}")
    print(f"Recon loss on first {n_species} channels only (nutrients = input-only)")
    print(f"Alternating training: {PA}:{PB}")
    print(f"Burst evaluation: top 10% |d log x| as extreme events\n")

    total_runs = len(SPECIES_ORDER) * len(SEEDS)
    print(f"Total: {total_runs} runs ({len(SPECIES_ORDER)} species x {len(SEEDS)} seeds)\n")

    results = {sp: [] for sp in SPECIES_ORDER}
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        # visible = all channels except hidden species (8 species + 4 nutrients = 12 ch)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        # n_species for this rotation: n_species - 1 (removed one species)
        n_sp_vis = n_species - 1
        print(f"--- hidden={h_name} (visible: {n_sp_vis} species + {len(nutrient_names)} nutrients = {visible.shape[1]} ch) ---")

        for seed in SEEDS:
            run_i += 1
            t0 = datetime.now()
            try:
                r = train_one(visible, hidden_raw, seed, device, n_sp_vis)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  [{run_i}/{total_runs}] seed={seed}  "
                      f"P={r['pearson']:+.3f}  "
                      f"burst_F={r['burst_f_score']:.3f} "
                      f"(P={r['burst_precision']:.2f} R={r['burst_recall']:.2f})  "
                      f"margin={r['margin']:.4f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "val_recon": float("nan"),
                     "margin": float("nan"),
                     "burst_precision": 0, "burst_recall": 0, "burst_f_score": 0,
                     "burst_tp": 0, "burst_fp": 0, "burst_fn": 0,
                     "n_true_bursts": 0, "n_pred_bursts": 0}
            r["seed"] = seed
            results[h_name].append(r)

    # Summary
    print(f"\n{'='*110}")
    print("RESULTS: Alternating 5:1 + Nutrients-as-Input + Burst Eval")
    print('='*110)
    print(f"{'Species':<16} {'Pearson':>10} {'Burst_F':>10} {'Burst_P':>10} {'Burst_R':>10} {'Margin':>10}")
    print('-'*110)

    all_pear = []; all_bf = []; all_bp = []; all_br = []
    for sp in SPECIES_ORDER:
        rs = results[sp]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        BF = np.array([r["burst_f_score"] for r in rs])
        BP = np.array([r["burst_precision"] for r in rs])
        BR = np.array([r["burst_recall"] for r in rs])
        MG = np.array([r["margin"] for r in rs if not np.isnan(r.get("margin", float("nan")))])
        mp = float(P.mean()) if len(P) else float("nan")
        mbf = float(BF.mean()) if len(BF) else 0
        mbp = float(BP.mean()) if len(BP) else 0
        mbr = float(BR.mean()) if len(BR) else 0
        mmg = float(MG.mean()) if len(MG) else float("nan")
        if not np.isnan(mp): all_pear.append(mp)
        all_bf.append(mbf); all_bp.append(mbp); all_br.append(mbr)
        print(f"{sp:<16} {mp:>+10.3f} {mbf:>10.3f} {mbp:>10.3f} {mbr:>10.3f} {mmg:>10.4f}")

    print('-'*110)
    op = np.mean(all_pear) if all_pear else float("nan")
    obf = np.mean(all_bf); obp = np.mean(all_bp); obr = np.mean(all_br)
    print(f"{'Overall':<16} {op:>+10.4f} {obf:>10.3f} {obp:>10.3f} {obr:>10.3f}")
    print(f"\n(S1b ref: Pearson +0.132, no burst metrics)")

    # Save summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Alternating 5:1 + Nutrients-as-Input + Burst P/R/F\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}, Alt: {PA}:{PB}\n")
        f.write(f"Recon loss: species-only ({n_species-1} visible species), nutrients input-only\n")
        f.write(f"Burst threshold: top 10% |d log x|\n\n")
        f.write("| Species | Pearson | Burst_F | Burst_P | Burst_R | Margin |\n")
        f.write("|---|---|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            rs = results[sp]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            BF = np.array([r["burst_f_score"] for r in rs])
            BP = np.array([r["burst_precision"] for r in rs])
            BR = np.array([r["burst_recall"] for r in rs])
            MG = np.array([r["margin"] for r in rs if not np.isnan(r.get("margin", float("nan")))])
            f.write(f"| {sp} | {P.mean():+.3f} | {BF.mean():.3f} | "
                    f"{BP.mean():.3f} | {BR.mean():.3f} | {MG.mean():.4f} |\n")
        f.write(f"\n**Overall**: Pearson={op:+.4f}, Burst_F={obf:.3f}, "
                f"Burst_P={obp:.3f}, Burst_R={obr:.3f}\n")
        f.write(f"\nRef: S1b Pearson = +0.132\n")

    # Save raw results
    raw = {}
    for sp in SPECIES_ORDER:
        raw[sp] = []
        for r in results[sp]:
            raw[sp].append({k: (float(v) if isinstance(v, (int, float, np.floating)) else v)
                            for k, v in r.items()})
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

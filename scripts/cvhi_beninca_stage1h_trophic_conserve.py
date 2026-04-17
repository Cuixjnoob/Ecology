"""Stage 1h - Trophic-layer group conservation.

Groups visible channels by trophic role (from Benincà 2008 Fig 1a food web):
  G1 [Phyto]:    Picophyto + Nanophyto + Filam_diatoms
  G2 [Pelagic]:  Cyclopoids + Calanoids + Rotifers
  G3 [Detrital]: Bacteria + Ostracods + Harpacticoids
  G4 [Nutrients]: NO2 + NO3 + NH4 + SRP

Each group has its own learnable scale c_g (signed).
Loss: L_conserve = mean_g Var_t(sum_{j in g} x_j + c_g * h)

Advantages over Stage 1f/1g:
  - Filam can't dominate (it's 1/3 of phyto group)
  - Each group still has strong signal (3 species)
  - 4 c_g values allow encoder to self-identify which trophic
    compartment hidden belongs to
  - Aligns with real ecological conservation (layer-wise balance)

Red line:
  - Trophic assignment uses only visible species (hidden's trophic
    position NOT encoded anywhere)
  - When hidden is rotated, its group loses one member naturally
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution


SEEDS = [42, 123, 456, 789, 2024]
EPOCHS = 500
LAMBDA_CONSERVE = 0.1

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

# Trophic-layer assignment (from Benincà 2008 Fig 1a food web)
TROPHIC = {
    "Picophyto":     "phyto",
    "Nanophyto":     "phyto",
    "Filam_diatoms": "phyto",
    "Cyclopoids":    "pelagic",
    "Calanoids":     "pelagic",
    "Rotifers":      "pelagic",
    "Bacteria":      "detrital",
    "Ostracods":     "detrital",
    "Harpacticoids": "detrital",
    "NO2":           "nutrient",
    "NO3":           "nutrient",
    "NH4":           "nutrient",
    "SRP":           "nutrient",
}
GROUP_NAMES = ["phyto", "pelagic", "detrital", "nutrient"]
GROUP_IDS = {g: i for i, g in enumerate(GROUP_NAMES)}


def build_group_mask(visible_channels):
    """Return (N_vis, 4) boolean mask: mask[j, g] = True iff channel j in group g."""
    N = len(visible_channels)
    mask = np.zeros((N, 4), dtype=np.float32)
    for j, ch in enumerate(visible_channels):
        if ch in TROPHIC:
            gid = GROUP_IDS[TROPHIC[ch]]
            mask[j, gid] = 1.0
    return mask


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


def train_one(visible, hidden, seed, visible_channels, device,
              lam_conserve=LAMBDA_CONSERVE, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    # Group mask: (N, 4)
    group_mask = build_group_mask(visible_channels)
    group_mask_t = torch.tensor(group_mask, dtype=torch.float32, device=device)  # (N, 4)
    # Group counts (for averaging within group)
    group_counts = group_mask_t.sum(dim=0) + 1e-6                                 # (4,)

    model = make_model(N, device)
    # 4 learnable scales, one per group, signed
    c_group = nn.Parameter(torch.zeros(4, device=device))
    params = list(model.parameters()) + [c_group]
    opt = torch.optim.AdamW(params, lr=BEST_HP["lr"], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_c = None

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
            dp_mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * dp_mask + (1 - dp_mask) * x_full.mean(dim=1, keepdim=True)
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
        )
        total = losses["total"]

        # === Trophic-layer conservation ===
        if lam_conserve > 0 and h_w > 0:
            x_slice = x_train[:, :train_end]                    # (1, T_train, N)
            # Group totals: (1, T_train, 4)
            # group_mask_t: (N, 4) so x_slice @ group_mask_t → (1, T_train, 4)
            group_totals = torch.einsum("btn,ng->btg", x_slice, group_mask_t)
            h_mean_post = tr_out["h_samples"].mean(dim=0)       # (1, T_train)
            # adjusted: (1, T_train, 4) = group_totals + c_group * h
            adjusted = group_totals + c_group.view(1, 1, 4) * h_mean_post.unsqueeze(-1)
            # Variance across time, per group, mean across groups
            L_conserve = adjusted.var(dim=1).mean()
            total = total + lam_conserve * h_w * L_conserve

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
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
            )
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_c = c_group.detach().cpu().clone()

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    final_c = best_c.numpy().tolist() if best_c is not None else [float("nan")] * 4
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "c_group": final_c,   # [phyto, pelagic, detrital, nutrient]
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_stage1h")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()
    species = [str(s) for s in species]

    print(f"\n=== Stage 1h: Trophic-layer Conservation ===")
    print(f"Groups: phyto(Pico+Nano+Filam), pelagic(Cyc+Cal+Rot),")
    print(f"        detrital(Bact+Ostr+Harp), nutrient(NO2+NO3+NH4+SRP)")
    print(f"lambda={LAMBDA_CONSERVE}")

    all_results = {}
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible_arr = np.delete(full, h_idx, axis=1)
        hidden_arr = full[:, h_idx]
        visible_channels = [s for i, s in enumerate(species) if i != h_idx]

        # Show group composition for this rotation
        hidden_group = TROPHIC.get(h_name, "?")
        print(f"\n--- hidden={h_name} (group:{hidden_group}) ---")
        rs = []
        for s in SEEDS:
            t0 = datetime.now()
            try:
                r = train_one(visible_arr, hidden_arr, s, visible_channels, device)
                dt = (datetime.now() - t0).total_seconds()
                c_str = f"[phy={r['c_group'][0]:+.2f}, pel={r['c_group'][1]:+.2f}, " \
                        f"det={r['c_group'][2]:+.2f}, nut={r['c_group'][3]:+.2f}]"
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  "
                      f"val={r['val_recon']:.4f}  c={c_str}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  seed={s}  FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "d_ratio": float("nan"),
                     "val_recon": float("nan"), "c_group": []}
            r["seed"] = s
            rs.append(r)
        all_results[h_name] = rs

    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    stage1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}
    stage1f = {"Cyclopoids": 0.074, "Calanoids": 0.044, "Rotifers": 0.103,
               "Nanophyto": 0.069, "Picophyto": 0.037, "Filam_diatoms": 0.043,
               "Ostracods": 0.207, "Harpacticoids": 0.258, "Bacteria": 0.197}

    print(f"\n{'='*110}")
    print("STAGE 1h RESULTS - Trophic-layer Conservation")
    print('='*110)
    print(f"{'Species':<18}{'group':<12}{'S1b':<10}{'S1f':<10}{'S1h':<12}{'D vs S1b':<12}")
    sum_s1b=0; sum_s1h=0; cnt=0
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s1h = float(P.mean()) if len(P) else float("nan")
        grp = TROPHIC.get(h, "?")
        print(f"{h:<18}{grp:<12}{stage1b[h]:<+10.3f}{stage1f[h]:<+10.3f}{s1h:<+12.3f}"
              f"{s1h-stage1b[h]:<+12.3f}")
        if not np.isnan(s1h):
            sum_s1b += stage1b[h]; sum_s1h += s1h; cnt += 1

    avg_s1b = sum_s1b/cnt; avg_s1h = sum_s1h/cnt
    print(f"\nOverall:  S1b={avg_s1b:+.4f}  S1h={avg_s1h:+.4f}  D={avg_s1h-avg_s1b:+.4f}")

    if avg_s1h > avg_s1b + 0.005:
        verdict = "[OK] Stage 1h better than S1b"
    elif avg_s1h > avg_s1b - 0.005:
        verdict = "[NEUTRAL] Stage 1h ~= S1b"
    else:
        verdict = "[FAIL] Stage 1h worse than S1b"
    print(f"\n{verdict}\n{'='*110}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1h - Trophic-layer Conservation\n\n")
        f.write(f"Seeds: {len(SEEDS)}, Epochs: {EPOCHS}, lambda={LAMBDA_CONSERVE}\n\n")
        f.write("Groups: phyto(3), pelagic(3), detrital(3), nutrient(4)\n\n")
        f.write("## Results\n\n")
        f.write("| Species | group | S1b | S1f | **S1h** | D vs S1b |\n|---|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s1h = float(P.mean()) if len(P) else float("nan")
            grp = TROPHIC.get(h, "?")
            f.write(f"| {h} | {grp} | {stage1b[h]:+.3f} | {stage1f[h]:+.3f} | "
                    f"**{s1h:+.3f}** | {s1h-stage1b[h]:+.3f} |\n")
        f.write(f"\n**Overall**: S1b={avg_s1b:+.4f}, **S1h={avg_s1h:+.4f}**, "
                f"D={avg_s1h-avg_s1b:+.4f}\n\n")
        f.write(f"**{verdict}**\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

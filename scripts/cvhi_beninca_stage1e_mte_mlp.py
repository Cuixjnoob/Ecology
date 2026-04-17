"""Stage 1e — MTE + MLP hybrid on f_visible intrinsic rate.

Fixed Stage 1c's failure: Bacteria outlier dominates 9-point Pearson, rate ordering ill-posed.

Stage 1e design (based on Kremer 2017 + Clarke 2025 literature review):
  - **Intercept per taxa group is learnable** (Kremer Table 2: group intercepts differ 2 orders of magnitude)
  - **b (mass exponent) per taxa is fixed** (Glazier / Kremer give stable group values)
  - **MLP learns small species-level delta** (low-capacity, anchored to 0)
  - **Same correlation-distance loss** as Stage 1c, but target is now adaptive

Strict unsupervised:
  - Hidden's taxa / body mass NEVER enters loss
  - Only visible species' known body mass + taxa group used
  - MLP input = [taxa_one_hot_visible, log10_M_visible] for visible species only

Taxa grouping (4 groups for Beninca 9 species):
  0 Bacteria    — {Bacteria}                                 b=1.00, B0=+0.50
  1 Phyto       — {Nanophyto, Picophyto, Filam_diatoms}      b=0.95, B0=+0.00
  2 Pelagic-zoo — {Cyclopoids, Calanoids, Rotifers}          b=0.88, B0=-0.20
  3 Benthic     — {Ostracods, Harpacticoids}                 b=0.75, B0=-0.40

Loss structure:
  L_total = model_loss (含 correlation distance on shape)
          + λ_anchor · || log_B0_taxa - KREMER_INIT ||²     (prevent drift)
          + λ_mlp · || MLP_delta_output ||²                  (MLP stays small)
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

# Visible species body mass (μg dry weight)
BODY_MASS_UG = {
    "Bacteria":      1e-6,
    "Picophyto":     1e-5,
    "Nanophyto":     1e-3,
    "Filam_diatoms": 1e-2,
    "Rotifers":      0.5,
    "Harpacticoids": 5.0,
    "Cyclopoids":    20.0,
    "Calanoids":     50.0,
    "Ostracods":     50.0,
}

# Taxa grouping (strict unsupervised — only for visible species)
TAXA_MAP = {
    "Bacteria":      0,   # prokaryote
    "Picophyto":     1,   # photoautotroph
    "Nanophyto":     1,
    "Filam_diatoms": 1,
    "Rotifers":      2,   # pelagic zoo
    "Cyclopoids":    2,
    "Calanoids":     2,
    "Harpacticoids": 3,   # benthic
    "Ostracods":     3,
}

# Kremer 2017 / Clarke 2025 — taxa-specific b (mass scaling exponent), FIXED
TAXA_B = torch.tensor([1.00, 0.95, 0.88, 0.75])   # bact, phyto, pelagic, benthic

# Kremer 2017 Table 2 — taxa intercept initialization (log_B0 in natural units)
# Values are relative; absorbed by correlation distance anyway. Set so that
# Kremer's group-intercept structure is roughly preserved.
KREMER_B0_INIT = torch.tensor([+0.50, +0.00, -0.20, -0.40])   # bact, phyto, pelagic, benthic


class MTE_MLP(nn.Module):
    """Learnable taxa-B0 + small MLP-based species delta. 4 taxa + small MLP.

    Output: log_r_target_i = log_B0_taxa[taxa_i] + (b_taxa[taxa_i] - 1) * log10(M_i) + delta_MLP_i
    """
    def __init__(self, n_taxa: int = 4, hidden: int = 12, device: str = "cpu"):
        super().__init__()
        self.log_B0_taxa = nn.Parameter(KREMER_B0_INIT.clone())   # (4,)
        self.register_buffer("b_taxa", TAXA_B.clone())             # (4,) fixed
        # Small MLP: input = [taxa_one_hot (4), log_M (1)] = 5 dims
        self.mlp = nn.Sequential(
            nn.Linear(n_taxa + 1, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        # Init MLP to ~0 output
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                nn.init.zeros_(m.bias)
        self.to(device)

    def forward(self, taxa_ids: torch.Tensor, log_M: torch.Tensor) -> tuple:
        """taxa_ids: (N_vis,) long, log_M: (N_vis,) float.
        Returns (log_r_target, delta_mlp) both shape (N_vis,).
        """
        dev = self.log_B0_taxa.device
        taxa_ids = taxa_ids.to(dev)
        log_M = log_M.to(dev)
        taxa_1h = F.one_hot(taxa_ids, num_classes=4).float()  # (N_vis, 4)
        feat = torch.cat([taxa_1h, log_M.unsqueeze(-1)], dim=-1)  # (N_vis, 5)
        delta = self.mlp(feat).squeeze(-1)                      # (N_vis,)
        b_i = self.b_taxa[taxa_ids]
        log_B0 = self.log_B0_taxa[taxa_ids]
        log_r = log_B0 + (b_i - 1.0) * log_M + delta
        return log_r, delta


def build_visible_info(visible_species):
    """visible_species is list of channel names (order matching data columns).
    Returns (taxa_ids: long (N,), log_M: float (N,), is_species_mask: bool (N,))
    Nutrients (not in TAXA_MAP) get taxa=0, log_M=0, mask=False.
    """
    taxa = []; logM = []; mask = []
    for s in visible_species:
        if s in TAXA_MAP:
            taxa.append(TAXA_MAP[s])
            logM.append(float(np.log10(BODY_MASS_UG[s])))
            mask.append(True)
        else:
            taxa.append(0)  # placeholder
            logM.append(0.0)
            mask.append(False)
    return (torch.tensor(taxa, dtype=torch.long),
            torch.tensor(logM, dtype=torch.float32),
            torch.tensor(mask, dtype=torch.bool))


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


def train_one(visible, hidden, seed, visible_species, device,
              lam_mte_shape=0.02, lam_anchor=0.05, lam_mlp_reg=0.1,
              epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    mte_mlp = MTE_MLP(n_taxa=4, hidden=12, device=device)

    taxa_ids, log_M, sp_mask = build_visible_info(visible_species)
    taxa_ids = taxa_ids.to(device)
    log_M = log_M.to(device)
    sp_mask = sp_mask.to(device)

    params = list(model.parameters()) + list(mte_mlp.parameters())
    opt = torch.optim.AdamW(params, lr=BEST_HP["lr"], weight_decay=1e-4)

    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_mte_state = None

    KREMER_INIT_DEV = KREMER_B0_INIT.to(device)

    for epoch in range(epochs):
        # G_anchor annealing
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

        # Input dropout aug
        if epoch > warmup:
            dp_mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * dp_mask + (1 - dp_mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        # Compute MTE target from current MLP
        log_r_pred, delta_mlp = mte_mlp(taxa_ids, log_M)
        # Apply NaN mask for non-species channels (nutrients)
        mte_target = torch.where(sp_mask, log_r_pred, torch.full_like(log_r_pred, float("nan")))

        model.train(); mte_mlp.train(); opt.zero_grad()
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
            lam_mte_shape=lam_mte_shape,            # correlation distance on shape
            mte_target_log_r=mte_target,             # adaptive target from MLP
        )
        total = losses["total"]

        # Anchor: keep taxa B_0 near Kremer init
        L_anchor = F.mse_loss(mte_mlp.log_B0_taxa, KREMER_INIT_DEV)
        # MLP-delta regularization: keep MLP output small
        L_mlp_reg = (delta_mlp ** 2).mean()
        total = total + lam_anchor * L_anchor + lam_mlp_reg * L_mlp_reg

        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step()

        # Validation
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
            best_mte_state = {k: v.detach().cpu().clone() for k, v in mte_mlp.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)
        mte_mlp.load_state_dict(best_mte_state)
    model.eval(); mte_mlp.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pear, _ = evaluate(h_mean, hidden)
    d = hidden_true_substitution(model, visible, hidden, device)

    # Final learned MTE state for logging
    final_log_r_pred, _ = mte_mlp(taxa_ids, log_M)
    learned_B0 = mte_mlp.log_B0_taxa.detach().cpu().tolist()
    learned_targets = final_log_r_pred.detach().cpu().tolist()

    del model, mte_mlp
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return {
        "pearson": pear,
        "d_ratio": d["recon_true_scaled"] / d["recon_encoder"],
        "val_recon": best_val,
        "learned_B0_taxa": learned_B0,
        "learned_targets": learned_targets,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_stage1e")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full, species, days = load_beninca()

    print(f"\n=== Stage 1e: MTE+MLP (learnable taxa B_0, per-species delta) ===")
    print(f"Taxa (4 groups): {list(TAXA_MAP.values())[:4]}")
    print(f"Kremer B_0 init: bact={KREMER_B0_INIT[0]:.2f}, phyto={KREMER_B0_INIT[1]:.2f}, "
          f"pelagic={KREMER_B0_INIT[2]:.2f}, benthic={KREMER_B0_INIT[3]:.2f}")

    all_results = {}
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible_arr = np.delete(full, h_idx, axis=1)
        hidden_arr = full[:, h_idx]
        visible_species = [s for s in species if s != h_name]

        print(f"\n--- hidden={h_name} ---  N_visible={len(visible_species)}")
        rs = []
        for s in SEEDS:
            t0 = datetime.now()
            try:
                r = train_one(visible_arr, hidden_arr, s, visible_species, device)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  seed={s}  P={r['pearson']:+.3f}  d_r={r['d_ratio']:.2f}  "
                      f"val={r['val_recon']:.4f}  ({dt:.1f}s)  "
                      f"B0=[{r['learned_B0_taxa'][0]:+.2f}, {r['learned_B0_taxa'][1]:+.2f}, "
                      f"{r['learned_B0_taxa'][2]:+.2f}, {r['learned_B0_taxa'][3]:+.2f}]")
            except Exception as e:
                print(f"  seed={s}  FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "d_ratio": float("nan"),
                     "val_recon": float("nan"), "learned_B0_taxa": [],
                     "learned_targets": []}
            r["seed"] = s
            rs.append(r)
        all_results[h_name] = rs

    # Baselines
    phase2 = {"Cyclopoids": 0.064, "Calanoids": 0.159, "Rotifers": 0.056,
              "Nanophyto": 0.204, "Picophyto": 0.062, "Filam_diatoms": 0.056,
              "Ostracods": 0.125, "Harpacticoids": 0.129, "Bacteria": 0.171}
    stage1b = {"Cyclopoids": 0.055, "Calanoids": 0.173, "Rotifers": 0.080,
               "Nanophyto": 0.141, "Picophyto": 0.116, "Filam_diatoms": 0.031,
               "Ostracods": 0.272, "Harpacticoids": 0.120, "Bacteria": 0.197}
    stage1c = {"Cyclopoids": 0.053, "Calanoids": 0.079, "Rotifers": 0.087,
               "Nanophyto": 0.094, "Picophyto": 0.061, "Filam_diatoms": 0.021,
               "Ostracods": 0.169, "Harpacticoids": 0.156, "Bacteria": 0.052}

    print(f"\n{'='*110}")
    print("STAGE 1e RESULTS — MTE+MLP (learnable taxa-B0, per-species delta)")
    print('='*110)
    print(f"{'Species':<18}{'P2':<10}{'S1b':<10}{'S1c':<10}{'S1e':<12}{'Δ vs P2':<10}{'Δ vs S1b':<10}{'Δ vs S1c':<10}")
    sum_p2=0; sum_s1b=0; sum_s1c=0; sum_s1e=0; cnt=0
    for h in SPECIES_ORDER:
        rs = all_results[h]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        s1e = float(P.mean()) if len(P) else float("nan")
        print(f"{h:<18}{phase2[h]:<+10.3f}{stage1b[h]:<+10.3f}{stage1c[h]:<+10.3f}"
              f"{s1e:<+12.3f}{s1e-phase2[h]:<+10.3f}{s1e-stage1b[h]:<+10.3f}{s1e-stage1c[h]:<+10.3f}")
        if not np.isnan(s1e):
            sum_p2+=phase2[h]; sum_s1b+=stage1b[h]; sum_s1c+=stage1c[h]; sum_s1e+=s1e; cnt+=1

    avg_p2=sum_p2/cnt; avg_s1b=sum_s1b/cnt; avg_s1c=sum_s1c/cnt; avg_s1e=sum_s1e/cnt
    print(f"\nOverall:          P2={avg_p2:+.4f}  S1b={avg_s1b:+.4f}  S1c={avg_s1c:+.4f}  S1e={avg_s1e:+.4f}")
    print(f"Δ(S1e-S1b)={avg_s1e-avg_s1b:+.4f}")

    if avg_s1e > avg_s1b + 0.005:
        verdict = "✓ Stage 1e 超过 Stage 1b — MTE+MLP 可学 intercept 有效"
    elif avg_s1e > avg_s1b - 0.005:
        verdict = "≈ Stage 1e ≈ Stage 1b — 中性(无副作用, 可作学术 prior)"
    else:
        verdict = "✗ Stage 1e < Stage 1b — 仍失败, MTE prior 不适用"
    print(f"\n{verdict}\n{'='*110}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Beninca Stage 1e — MTE + MLP (learnable taxa B_0)\n\n")
        f.write(f"Seeds: {len(SEEDS)}, Epochs: {EPOCHS}\n\n")
        f.write("Taxa groups + Kremer init:\n")
        f.write("- 0 Bacteria   b=1.00  B0_init=+0.50\n")
        f.write("- 1 Phyto      b=0.95  B0_init=+0.00\n")
        f.write("- 2 Pelagic    b=0.88  B0_init=-0.20\n")
        f.write("- 3 Benthic    b=0.75  B0_init=-0.40\n\n")
        f.write("## Results\n\n")
        f.write("| Species | P2 | S1b | S1c | **S1e** | Δ vs S1b |\n|---|---|---|---|---|---|\n")
        for h in SPECIES_ORDER:
            rs = all_results[h]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            s1e = float(P.mean()) if len(P) else float("nan")
            f.write(f"| {h} | {phase2[h]:+.3f} | {stage1b[h]:+.3f} | {stage1c[h]:+.3f} | "
                    f"**{s1e:+.3f}** | {s1e-stage1b[h]:+.3f} |\n")
        f.write(f"\n**Overall**: P2={avg_p2:+.4f}, S1b={avg_s1b:+.4f}, S1c={avg_s1c:+.4f}, "
                f"**S1e={avg_s1e:+.4f}**, Δ(S1e−S1b)={avg_s1e-avg_s1b:+.4f}\n\n")
        f.write(f"**{verdict}**\n")

    with open(out_dir / "raw.json", "w") as f:
        json.dump(all_results, f, indent=2, default=float)
    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

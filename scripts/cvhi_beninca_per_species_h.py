"""Per-species h_i: each species gets its own hidden influence trajectory.

Key change: h goes from scalar (B, T) to per-species (B, T, N).
No G_field needed - h_i(t) directly added to base_i(t).

log(x_{t+1}/x_t)_i = f_visible_i(x_t) + h_i(t)

Combined with: NbedDyn ODE, VAE sampling, alt 5:1.
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
from scripts.cvhi_residual_L1L3_diagnostics import evaluate
from scripts.cvhi_beninca_nbeddyn import (
    find_burst_mask, burst_precision_recall,
)

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
PA, PB = 5, 1
LAM_H_ODE = 0.5


class PerSpeciesLatentDyn(nn.Module):
    """f_h for per-species h: shared MLP, applied independently to each species."""
    def __init__(self, n_visible, d_hidden=32):
        super().__init__()
        # Input: h_i(t-1) + context from all species
        self.ctx = nn.Sequential(
            nn.Linear(n_visible, d_hidden), nn.GELU(),
        )
        self.dyn = nn.Sequential(
            nn.Linear(1 + d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h_prev, x_prev):
        """h_prev: (B, T, N), x_prev: (B, T, N). Returns h_pred: (B, T, N)."""
        ctx = self.ctx(x_prev)  # (B, T, d_hidden)
        B, T, N = h_prev.shape
        # Apply shared dyn to each species
        ctx_exp = ctx.unsqueeze(2).expand(B, T, N, -1)       # (B, T, N, d_h)
        h_in = h_prev.unsqueeze(-1)                           # (B, T, N, 1)
        inp = torch.cat([h_in, ctx_exp], dim=-1)              # (B, T, N, 1+d_h)
        delta = self.dyn(inp).squeeze(-1)                     # (B, T, N)
        return h_prev + delta


def make_model_and_patch(N, device):
    """Create CVHI_Residual and patch encoder to output per-species h."""
    model = CVHI_Residual(
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
    ).to(device)

    # Patch: replace scalar readout with per-species readout
    d = HP["encoder_d"]
    enc = model.encoder.encoders[0]
    enc.readout_mu[-1] = nn.Linear(d, N).to(device)
    enc.readout_logsigma[-1] = nn.Linear(d, N).to(device)
    # Re-init
    nn.init.xavier_uniform_(enc.readout_mu[-1].weight)
    nn.init.zeros_(enc.readout_mu[-1].bias)
    nn.init.xavier_uniform_(enc.readout_logsigma[-1].weight)
    nn.init.constant_(enc.readout_logsigma[-1].bias, -2.0)

    return model


def custom_forward(model, visible, n_samples=2, rollout_K=0):
    """Custom forward with per-species h (S, B, T, N)."""
    if visible.dim() == 2:
        visible = visible.unsqueeze(0)
    B, T, N = visible.shape

    # Encoder -> per-species mu, log_sigma: (B, T, N, 1) via MultiChannel
    mu_k, log_sigma_k = model.encoder(visible)
    mu = mu_k[..., 0]            # (B, T, N) ← per-species!
    log_sigma = log_sigma_k[..., 0]  # (B, T, N)

    # Sample per-species h
    sigma = log_sigma.exp()
    eps = torch.randn(n_samples, B, T, N, device=visible.device)
    h_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, B, T, N)

    # f_visible (reuse existing GNN)
    base = model.compute_f_visible(visible)  # (B, T, N)

    # pred = base + h_i (no G needed!)
    pred_full = base.unsqueeze(0) + h_samples  # (S, B, T, N)

    # Null: h=0
    pred_null = base  # (B, T, N)

    # Shuffle: permute time axis
    perm = torch.randperm(T, device=visible.device)
    h_shuf = h_samples[:, :, perm, :]  # (S, B, T, N)
    pred_shuf = base.unsqueeze(0) + h_shuf

    # Actual log_ratio
    safe = torch.clamp(visible, min=1e-6)
    actual = torch.log(safe[:, 1:] / safe[:, :-1])
    actual = torch.clamp(actual, model.clamp_min, model.clamp_max)

    pred_full = pred_full[:, :, :-1, :]
    pred_null = pred_null[:, :-1, :]
    pred_shuf = pred_shuf[:, :, :-1, :]

    return {
        "mu": mu, "log_sigma": log_sigma,
        "h_samples": h_samples,
        "pred_full": pred_full, "pred_null": pred_null, "pred_shuf": pred_shuf,
        "actual": actual, "visible": visible,
        "base": base, "G": torch.ones_like(base),  # dummy G for compatibility
    }


def custom_loss(out, h_weight=1.0, lam_kl=0.017, lam_cf=9.5, min_energy=0.14,
                lam_smooth=0.02, lam_sparse=0.0, lam_rmse_log=0.1):
    """Loss for per-species h."""
    pred_full = out["pred_full"]
    pred_null = out["pred_null"]
    pred_shuf = out["pred_shuf"]
    actual = out["actual"]
    mu = out["mu"]
    log_sigma = out["log_sigma"]
    h_samples = out["h_samples"]

    # Recon losses
    recon_full = F.mse_loss(pred_full, actual.unsqueeze(0).expand_as(pred_full))
    recon_null = F.mse_loss(pred_null, actual)
    recon_shuf = F.mse_loss(pred_shuf, actual.unsqueeze(0).expand_as(pred_shuf))

    recon_loss = h_weight * recon_full + (1.0 - h_weight) * recon_null

    # Counterfactual
    margin_null_obs = recon_null - recon_full
    margin_shuf_obs = recon_shuf - recon_full
    loss_necessary = F.relu(0.002 - margin_null_obs)
    loss_shuffle = F.relu(0.001 - margin_shuf_obs)

    # KL per species (sum across species, mean across time)
    sigma_sq = torch.exp(2 * log_sigma)
    kl_per = 0.5 * (-2 * log_sigma + sigma_sq + mu ** 2 - 1)  # (B, T, N)
    kl = torch.clamp(kl_per - 0.02, min=0).mean()

    # Energy: variance of h across time, per species, then mean
    h_var = h_samples.var(dim=2).mean()  # var over time
    loss_energy = F.relu(min_energy - h_var)

    # Smoothness per species
    dh = h_samples[:, :, 2:, :] - 2 * h_samples[:, :, 1:-1, :] + h_samples[:, :, :-2, :]
    loss_smooth = (dh ** 2).mean()

    # RMSE log
    loss_rmse_log = torch.tensor(0.0, device=mu.device)
    if lam_rmse_log > 0 and "visible" in out:
        vis = out["visible"]
        safe = torch.clamp(vis, min=1e-6)
        log_safe = torch.log(safe)
        T_vis = log_safe.shape[1]
        T_pred = pred_full.shape[2]
        T_use = min(T_vis - 1, T_pred)
        log_actual = log_safe[:, 1:1+T_use]
        log_prev = log_safe[:, :T_use]
        pred_log_x = log_prev.unsqueeze(0) + pred_full[:, :, :T_use]
        loss_rmse_log = ((pred_log_x - log_actual.unsqueeze(0)) ** 2).mean()

    total = (recon_loss
             + lam_kl * kl
             + h_weight * lam_cf * loss_necessary
             + h_weight * lam_cf * 0.6 * loss_shuffle
             + 2.0 * loss_energy
             + lam_smooth * loss_smooth
             + lam_rmse_log * loss_rmse_log)

    return {"total": total, "recon_full": recon_full, "recon_null": recon_null}


def get_param_groups(model):
    enc_ids = set()
    for name, p in model.named_parameters():
        if 'encoder' in name or 'readout' in name:
            enc_ids.add(id(p))
    fvis = [p for p in model.parameters() if id(p) not in enc_ids]
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    return fvis, enc

def freeze(params):
    for p in params: p.requires_grad_(False)
def unfreeze(params):
    for p in params: p.requires_grad_(True)

def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden, seed, device, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model_and_patch(N, device)
    f_h = PerSpeciesLatentDyn(N, d_hidden=32).to(device)
    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters())
    opt = torch.optim.AdamW(all_params, lr=HP["lr"], weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_fh = None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup: h_w, K_r = 0.0, 0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            K_r = 0  # no rollout in custom forward for now

        if epoch >= warmup:
            cyc = (epoch - warmup) % (PA + PB)
            if cyc < PA:
                freeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(False)
                unfreeze(fvis_params)
            else:
                freeze(fvis_params); unfreeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(True)
        else:
            unfreeze(fvis_params); unfreeze(enc_params)
            for p in f_h.parameters(): p.requires_grad_(True)

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); f_h.train(); opt.zero_grad()
        out = custom_forward(model, x_train, n_samples=2)

        # Slice to train portion
        out_tr = {
            "pred_full": out["pred_full"][:, :, :train_end-1],
            "pred_null": out["pred_null"][:, :train_end-1],
            "pred_shuf": out["pred_shuf"][:, :, :train_end-1],
            "actual": out["actual"][:, :train_end-1],
            "mu": out["mu"][:, :train_end],
            "log_sigma": out["log_sigma"][:, :train_end],
            "h_samples": out["h_samples"][:, :, :train_end],
            "visible": out["visible"][:, :train_end],
        }

        losses = custom_loss(out_tr, h_weight=h_w)

        # NbedDyn ODE on per-species h
        loss_h_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and LAM_H_ODE > 0:
            h_mu = out["mu"][:, :train_end]       # (B, T, N)
            x_vis = out["visible"][:, :train_end]  # (B, T, N)
            h_pred = f_h(h_mu[:, :-1], x_vis[:, :-1])  # (B, T-1, N)
            loss_h_ode = F.mse_loss(h_pred, h_mu[:, 1:].detach())

        total = losses["total"] + LAM_H_ODE * h_w * loss_h_ode
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step(); sched.step()

        # Val
        with torch.no_grad():
            out_val = {
                "pred_full": out["pred_full"][:, :, train_end-1:],
                "pred_null": out["pred_null"][:, train_end-1:],
                "pred_shuf": out["pred_shuf"][:, :, train_end-1:],
                "actual": out["actual"][:, train_end-1:],
                "mu": out["mu"][:, train_end:],
                "log_sigma": out["log_sigma"][:, train_end:],
                "h_samples": out["h_samples"][:, :, train_end:],
                "visible": out["visible"][:, train_end:],
            }
            vl = custom_loss(out_val, h_weight=1.0)
            vr = vl["recon_full"].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    # Eval
    unfreeze(fvis_params); unfreeze(enc_params)
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = custom_forward(model, x_full, n_samples=30)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()  # (T, N)
        # Aggregate per-species h to scalar for Pearson with hidden species
        # Use mean across species (simple aggregation)
        h_scalar = h_mean.mean(axis=-1)  # (T,)

    pear, h_scaled = evaluate(h_scalar, hidden)
    burst_m = burst_precision_recall(h_scaled.flatten(), hidden, pct=10)

    # Also try: which single h_i has best correlation with hidden?
    best_single_pear = -1
    best_single_idx = -1
    for j in range(h_mean.shape[1]):
        p_j, _ = evaluate(h_mean[:, j], hidden)
        if p_j > best_single_pear:
            best_single_pear = p_j
            best_single_idx = j

    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"pearson": pear, "val_recon": best_val,
            "best_single_pear": best_single_pear,
            "best_single_idx": best_single_idx,
            **burst_m}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_per_species_h")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, _ = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]

    print(f"Per-species h_i + NbedDyn ODE + VAE + alt {PA}:{PB}")
    print(f"h_i(t) replaces h(t)*G_i(x). Each species has own hidden trajectory.\n")

    total_runs = len(SPECIES_ORDER) * len(SEEDS)
    results = {sp: [] for sp in SPECIES_ORDER}
    run_i = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden_raw = full[:, h_idx].astype(np.float32)
        print(f"--- hidden={h_name} ---")
        for seed in SEEDS:
            run_i += 1
            t0 = datetime.now()
            try:
                r = train_one(visible, hidden_raw, seed, device)
                dt = (datetime.now() - t0).total_seconds()
                vis_names = [s for s in species if s != h_name]
                best_ch = vis_names[r['best_single_idx']] if r['best_single_idx'] < len(vis_names) else '?'
                print(f"  [{run_i}/{total_runs}] seed={seed}  "
                      f"P={r['pearson']:+.3f}  "
                      f"best_h_i={r['best_single_pear']:+.3f}({best_ch})  "
                      f"burst_F={r['burst_f_score']:.3f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "val_recon": float("nan"),
                     "best_single_pear": float("nan"), "best_single_idx": -1,
                     "burst_precision": 0, "burst_recall": 0, "burst_f_score": 0}
            r["seed"] = seed
            results[h_name].append(r)

    # Summary
    print(f"\n{'='*110}")
    print("RESULTS: Per-species h_i + NbedDyn + Alt 5:1")
    print('='*110)
    print(f"{'Species':<16} {'P(mean_h)':>10} {'P(best_hi)':>12} {'Burst_F':>10}")
    print('-'*110)

    all_pm = []; all_pb = []; all_bf = []
    for sp in SPECIES_ORDER:
        rs = results[sp]
        PM = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        PB_ = np.array([r["best_single_pear"] for r in rs if not np.isnan(r["best_single_pear"])])
        BF = np.array([r["burst_f_score"] for r in rs])
        mpm = float(PM.mean()) if len(PM) else float("nan")
        mpb = float(PB_.mean()) if len(PB_) else float("nan")
        mbf = float(BF.mean()) if len(BF) else 0
        if not np.isnan(mpm): all_pm.append(mpm)
        if not np.isnan(mpb): all_pb.append(mpb)
        all_bf.append(mbf)
        print(f"{sp:<16} {mpm:>+10.3f} {mpb:>+12.3f} {mbf:>10.3f}")

    print('-'*110)
    opm = np.mean(all_pm) if all_pm else float("nan")
    opb = np.mean(all_pb) if all_pb else float("nan")
    obf = np.mean(all_bf)
    print(f"{'Overall':<16} {opm:>+10.4f} {opb:>+12.4f} {obf:>10.3f}")
    print(f"\nRef: NbedDyn(scalar h)=+0.1620, alt_5_1=+0.1595")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Per-species h_i + NbedDyn + Alt 5:1\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n\n")
        f.write("| Species | P(mean_h) | P(best_hi) | Burst_F |\n|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            rs = results[sp]
            PM = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            PB_ = np.array([r["best_single_pear"] for r in rs if not np.isnan(r["best_single_pear"])])
            BF = np.array([r["burst_f_score"] for r in rs])
            f.write(f"| {sp} | {PM.mean():+.3f} | {PB_.mean():+.3f} | {BF.mean():.3f} |\n")
        f.write(f"\n**Overall**: P(mean_h)={opm:+.4f}, P(best_hi)={opb:+.4f}, Burst_F={obf:.3f}\n")

    raw = {}
    for sp in SPECIES_ORDER:
        raw[sp] = [{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                     for k, v in r.items()} for r in results[sp]]
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

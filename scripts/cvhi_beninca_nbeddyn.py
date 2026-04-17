"""NbedDyn-inspired latent dynamics: h(t) follows learned ODE, not just encoder.

Key change (from Ouala et al. 2019 NbedDyn):
  h(t) = encoder output (current approach)
  h_pred(t) = h(t-1) + f_h(h(t-1), context(t-1))  # learned latent dynamics
  loss_h_ode = MSE(h_encoder, h_pred)               # ODE consistency

Combined with:
  - alt_5_1 (current best training)
  - Nutrients in recon loss (original, proven better)
  - Burst P/R/F evaluation
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
LAM_H_ODE = 0.5    # weight for NbedDyn ODE consistency loss


# --------------- NbedDyn: Latent dynamics network ---------------

class LatentDynamicsNet(nn.Module):
    """f_h: predict h(t) from h(t-1) and visible context.

    h_pred(t) = h(t-1) + f_h(h(t-1), summary(x(t-1)))
    Inspired by NbedDyn (Ouala et al. 2019) Eq.4-6.
    """
    def __init__(self, n_visible, d_hidden=32):
        super().__init__()
        # Context: compress visible state to a small vector
        self.context_net = nn.Sequential(
            nn.Linear(n_visible, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        # Dynamics: predict delta_h from (h, context)
        self.dyn_net = nn.Sequential(
            nn.Linear(1 + d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h_prev, x_prev):
        """h_prev: (B, T), x_prev: (B, T, N). Returns h_pred: (B, T)."""
        ctx = self.context_net(x_prev)           # (B, T, d_hidden)
        inp = torch.cat([h_prev.unsqueeze(-1), ctx], dim=-1)  # (B, T, 1+d_hidden)
        delta_h = self.dyn_net(inp).squeeze(-1)  # (B, T)
        return h_prev + delta_h


# --------------- Burst evaluation ---------------

def find_burst_mask(series, pct=10, eps=1e-6):
    safe = np.maximum(np.abs(series), eps)
    log_x = np.log(safe)
    dlog = np.abs(np.diff(log_x))
    if len(dlog) == 0:
        return np.zeros(len(series), dtype=bool)
    threshold = np.percentile(dlog, 100 - pct)
    return np.concatenate([dlog > threshold, [False]])


def burst_precision_recall(h_recovered, hidden_true, pct=10):
    burst_true = find_burst_mask(hidden_true, pct=pct)
    burst_pred = find_burst_mask(h_recovered, pct=pct)
    tp = int((burst_true & burst_pred).sum())
    fp = int((~burst_true & burst_pred).sum())
    fn = int((burst_true & ~burst_pred).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f_score = 2 * precision * recall / max(precision + recall, 1e-8)
    return {"burst_precision": precision, "burst_recall": recall,
            "burst_f_score": f_score}


# --------------- Model & training ---------------

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
    ).to(device)


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


def train_one(visible, hidden, seed, device, lam_h_ode=LAM_H_ODE, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model(N, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)

    fvis_params, enc_params = get_param_groups(model)
    # f_h trains with encoder (Phase B) since it models h's dynamics
    all_params = list(model.parameters()) + list(f_h.parameters())
    f_h_param_ids = set(id(p) for p in f_h.parameters())

    opt = torch.optim.AdamW(all_params, lr=HP["lr"], weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_fh_state = None

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

        # Alternating
        if epoch >= warmup:
            cycle_pos = (epoch - warmup) % (PA + PB)
            if cycle_pos < PA:
                # Phase A: train f_visible/G, freeze encoder + f_h
                freeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(False)
                unfreeze(fvis_params)
            else:
                # Phase B: freeze f_visible/G, train encoder + f_h
                freeze(fvis_params)
                unfreeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(True)
        else:
            unfreeze(fvis_params); unfreeze(enc_params)
            for p in f_h.parameters(): p.requires_grad_(True)

        # Input augmentation
        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); f_h.train()
        opt.zero_grad()

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

        # NbedDyn ODE consistency loss on h
        loss_h_ode = torch.tensor(0.0, device=device)
        if h_w > 0 and lam_h_ode > 0:
            h_mu = out["mu"][:, :train_end]       # (B, T_train)
            x_vis = out["visible"][:, :train_end]  # (B, T_train, N)
            # h_pred(t) = h(t-1) + f_h(h(t-1), x(t-1))
            h_pred = f_h(h_mu[:, :-1], x_vis[:, :-1])  # (B, T_train-1)
            h_target = h_mu[:, 1:]                       # (B, T_train-1)
            loss_h_ode = F.mse_loss(h_pred, h_target.detach())

        total = losses["total"] + lam_h_ode * h_w * loss_h_ode
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step(); sched.step()

        # Validation
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
            best_fh_state = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    # Eval
    unfreeze(fvis_params); unfreeze(enc_params)
    for p in f_h.parameters(): p.requires_grad_(True)
    if best_state is not None:
        model.load_state_dict(best_state)
    if best_fh_state is not None:
        f_h.load_state_dict(best_fh_state)
    model.eval(); f_h.eval()

    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

    pear, h_scaled = evaluate(h_mean, hidden)
    burst_m = burst_precision_recall(h_scaled.flatten(), hidden, pct=10)

    # Margin
    with torch.no_grad():
        vis = out_eval["visible"]
        safe_v = torch.clamp(vis, min=1e-6)
        log_ratio = torch.log(safe_v[:, 1:] / safe_v[:, :-1]).unsqueeze(0)
        recon_full_val = ((out_eval["pred_full"] - log_ratio) ** 2).mean().item()
        recon_null_val = ((out_eval["pred_null"] - log_ratio[0]) ** 2).mean().item()
    margin = recon_null_val - recon_full_val

    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {"pearson": pear, "val_recon": best_val,
            "margin": margin, **burst_m}


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_nbeddyn")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, _ = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]

    print(f"NbedDyn-inspired: alt_5_1 + latent ODE consistency (lam={LAM_H_ODE})")
    print(f"Nutrients in recon loss (original, proven better)")
    print(f"Burst eval: top 10% |d log x|\n")

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
                print(f"  [{run_i}/{total_runs}] seed={seed}  "
                      f"P={r['pearson']:+.3f}  "
                      f"burst_F={r['burst_f_score']:.3f}  "
                      f"margin={r['margin']:.4f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"pearson": float("nan"), "val_recon": float("nan"),
                     "margin": float("nan"),
                     "burst_precision": 0, "burst_recall": 0, "burst_f_score": 0}
            r["seed"] = seed
            results[h_name].append(r)

    # Summary
    print(f"\n{'='*100}")
    print(f"RESULTS: NbedDyn + Alt 5:1 (lam_h_ode={LAM_H_ODE})")
    print('='*100)
    print(f"{'Species':<16} {'Pearson':>10} {'Burst_F':>10} {'Margin':>10}")
    print('-'*100)

    all_pear = []; all_bf = []
    for sp in SPECIES_ORDER:
        rs = results[sp]
        P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
        BF = np.array([r["burst_f_score"] for r in rs])
        MG = np.array([r["margin"] for r in rs if not np.isnan(r.get("margin", float("nan")))])
        mp = float(P.mean()) if len(P) else float("nan")
        mbf = float(BF.mean()) if len(BF) else 0
        mmg = float(MG.mean()) if len(MG) else float("nan")
        if not np.isnan(mp): all_pear.append(mp)
        all_bf.append(mbf)
        print(f"{sp:<16} {mp:>+10.3f} {mbf:>10.3f} {mmg:>10.4f}")

    print('-'*100)
    op = np.mean(all_pear) if all_pear else float("nan")
    obf = np.mean(all_bf)
    print(f"{'Overall':<16} {op:>+10.4f} {obf:>10.3f}")
    print(f"\nRef: alt_5_1 = +0.1595, nutrient-input = +0.1446")

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write(f"# NbedDyn + Alt 5:1 (lam_h_ode={LAM_H_ODE})\n\n")
        f.write(f"Seeds: {SEEDS}, Epochs: {EPOCHS}, Alt: {PA}:{PB}\n\n")
        f.write("| Species | Pearson | Burst_F | Margin |\n|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            rs = results[sp]
            P = np.array([r["pearson"] for r in rs if not np.isnan(r["pearson"])])
            BF = np.array([r["burst_f_score"] for r in rs])
            MG = np.array([r["margin"] for r in rs if not np.isnan(r.get("margin", float("nan")))])
            f.write(f"| {sp} | {P.mean():+.3f} | {BF.mean():.3f} | {MG.mean():.4f} |\n")
        f.write(f"\n**Overall**: Pearson={op:+.4f}, Burst_F={obf:.3f}\n")
        f.write(f"\nRef: alt_5_1=+0.1595, nutrient-input=+0.1446\n")

    raw = {}
    for sp in SPECIES_ORDER:
        raw[sp] = [{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                     for k, v in r.items()} for r in results[sp]]
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

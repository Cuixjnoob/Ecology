"""Per-species h_i + learned aggregation + NbedDyn ODE + no VAE.

Architecture:
  encoder → h_i(t) per species (N-dim, point estimate)
  f_h: shared MLP, h_i(t) = h_i(t-1) + f_h(h_i(t-1), ctx)  [NbedDyn ODE]
  h_agg(t) = softmax(w) @ h(t)  [learned aggregation → scalar]

  pred_i = f_visible_i(x) + h_i(t) + h_agg(t) * v_i
           \_____________/   \____/   \____________/
           visible dynamics   local    global via aggregation

Evaluation: compare h_agg, PCA(h)[:,0], and oracle lstsq with hidden_true.
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
from sklearn.decomposition import PCA

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca
from scripts.cvhi_residual_L1L3_diagnostics import evaluate
from scripts.cvhi_beninca_nbeddyn import find_burst_mask, burst_precision_recall
from scripts.cvhi_beninca_per_species_h import (
    make_model_and_patch, custom_forward, get_param_groups,
    freeze, unfreeze, alpha_schedule, PerSpeciesLatentDyn,
)

SEEDS = [42, 123, 456]
EPOCHS = 500
HP = dict(
    encoder_d=96, encoder_blocks=3, encoder_dropout=0.1,
    takens_lags=(1, 2, 4, 8),
    lr=0.0006033475528697158,
    lam_cf=9.517725868477207,
    min_energy=0.14353013693386804,
)
SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]
PA, PB = 5, 1
LAM_H_ODE = 0.5


def custom_loss_with_agg(out, agg_w, agg_v, h_weight=1.0,
                          lam_cf=9.5, min_energy=0.14, lam_smooth=0.02,
                          lam_rmse_log=0.1):
    """Loss with per-species h_i + learned aggregation."""
    pred_full = out["pred_full"]     # (S, B, T-1, N)
    pred_null = out["pred_null"]     # (B, T-1, N)
    actual = out["actual"]           # (B, T-1, N)
    h_samples = out["h_samples"]     # (S, B, T, N)
    mu = out["mu"]                   # (B, T, N)

    # Compute h_agg and add global contribution
    w = F.softmax(agg_w, dim=0)                    # (N,)
    h_agg = (mu * w.unsqueeze(0).unsqueeze(0)).sum(-1)  # (B, T)
    # Add global effect: pred += h_agg * v_i, align time dim with pred_full
    T_pred = pred_full.shape[2]
    global_effect = h_agg[:, :T_pred].unsqueeze(-1) * agg_v.unsqueeze(0).unsqueeze(0)  # (B, T_pred, N)
    pred_with_global = pred_full + global_effect.unsqueeze(0)  # (S, B, T_pred, N)
    pred_null_with_global = pred_null  # null: no h at all

    # Recon
    recon_full = F.mse_loss(pred_with_global, actual.unsqueeze(0).expand_as(pred_with_global))
    recon_null = F.mse_loss(pred_null_with_global, actual)

    # Shuffle h in time
    T_mu = mu.shape[1]
    T_h = h_samples.shape[2]
    perm_mu = torch.randperm(T_mu, device=mu.device)
    perm_h = torch.randperm(T_h, device=mu.device)
    mu_shuf = mu[:, perm_mu]
    h_agg_shuf = (mu_shuf * w.unsqueeze(0).unsqueeze(0)).sum(-1)
    h_shuf = h_samples[:, :, perm_h, :]
    T_use_shuf = min(T_pred, h_shuf.shape[2] - 1)
    base_exp = out["base"].unsqueeze(0)[:, :, :T_use_shuf]
    pred_shuf_base = base_exp + h_shuf[:, :, :T_use_shuf]
    global_shuf = h_agg_shuf[:, :T_use_shuf].unsqueeze(-1) * agg_v.unsqueeze(0).unsqueeze(0)
    pred_shuf = pred_shuf_base + global_shuf.unsqueeze(0)
    # Align actual for shuf loss
    actual_shuf = actual[:, :T_use_shuf]
    recon_shuf = F.mse_loss(pred_shuf, actual_shuf.unsqueeze(0).expand_as(pred_shuf))

    recon_loss = h_weight * recon_full + (1 - h_weight) * recon_null

    # Counterfactual
    loss_necessary = F.relu(0.002 - (recon_null - recon_full))
    loss_shuffle = F.relu(0.001 - (recon_shuf - recon_full))

    # Energy per species
    h_var = h_samples.var(dim=2).mean()
    loss_energy = F.relu(min_energy - h_var)

    # Smoothness per species
    dh = h_samples[:, :, 2:, :] - 2 * h_samples[:, :, 1:-1, :] + h_samples[:, :, :-2, :]
    loss_smooth = (dh ** 2).mean()

    # RMSE log (with correct slicing)
    loss_rmse_log = torch.tensor(0.0, device=mu.device)
    if lam_rmse_log > 0 and "visible" in out:
        vis = out["visible"]
        safe = torch.clamp(vis, min=1e-6)
        log_safe = torch.log(safe)
        T_pred = pred_with_global.shape[2]
        T_vis = log_safe.shape[1]
        T_use = min(T_vis - 1, T_pred)
        log_actual = log_safe[:, 1:1+T_use]
        log_prev = log_safe[:, :T_use]
        plx = log_prev.unsqueeze(0) + pred_with_global[:, :, :T_use]
        loss_rmse_log = ((plx - log_actual.unsqueeze(0)) ** 2).mean()

    total = (recon_loss
             + h_weight * lam_cf * loss_necessary
             + h_weight * lam_cf * 0.6 * loss_shuffle
             + 2.0 * loss_energy
             + lam_smooth * loss_smooth
             + lam_rmse_log * loss_rmse_log)

    return {"total": total, "recon_full": recon_full, "recon_null": recon_null,
            "h_agg": h_agg}


def train_one(visible, hidden, seed, device, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    model = make_model_and_patch(N, device)
    model.point_estimate = True  # no VAE sampling
    f_h = PerSpeciesLatentDyn(N, d_hidden=32).to(device)

    # Learned aggregation: softmax(w) for weighting, v for global effect
    agg_w = nn.Parameter(torch.zeros(N, device=device))     # aggregation weights
    agg_v = nn.Parameter(torch.zeros(N, device=device))     # per-species global sensitivity

    fvis_params, enc_params = get_param_groups(model)
    all_params = list(model.parameters()) + list(f_h.parameters()) + [agg_w, agg_v]
    opt = torch.optim.AdamW(all_params, lr=HP["lr"], weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_fh = None
    best_agg_w = None; best_agg_v = None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup: h_w = 0.0
        else: h_w = min(1.0, (epoch - warmup) / ramp)

        # Alternating
        if epoch >= warmup:
            cyc = (epoch - warmup) % (PA + PB)
            if cyc < PA:
                freeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(False)
                agg_w.requires_grad_(False); agg_v.requires_grad_(False)
                unfreeze(fvis_params)
            else:
                freeze(fvis_params); unfreeze(enc_params)
                for p in f_h.parameters(): p.requires_grad_(True)
                agg_w.requires_grad_(True); agg_v.requires_grad_(True)
        else:
            unfreeze(fvis_params); unfreeze(enc_params)
            for p in f_h.parameters(): p.requires_grad_(True)
            agg_w.requires_grad_(True); agg_v.requires_grad_(True)

        if epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > 0.05).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train(); f_h.train(); opt.zero_grad()
        out = custom_forward(model, x_train, n_samples=2)

        # Slice train
        out_tr = {
            "pred_full": out["pred_full"][:, :, :train_end-1],
            "pred_null": out["pred_null"][:, :train_end-1],
            "actual": out["actual"][:, :train_end-1],
            "mu": out["mu"][:, :train_end],
            "log_sigma": out["log_sigma"][:, :train_end],
            "h_samples": out["h_samples"][:, :, :train_end],
            "visible": out["visible"][:, :train_end],
            "base": out["base"][:, :train_end],
        }

        losses = custom_loss_with_agg(out_tr, agg_w, agg_v, h_weight=h_w)

        # NbedDyn ODE on per-species h
        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0:
            hm = out["mu"][:, :train_end]
            xv = out["visible"][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = F.mse_loss(hp, hm[:, 1:].detach())

        total = losses["total"] + LAM_H_ODE * h_w * loss_ode
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step(); sched.step()

        with torch.no_grad():
            out_val = {
                "pred_full": out["pred_full"][:, :, train_end-1:],
                "pred_null": out["pred_null"][:, train_end-1:],
                "actual": out["actual"][:, train_end-1:],
                "mu": out["mu"][:, train_end:],
                "log_sigma": out["log_sigma"][:, train_end:],
                "h_samples": out["h_samples"][:, :, train_end:],
                "visible": out["visible"][:, train_end:],
                "base": out["base"][:, train_end:],
            }
            vl = custom_loss_with_agg(out_val, agg_w, agg_v, h_weight=1.0)
            vr = vl["recon_full"].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}
            best_agg_w = agg_w.detach().cpu().clone()
            best_agg_v = agg_v.detach().cpu().clone()

    # Eval
    unfreeze(fvis_params); unfreeze(enc_params)
    if best_state: model.load_state_dict(best_state)
    if best_fh: f_h.load_state_dict(best_fh)
    if best_agg_w is not None:
        agg_w.data = best_agg_w.to(device)
        agg_v.data = best_agg_v.to(device)
    model.eval()

    with torch.no_grad():
        out_eval = custom_forward(model, x_full, n_samples=1)
        h_all = out_eval["mu"][0].cpu().numpy()  # (T, N)

    L = min(len(h_all), len(hidden))
    h_all = h_all[:L]
    hidden = hidden[:L]

    # Method 1: learned aggregation weights
    w_np = F.softmax(agg_w, dim=0).detach().cpu().numpy()
    h_learned = (h_all * w_np).sum(axis=1)
    p_learned, _ = evaluate(h_learned, hidden)

    # Method 2: PCA first component (unsupervised)
    if h_all.std() > 1e-8:
        pca = PCA(n_components=1)
        h_pca = pca.fit_transform(h_all).flatten()
        p_pca, _ = evaluate(h_pca, hidden)
    else:
        p_pca = 0.0

    # Method 3: oracle lstsq (uses hidden_true)
    X = np.concatenate([h_all, np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden, rcond=None)
    h_oracle = X @ coef
    p_oracle = float(np.corrcoef(h_oracle, hidden)[0, 1])

    # Method 4: best single h_i
    p_best_single = -1
    for j in range(h_all.shape[1]):
        pj, _ = evaluate(h_all[:, j], hidden)
        if pj > p_best_single: p_best_single = pj

    # Burst on learned agg
    _, h_sc = evaluate(h_learned, hidden)
    burst_m = burst_precision_recall(h_sc.flatten(), hidden, pct=10)

    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return {
        "p_learned": p_learned, "p_pca": p_pca,
        "p_oracle": p_oracle, "p_best_single": p_best_single,
        "val_recon": best_val, **burst_m,
    }


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_psh_agg")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, _ = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]

    print(f"Per-species h + learned agg + NbedDyn ODE + no VAE + alt {PA}:{PB}\n")

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
                      f"learned={r['p_learned']:+.3f}  "
                      f"pca={r['p_pca']:+.3f}  "
                      f"oracle={r['p_oracle']:+.3f}  "
                      f"best_i={r['p_best_single']:+.3f}  "
                      f"burst_F={r['burst_f_score']:.3f}  ({dt:.1f}s)")
            except Exception as e:
                print(f"  [{run_i}/{total_runs}] FAILED: {e}")
                import traceback; traceback.print_exc()
                r = {"p_learned": float("nan"), "p_pca": float("nan"),
                     "p_oracle": float("nan"), "p_best_single": float("nan"),
                     "val_recon": float("nan"),
                     "burst_precision": 0, "burst_recall": 0, "burst_f_score": 0}
            r["seed"] = seed
            results[h_name].append(r)

    # Summary
    print(f"\n{'='*120}")
    print("Per-species h + Learned Aggregation + NbedDyn ODE")
    print('='*120)
    print(f"{'Species':<16} {'Learned':>10} {'PCA':>10} {'Oracle':>10} {'Best_i':>10} {'Burst_F':>10}")
    print('-'*120)

    all_l = []; all_p = []; all_o = []; all_b = []; all_bf = []
    for sp in SPECIES_ORDER:
        rs = results[sp]
        PL = np.array([r["p_learned"] for r in rs if not np.isnan(r["p_learned"])])
        PP = np.array([r["p_pca"] for r in rs if not np.isnan(r["p_pca"])])
        PO = np.array([r["p_oracle"] for r in rs if not np.isnan(r["p_oracle"])])
        PB_ = np.array([r["p_best_single"] for r in rs if not np.isnan(r["p_best_single"])])
        BF = np.array([r["burst_f_score"] for r in rs])
        ml = float(PL.mean()) if len(PL) else float("nan")
        mp = float(PP.mean()) if len(PP) else float("nan")
        mo = float(PO.mean()) if len(PO) else float("nan")
        mb = float(PB_.mean()) if len(PB_) else float("nan")
        mbf = float(BF.mean()) if len(BF) else 0
        if not np.isnan(ml): all_l.append(ml)
        if not np.isnan(mp): all_p.append(mp)
        if not np.isnan(mo): all_o.append(mo)
        if not np.isnan(mb): all_b.append(mb)
        all_bf.append(mbf)
        print(f"{sp:<16} {ml:>+10.3f} {mp:>+10.3f} {mo:>+10.3f} {mb:>+10.3f} {mbf:>10.3f}")

    print('-'*120)
    ol = np.mean(all_l) if all_l else float("nan")
    op = np.mean(all_p) if all_p else float("nan")
    oo = np.mean(all_o) if all_o else float("nan")
    ob = np.mean(all_b) if all_b else float("nan")
    obf = np.mean(all_bf)
    print(f"{'Overall':<16} {ol:>+10.4f} {op:>+10.4f} {oo:>+10.4f} {ob:>+10.4f} {obf:>10.3f}")
    print(f"\nRef: scalar_h(NbedDyn)=+0.162, per_species_h(mean)=+0.070, per_species_h(best_i)=+0.178")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Per-species h + Learned Aggregation\n\n")
        f.write(f"No VAE, NbedDyn ODE, alt {PA}:{PB}\n\n")
        f.write("| Species | Learned | PCA | Oracle | Best_i | Burst_F |\n|---|---|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            rs = results[sp]
            PL = np.array([r["p_learned"] for r in rs if not np.isnan(r["p_learned"])])
            PP = np.array([r["p_pca"] for r in rs if not np.isnan(r["p_pca"])])
            PO = np.array([r["p_oracle"] for r in rs if not np.isnan(r["p_oracle"])])
            PB_ = np.array([r["p_best_single"] for r in rs if not np.isnan(r["p_best_single"])])
            BF = np.array([r["burst_f_score"] for r in rs])
            f.write(f"| {sp} | {PL.mean():+.3f} | {PP.mean():+.3f} | {PO.mean():+.3f} | {PB_.mean():+.3f} | {BF.mean():.3f} |\n")
        f.write(f"\n**Overall**: Learned={ol:+.4f}, PCA={op:+.4f}, Oracle={oo:+.4f}, Best_i={ob:+.4f}\n")

    raw = {}
    for sp in SPECIES_ORDER:
        raw[sp] = [{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                     for k, v in r.items()} for r in results[sp]]
    with open(out_dir / "results.json", "w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2, ensure_ascii=False)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

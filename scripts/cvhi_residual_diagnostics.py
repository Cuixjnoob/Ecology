"""CVHI_Residual 诊断实验包 (Exp A-D).

判定方向 1 (val-selection + ensemble) 是否够用, 还是必须上方向 2 (H-step).

红线: hidden_true 绝对不进训练. 仅在:
  - synthetic eval 时做 Pearson 计算 (合规)
  - Exp D 诊断 dynamics 潜力 (合规: 不反传回模型)
"""
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import spearmanr

from models.cvhi_residual import CVHI_Residual

TOP12 = ["PP", "DM", "PB", "DO", "OT", "RM", "PE", "DS", "PF", "NA", "OL", "PM"]


def _configure_matplotlib():
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC", "Microsoft YaHei", "SimHei", "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.size"] = 11


def load_portal(hidden_species="OT"):
    counts = defaultdict(lambda: defaultdict(int))
    with open("data/real_datasets/portal_rodent.csv") as f:
        r = csv.DictReader(f)
        for row in r:
            try:
                year = int(row['year']); month = int(row['month'])
            except (ValueError, KeyError):
                continue
            sp = row['species']
            if sp in TOP12:
                counts[(year, month)][sp] += 1
    all_months = sorted(counts.keys())
    matrix = np.zeros((len(all_months), len(TOP12)), dtype=np.float32)
    for t, (y, m) in enumerate(all_months):
        for j, sp in enumerate(TOP12):
            matrix[t, j] = counts[(y, m)].get(sp, 0)
    from scipy.ndimage import uniform_filter1d
    def smooth(x, w=3):
        pad = np.pad(x, ((w//2, w//2), (0, 0)), mode="edge")
        return uniform_filter1d(pad, size=w, axis=0)[w//2:w//2+x.shape[0]]
    matrix_s = smooth(matrix, w=3)
    valid = matrix_s.sum(axis=1) > 10
    matrix_s = matrix_s[valid]
    h_idx = TOP12.index(hidden_species)
    keep = [i for i in range(len(TOP12)) if i != h_idx]
    return matrix_s[:, keep] + 0.5, matrix_s[:, h_idx] + 0.5


def load_lv():
    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    return (d["states_B_5species"].astype(np.float32) + 0.01,
            d["hidden_B"].astype(np.float32) + 0.01)


def evaluate(h_pred, hidden_true):
    L = min(len(h_pred), len(hidden_true))
    h_pred = h_pred[:L]; hidden_true = hidden_true[:L]
    X = np.concatenate([h_pred.reshape(-1, 1), np.ones((L, 1))], axis=1)
    coef, _, _, _ = np.linalg.lstsq(X, hidden_true, rcond=None)
    h_scaled = X @ coef
    return float(np.corrcoef(h_scaled, hidden_true)[0, 1]), h_scaled


def pairwise_corr(h_list):
    """Return K×K Pearson matrix among list of (T,) arrays."""
    K = len(h_list)
    M = np.ones((K, K))
    for i in range(K):
        for j in range(K):
            if i == j: continue
            L = min(len(h_list[i]), len(h_list[j]))
            M[i, j] = float(np.corrcoef(h_list[i][:L], h_list[j][:L])[0, 1])
    return M


def make_model(N, is_portal):
    if is_portal:
        return CVHI_Residual(
            num_visible=N,
            encoder_d=48, encoder_blocks=2, encoder_heads=4,
            takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
            d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
            d_species_G=12, G_field_layers=1, G_field_top_k=3,
            prior_std=1.0,
        )
    return CVHI_Residual(
        num_visible=N,
        encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=24, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=16, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0,
    )


def train_one(visible, hidden_for_eval, device, seed, epochs=300,
               warmup_frac=0.2, is_portal=False):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, is_portal).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup_epochs = int(warmup_frac * epochs)
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None; best_epoch = -1

    margin_null = 0.002 if is_portal else 0.003
    margin_shuf = 0.001 if is_portal else 0.002
    min_energy = 0.05 if is_portal else 0.02

    for epoch in range(epochs):
        if epoch < warmup_epochs:
            h_weight = 0.0
        else:
            h_weight = min(1.0, (epoch - warmup_epochs) / max(1, int(0.2 * epochs)))

        model.train(); opt.zero_grad()
        out = model(x, n_samples=2)
        train_out = model.slice_out(out, 0, train_end)
        losses = model.loss(
            train_out, beta_kl=0.03, free_bits=0.02,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_energy,
            lam_smooth=0.05, lam_sparse=0.02, h_weight=h_weight,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                                     margin_null=margin_null, margin_shuf=margin_shuf,
                                     lam_energy=2.0, min_energy=min_energy)
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup_epochs + 20 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x, n_samples=30)
        h_mean = out_eval["H_samples"].mean(dim=0)[0].cpu().numpy() if False else \
                 out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
        full_losses = model.loss(out_eval, h_weight=1.0,
                                  margin_null=margin_null, margin_shuf=margin_shuf,
                                  lam_energy=2.0, min_energy=min_energy)
        val_out = model.slice_out(out_eval, train_end, T)
        val_losses_final = model.loss(val_out, h_weight=1.0,
                                       margin_null=margin_null, margin_shuf=margin_shuf,
                                       lam_energy=2.0, min_energy=min_energy)

    pear, h_scaled = evaluate(h_mean, hidden_for_eval)
    return {
        "seed": seed,
        "model": model,
        "pearson": pear,
        "train_recon": float(full_losses["recon_full"]),
        "val_recon": float(val_losses_final["recon_full"]),
        "m_null": float(full_losses["margin_null_obs"]),
        "m_shuf": float(full_losses["margin_shuf_obs"]),
        "h_var": float(full_losses["h_var"]),
        "kl": float(full_losses["kl"]),
        "sigma": float(full_losses["sigma_mean"]),
        "h_mean": h_mean,
        "h_scaled": h_scaled,
        "best_epoch": best_epoch,
    }


# ============================================================================
# Exp C: H-step prototype (frozen dynamics, optimize h_free)
# ============================================================================
def h_step_prototype(model, visible, n_inner_steps=100, lr=0.05,
                      encoder_init_from_another=None, device="cpu"):
    """Given frozen dynamics, run H-step from 4 different h_init.

    Returns dict with h_final per init + pairwise similarity.
    No hidden supervision. Only recon + priors.
    """
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    B, T, N = x.shape

    # Freeze all model parameters
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    with torch.no_grad():
        base = model.compute_f_visible(x)  # (B, T, N)
        G = model.compute_G(x)              # (B, T, N)
        # Encoder's h for this seed
        mu_k, _ = model.encoder(x)
        h_self_encoder = mu_k[..., 0].detach().clone()  # (B, T)

    safe = torch.clamp(x, min=1e-6)
    actual = torch.log(safe[:, 1:] / safe[:, :-1])
    actual = torch.clamp(actual, -2.5, 2.5)

    inits = {
        "encoder_self": h_self_encoder,
        "zero": torch.zeros_like(h_self_encoder),
        "random_gauss": torch.randn_like(h_self_encoder) * 0.3,
    }
    if encoder_init_from_another is not None:
        inits["encoder_other"] = torch.tensor(encoder_init_from_another, dtype=torch.float32, device=device).unsqueeze(0)

    results = {}
    for init_name, h_init in inits.items():
        h_free = torch.nn.Parameter(h_init.clone())
        inner_opt = torch.optim.Adam([h_free], lr=lr)
        traj = []
        for step in range(n_inner_steps):
            pred = base + h_free.unsqueeze(-1) * G  # (B, T, N)
            pred_trim = pred[:, :-1, :]
            recon = F.mse_loss(pred_trim, actual)
            # Priors (no hidden supervision)
            dh = h_free[:, 2:] - 2 * h_free[:, 1:-1] + h_free[:, :-2]
            smooth = (dh ** 2).mean()
            # Counterfactual null (encourage h use)
            pred_null_trim = base[:, :-1, :]
            recon_null = F.mse_loss(pred_null_trim, actual)
            necessary = F.relu(0.003 - (recon_null - recon))
            # Energy lower bound
            h_var = h_free.var(dim=-1).mean()
            energy_pen = F.relu(0.02 - h_var)
            loss = recon + 0.05 * smooth + 5.0 * necessary + 2.0 * energy_pen
            inner_opt.zero_grad()
            loss.backward()
            inner_opt.step()
            if step % 20 == 0 or step == n_inner_steps - 1:
                traj.append({"step": step, "loss": loss.item(), "recon": recon.item()})
        results[init_name] = {
            "h_final": h_free.detach()[0].cpu().numpy(),
            "h_init": h_init[0].cpu().numpy(),
            "trajectory": traj,
        }

    # Pairwise similarity of h_final
    names = list(results.keys())
    h_list = [results[n]["h_final"] for n in names]
    sim_matrix = pairwise_corr(h_list)
    return {
        "per_init": results, "init_names": names,
        "pairwise_sim": sim_matrix,
    }


# ============================================================================
# Exp D: Substitute hidden_true (diagnostic, no backprop to model)
# ============================================================================
def hidden_true_substitution(model, visible, hidden_true, device="cpu"):
    """Compute recon when using hidden_true as h. Diagnostic only."""
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    h_true = torch.tensor(hidden_true, dtype=torch.float32, device=device).unsqueeze(0)
    # Normalize h_true to zero mean (so it's a "residual-style" signal)
    h_true_centered = (h_true - h_true.mean())
    # Scale to match encoder's typical σ (diagnostic)
    with torch.no_grad():
        mu_k, log_sigma_k = model.encoder(x)
        encoder_h = mu_k[..., 0]
        encoder_h_centered = encoder_h - encoder_h.mean()
        encoder_std = encoder_h_centered.std()
        h_true_scaled = h_true_centered * (encoder_std / (h_true_centered.std() + 1e-6))

    safe = torch.clamp(x, min=1e-6)
    actual = torch.log(safe[:, 1:] / safe[:, :-1])
    actual = torch.clamp(actual, -2.5, 2.5)

    with torch.no_grad():
        base = model.compute_f_visible(x)
        G = model.compute_G(x)
        pred_encoder = base + encoder_h.unsqueeze(-1) * G
        pred_true = base + h_true_scaled.unsqueeze(-1) * G
        pred_null = base
        # also try: centered encoder
        pred_enc_ctr = base + encoder_h_centered.unsqueeze(-1) * G

    recon_encoder = F.mse_loss(pred_encoder[:, :-1], actual).item()
    recon_true = F.mse_loss(pred_true[:, :-1], actual).item()
    recon_null = F.mse_loss(pred_null[:, :-1], actual).item()
    recon_enc_ctr = F.mse_loss(pred_enc_ctr[:, :-1], actual).item()
    return {
        "recon_encoder": recon_encoder,
        "recon_true_scaled": recon_true,
        "recon_null": recon_null,
        "recon_encoder_centered": recon_enc_ctr,
    }


# ============================================================================
# Main diagnostic driver
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_seeds", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--datasets", nargs="+", default=["portal", "lv"])
    args = parser.parse_args()

    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_cvhi_residual_diagnostics")
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out_dir}\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    seeds = [42, 123, 456, 789, 2024, 31415, 27182, 65537][:args.n_seeds]
    print(f"Seeds: {seeds}\n")

    datasets = {}
    if "portal" in args.datasets:
        v, h = load_portal("OT")
        datasets["portal"] = {"visible": v, "hidden": h, "is_portal": True,
                               "name": "Portal OT"}
    if "lv" in args.datasets:
        v, h = load_lv()
        datasets["lv"] = {"visible": v, "hidden": h, "is_portal": False,
                           "name": "Synthetic LV"}

    all_results = {}
    for ds_key, ds in datasets.items():
        print(f"\n{'='*72}\nExp A: multi-seed on {ds['name']}\n{'='*72}")
        results = []
        for seed in seeds:
            r = train_one(ds["visible"], ds["hidden"], device, seed,
                           epochs=args.epochs, is_portal=ds["is_portal"])
            results.append(r)
            print(f"  seed {seed:5d}: P={r['pearson']:+.4f}  val_recon={r['val_recon']:.4f}  "
                  f"m_null={r['m_null']:+.4f}  m_shuf={r['m_shuf']:+.4f}  "
                  f"h_var={r['h_var']:.3f}")
        all_results[ds_key] = {"data": ds, "results": results}

    # ========================================================================
    # Analysis
    # ========================================================================
    analysis = {}
    for ds_key, rr in all_results.items():
        results = rr["results"]
        ds = rr["data"]
        print(f"\n{'='*72}\nAnalysis: {ds['name']}\n{'='*72}")

        # Exp A: Correlation of unsupervised metrics with Pearson
        pearsons = np.array([r["pearson"] for r in results])
        val_recons = np.array([r["val_recon"] for r in results])
        train_recons = np.array([r["train_recon"] for r in results])
        m_nulls = np.array([r["m_null"] for r in results])
        m_shufs = np.array([r["m_shuf"] for r in results])
        h_vars = np.array([r["h_var"] for r in results])

        rho_val, p_val = spearmanr(val_recons, pearsons)
        rho_train, p_train = spearmanr(train_recons, pearsons)
        rho_mnull, _ = spearmanr(m_nulls, pearsons)
        rho_mshuf, _ = spearmanr(m_shufs, pearsons)
        rho_hvar, _ = spearmanr(h_vars, pearsons)

        print(f"  Spearman correlations (vs |Pearson|):")
        print(f"    val_recon   ρ = {rho_val:+.3f}  (p={p_val:.3f})")
        print(f"    train_recon ρ = {rho_train:+.3f}  (p={p_train:.3f})")
        print(f"    m_null      ρ = {rho_mnull:+.3f}")
        print(f"    m_shuf      ρ = {rho_mshuf:+.3f}")
        print(f"    h_var       ρ = {rho_hvar:+.3f}")
        # Decision for val_recon
        print(f"\n  Exp A verdict:")
        if rho_val < -0.5:
            print(f"    [OK] val_recon is RELIABLE selector (ρ={rho_val:.2f} < -0.5)")
        elif rho_val < -0.2:
            print(f"    [~] val_recon is WEAK selector (ρ={rho_val:.2f})")
        else:
            print(f"    [NO] val_recon FAILS as selector (ρ={rho_val:.2f})")

        # Exp B: Top-K ensemble + cross-seed similarity
        sort_by_val = sorted(range(len(results)), key=lambda i: results[i]["val_recon"])
        top_k_idx = sort_by_val[:args.top_k]
        bottom_k_idx = sort_by_val[-args.top_k:]
        top_k_pearsons = [results[i]["pearson"] for i in top_k_idx]
        bot_k_pearsons = [results[i]["pearson"] for i in bottom_k_idx]
        top_k_h = [results[i]["h_mean"] for i in top_k_idx]
        bot_k_h = [results[i]["h_mean"] for i in bottom_k_idx]
        # Pairwise within top-K
        top_sim = pairwise_corr(top_k_h)
        bot_sim = pairwise_corr(bot_k_h)
        C_in = np.mean(top_sim[~np.eye(args.top_k, dtype=bool)])
        C_out = np.mean(bot_sim[~np.eye(args.top_k, dtype=bool)])
        # Ensemble
        h_ensemble = np.mean(top_k_h, axis=0)
        ens_pear, _ = evaluate(h_ensemble, ds["hidden"])

        print(f"\n  Exp B: Top-{args.top_k} by val_recon")
        print(f"    top-K seeds (sorted by val_recon): {[results[i]['seed'] for i in top_k_idx]}")
        print(f"    top-K pearsons:   {[f'{p:+.3f}' for p in top_k_pearsons]}")
        print(f"    bottom-K pearsons: {[f'{p:+.3f}' for p in bot_k_pearsons]}")
        print(f"    top-K pairwise h-sim (C_in):  {C_in:+.3f}")
        print(f"    bot-K pairwise h-sim (C_out): {C_out:+.3f}")
        print(f"    top-K ensemble Pearson:       {ens_pear:+.4f}")
        print(f"    top-K mean Pearson:           {np.mean(top_k_pearsons):+.4f}")

        print(f"\n  Exp B verdict:")
        if C_in > 0.7 and C_in > C_out + 0.2:
            print(f"    [OK] top-K are SIMILAR solutions → ensemble stabilizes (direction 1 OK)")
        elif C_in < 0.3:
            print(f"    [NO] top-K are DIFFERENT solutions → ensemble averages noise (direction 1 insufficient)")
        else:
            print(f"    [~] MIDDLE zone: C_in={C_in:.2f}, partial clustering")

        # Exp C: H-step prototype (only on 1 model, most-val seed)
        print(f"\n  Exp C: H-step prototype (frozen dynamics, 100 inner steps)")
        seed_for_C_idx = top_k_idx[0]  # best val
        model_for_C = results[seed_for_C_idx]["model"]
        # Use another seed's encoder output as one init
        other_seed_idx = (seed_for_C_idx + 1) % len(results)
        other_seed_encoder_h = results[other_seed_idx]["h_mean"]

        c_result = h_step_prototype(
            model_for_C, ds["visible"],
            n_inner_steps=100, lr=0.05,
            encoder_init_from_another=other_seed_encoder_h,
            device=device,
        )
        # Pearson of each h_final vs truth (diagnostic only)
        c_pearsons = {}
        for name in c_result["init_names"]:
            h_final = c_result["per_init"][name]["h_final"]
            p, _ = evaluate(h_final, ds["hidden"])
            c_pearsons[name] = p
            print(f"    init={name:<20s}  h_final Pearson={p:+.4f}")
        print(f"    pairwise h_final similarity matrix:")
        for i, name_i in enumerate(c_result["init_names"]):
            print(f"      {name_i:<20s}: " + " ".join(
                [f"{c_result['pairwise_sim'][i, j]:+.3f}" for j in range(len(c_result['init_names']))]
            ))
        # Average off-diagonal similarity
        K_c = len(c_result["init_names"])
        avg_sim_C = np.mean(c_result["pairwise_sim"][~np.eye(K_c, dtype=bool)])
        seed_original_pearson = results[seed_for_C_idx]["pearson"]
        print(f"    avg pairwise similarity of h_final across inits: {avg_sim_C:+.3f}")
        print(f"    seed's original Pearson: {seed_original_pearson:+.4f}")
        print(f"    H-step best Pearson across inits: {max(c_pearsons.values()):+.4f}")

        print(f"\n  Exp C verdict:")
        if avg_sim_C > 0.9:
            print(f"    [OK] H-step CONVERGES across inits (sim={avg_sim_C:.2f}) → direction 2 promising")
        elif avg_sim_C > 0.5:
            print(f"    [~] H-step PARTIALLY converges (sim={avg_sim_C:.2f}) → H-step helps but not fully")
        else:
            print(f"    [NO] H-step has MULTIPLE basins (sim={avg_sim_C:.2f}) → H-step also variance-prone")

        # Exp D: hidden-true substitution (diagnostic, model untouched)
        print(f"\n  Exp D: Substitute hidden_true into frozen dynamics")
        d_result = hidden_true_substitution(model_for_C, ds["visible"], ds["hidden"], device=device)
        print(f"    recon_null (no h):             {d_result['recon_null']:.4f}")
        print(f"    recon_encoder (self seed):     {d_result['recon_encoder']:.4f}")
        print(f"    recon_encoder_centered:        {d_result['recon_encoder_centered']:.4f}")
        print(f"    recon_true_scaled:             {d_result['recon_true_scaled']:.4f}")
        ratio = d_result['recon_true_scaled'] / d_result['recon_encoder']
        print(f"    ratio recon_true/recon_enc: {ratio:.3f}")
        print(f"\n  Exp D verdict:")
        if ratio < 0.9:
            print(f"    [OK] true h gives BETTER recon → dynamics OK, encoder undershoots → direction 2 helps")
        elif 0.95 <= ratio <= 1.05:
            print(f"    [NO] recon equivalent → ENCODER FOUND ALTERNATIVE SOLUTION (identifiability) → need stronger priors")
        elif ratio > 1.1:
            print(f"    [!!] encoder h gives BETTER recon than true h → dynamics learned pseudo-solution")

        analysis[ds_key] = {
            "rho_val": rho_val,
            "rho_mnull": rho_mnull,
            "rho_mshuf": rho_mshuf,
            "rho_hvar": rho_hvar,
            "C_in": C_in,
            "C_out": C_out,
            "ens_pearson": ens_pear,
            "hstep_avg_sim": avg_sim_C,
            "hstep_best_pearson": max(c_pearsons.values()),
            "hstep_original_pearson": seed_original_pearson,
            "d_ratio": ratio,
            "recon_null": d_result['recon_null'],
            "recon_encoder": d_result['recon_encoder'],
            "recon_true": d_result['recon_true_scaled'],
            "pearsons": pearsons.tolist(),
            "val_recons": val_recons.tolist(),
            "seeds": [r["seed"] for r in results],
        }

    # =========== Plots ============
    def save_single(title, plot_fn, path, figsize=(10, 6)):
        fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
        plot_fn(ax)
        fig.suptitle(title, fontsize=13, fontweight="bold")
        fig.savefig(path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    for ds_key, rr in all_results.items():
        results = rr["results"]; ds = rr["data"]
        a = analysis[ds_key]

        # Fig: val_recon vs Pearson scatter
        def plot_val_vs_pearson(ax, results=results, a=a):
            vr = np.array([r["val_recon"] for r in results])
            pe = np.array([r["pearson"] for r in results])
            ax.scatter(vr, pe, s=60, color="#1565c0")
            for r in results:
                ax.annotate(str(r["seed"]), (r["val_recon"], r["pearson"]),
                             fontsize=8, xytext=(3, 3), textcoords="offset points")
            ax.set_xlabel("val_recon (lower = better 重构)")
            ax.set_ylabel("Pearson to hidden_true")
            ax.axhline(0, color="grey", linewidth=0.5)
            ax.text(0.05, 0.95, f"Spearman ρ = {a['rho_val']:+.3f}",
                     transform=ax.transAxes, fontsize=11,
                     bbox=dict(facecolor="white", alpha=0.7))
            ax.grid(alpha=0.3)
        save_single(f"{ds['name']}: val_recon vs Pearson", plot_val_vs_pearson,
                     out_dir / f"fig_{ds_key}_val_vs_pearson.png")

        # Fig: all 4 unsup indicators
        def plot_all_rhos(ax, a=a):
            names = ["val_recon", "m_null", "m_shuf", "h_var"]
            rhos = [a['rho_val'], a['rho_mnull'], a['rho_mshuf'], a['rho_hvar']]
            colors = ["#c62828" if abs(r) > 0.5 else "#ff9800" if abs(r) > 0.3 else "#9e9e9e" for r in rhos]
            ax.bar(names, rhos, color=colors)
            ax.axhline(0, color="black", linewidth=0.5)
            ax.axhline(-0.5, color="green", linestyle="--", alpha=0.5, label="|ρ|=0.5 (strong)")
            ax.axhline(0.5, color="green", linestyle="--", alpha=0.5)
            ax.set_ylabel("Spearman ρ (vs Pearson)"); ax.grid(alpha=0.3, axis="y")
            for i, r in enumerate(rhos):
                ax.text(i, r + (0.03 if r >= 0 else -0.06), f"{r:+.2f}", ha="center", fontsize=10)
            ax.legend(fontsize=10)
        save_single(f"{ds['name']}: 无监督指标 vs Pearson 相关性",
                     plot_all_rhos, out_dir / f"fig_{ds_key}_unsup_correlations.png")

    # Summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# CVHI_Residual 诊断实验 (Exp A-D)\n\n")
        f.write(f"seeds: {seeds}, epochs={args.epochs}, top_k={args.top_k}\n\n")
        for ds_key, rr in all_results.items():
            results = rr["results"]; ds = rr["data"]; a = analysis[ds_key]
            f.write(f"## {ds['name']}\n\n")
            f.write(f"### Exp A: Multi-seed全指标\n\n")
            f.write(f"| seed | Pearson | val_recon | train_recon | m_null | m_shuf | h_var |\n")
            f.write(f"|---|---|---|---|---|---|---|\n")
            for r in results:
                f.write(f"| {r['seed']} | {r['pearson']:+.4f} | {r['val_recon']:.4f} | "
                        f"{r['train_recon']:.4f} | {r['m_null']:+.4f} | {r['m_shuf']:+.4f} | "
                        f"{r['h_var']:.3f} |\n")
            f.write(f"\n无监督指标与 |Pearson| 的 Spearman ρ:\n")
            f.write(f"- val_recon:   {a['rho_val']:+.3f}  (lower val → higher P = good)\n")
            f.write(f"- m_null:      {a['rho_mnull']:+.3f}\n")
            f.write(f"- m_shuf:      {a['rho_mshuf']:+.3f}\n")
            f.write(f"- h_var:       {a['rho_hvar']:+.3f}\n")
            f.write(f"\n### Exp B: Top-{args.top_k} val-selection + ensemble\n\n")
            f.write(f"- top-K 内 h_mean 两两 Pearson: C_in = {a['C_in']:+.3f}\n")
            f.write(f"- bot-K 内 h_mean 两两 Pearson: C_out = {a['C_out']:+.3f}\n")
            f.write(f"- top-K ensemble Pearson: {a['ens_pearson']:+.4f}\n")
            f.write(f"- top-K mean Pearson: {np.mean([results[i]['pearson'] for i in sorted(range(len(results)), key=lambda i: results[i]['val_recon'])[:args.top_k]]):+.4f}\n")
            f.write(f"\n### Exp C: H-step prototype\n\n")
            f.write(f"- seed (used for H-step): best val seed\n")
            f.write(f"- 4 个 init 的 h_final 两两 Pearson 平均: {a['hstep_avg_sim']:+.3f}\n")
            f.write(f"- seed 原 Pearson: {a['hstep_original_pearson']:+.4f}\n")
            f.write(f"- H-step 跨 init 最佳 Pearson: {a['hstep_best_pearson']:+.4f}\n")
            f.write(f"\n### Exp D: hidden_true 替代诊断\n\n")
            f.write(f"- recon_null (无 h):       {a['recon_null']:.4f}\n")
            f.write(f"- recon_encoder:           {a['recon_encoder']:.4f}\n")
            f.write(f"- recon_true (替换):       {a['recon_true']:.4f}\n")
            f.write(f"- ratio recon_true/recon_enc: {a['d_ratio']:.3f}\n")
        f.write("\n## 决策树\n\n")
        f.write("按 Exp A/B/C/D 结果综合判定是否需要上方向 2.\n")

    # Save raw results (no models, too big)
    clean_results = {ds_key: {
        "seeds": [r["seed"] for r in rr["results"]],
        "pearsons": [r["pearson"] for r in rr["results"]],
        "val_recons": [r["val_recon"] for r in rr["results"]],
        "m_nulls": [r["m_null"] for r in rr["results"]],
        "m_shufs": [r["m_shuf"] for r in rr["results"]],
        "h_vars": [r["h_var"] for r in rr["results"]],
        "h_means": [r["h_mean"].tolist() for r in rr["results"]],
        "analysis": analysis[ds_key],
    } for ds_key, rr in all_results.items()}

    import json
    with open(out_dir / "raw_results.json", "w") as f:
        json.dump(clean_results, f, indent=2, default=float)
    print(f"\n[OK] saved to: {out_dir}")


if __name__ == "__main__":
    main()

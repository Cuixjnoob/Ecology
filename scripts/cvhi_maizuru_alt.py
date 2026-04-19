"""Maizuru Bay with alternating 5:1 training (like Beninca best config).

Compare with joint training (current Maizuru result).
No temperature input (proven to work from species interactions alone).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.load_maizuru import load_maizuru
from scripts.cvhi_beninca_nbeddyn import (
    LatentDynamicsNet, make_model as make_model_beninca, get_param_groups,
    freeze, unfreeze, alpha_schedule, HP, PA, PB, LAM_H_ODE,
)

SEEDS = [42, 123, 456]
EPOCHS = 500
SP_ALL = ['Aurelia.sp', 'Engraulis.japonicus', 'Plotosus.japonicus',
          'Sebastes.inermis', 'Trachurus.japonicus', 'Girella.punctata',
          'Pseudolabrus.sieboldi', 'Parajulis.poecilepterus',
          'Halichoeres.tenuispinis', 'Chaenogobius.gulosus',
          'Pterogobius.zonoleucus', 'Tridentiger.trigonocephalus',
          'Siganus.fuscescens', 'Sphyraena.pinguis', 'Rudarius.ercodes']


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


def train_one(visible, hidden, seed, device, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    f_h = LatentDynamicsNet(N, d_hidden=32).to(device)
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
            post = epoch - warmup; h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))

        # Alternating 5:1
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
        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr = model.slice_out(out, 0, train_end)
        tr['visible'] = out['visible'][:, :train_end]; tr['G'] = out['G'][:, :train_end]
        losses = model.loss(tr, beta_kl=HP['lam_kl'], free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=HP['lam_cf'], lam_shuffle=HP['lam_cf'] * 0.6,
            lam_energy=2.0, min_energy=HP['min_energy'],
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.5 * h_w,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.2, lowpass_sigma=6.0, lam_rmse_log=0.1)
        loss_ode = torch.tensor(0.0, device=device)
        if h_w > 0:
            hm = out['mu'][:, :train_end]; xv = out['visible'][:, :train_end]
            hp = f_h(hm[:, :-1], xv[:, :-1])
            loss_ode = F.mse_loss(hp, hm[:, 1:].detach())
        (losses['total'] + LAM_H_ODE * h_w * loss_ode).backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out, train_end, T)
            vo['visible'] = out['visible'][:, train_end:T]; vo['G'] = out['G'][:, train_end:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=0.002, margin_shuf=0.001,
                lam_energy=2.0, min_energy=HP['min_energy'],
                lam_rollout=0.5, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.2, lowpass_sigma=6.0)
            vr = vl['recon_full'].item()
        if epoch > warmup + 15 and vr < best_val:
            best_val = vr
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_fh = {k: v.detach().cpu().clone() for k, v in f_h.state_dict().items()}

    unfreeze(fvis_params); unfreeze(enc_params)
    if best_state: model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oe = model(x_full, n_samples=30, rollout_K=3)
        hm = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    L = min(len(hm), len(hidden))
    te = train_end
    X_tr = np.column_stack([hm[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:te], rcond=None)
    X_all = np.column_stack([hm[:L], np.ones(L)])
    h_sc = X_all @ coef
    r_val = float(np.corrcoef(h_sc[te:L], hidden[te:L])[0, 1]) if L > te + 2 else 0
    del model, f_h
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return r_val


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_maizuru_alt")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, _ = load_maizuru(include_temp=False)
    species = [str(s) for s in species]

    print(f"Maizuru: alternating 5:1 + NbedDyn ODE (Beninca config)")
    print(f"No temperature. {len(species)} species.\n")

    results = {sp: [] for sp in species}
    total = len(species) * len(SEEDS)
    ri = 0
    for h_name in species:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        print(f"--- hidden={h_name} ---")
        for seed in SEEDS:
            ri += 1
            t0 = datetime.now()
            rv = train_one(visible, hidden, seed, device)
            dt = (datetime.now() - t0).total_seconds()
            print(f"  [{ri}/{total}] seed={seed}  val={rv:+.3f}  ({dt:.1f}s)")
            results[h_name].append(rv)

    print(f"\n{'='*60}")
    print(f"{'Species':<35} {'alt_5_1':>10} {'joint(ref)':>12}")
    print('-' * 60)
    # Joint training reference (no-temp results)
    ref = {'Aurelia.sp':-0.071,'Engraulis.japonicus':-0.030,'Plotosus.japonicus':0.318,
           'Sebastes.inermis':0.379,'Trachurus.japonicus':0.249,'Girella.punctata':0.226,
           'Pseudolabrus.sieboldi':0.500,'Parajulis.poecilepterus':0.439,
           'Halichoeres.tenuispinis':0.291,'Chaenogobius.gulosus':0.125,
           'Pterogobius.zonoleucus':0.270,'Tridentiger.trigonocephalus':-0.004,
           'Siganus.fuscescens':0.128,'Sphyraena.pinguis':-0.028,'Rudarius.ercodes':0.357}
    all_alt = []; all_jt = []
    for sp in species:
        alt = np.mean(results[sp])
        jt = ref.get(sp, 0)
        all_alt.append(alt); all_jt.append(jt)
        print(f"{sp:<35} {alt:>+10.3f} {jt:>+12.3f}")
    print('-' * 60)
    print(f"{'Overall':<35} {np.mean(all_alt):>+10.3f} {np.mean(all_jt):>+12.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Maizuru: alt 5:1 vs joint\n\n")
        f.write("| Species | alt_5_1 | joint |\n|---|---|---|\n")
        for sp in species:
            alt = np.mean(results[sp])
            jt = ref.get(sp, 0)
            f.write(f"| {sp} | {alt:+.3f} | {jt:+.3f} |\n")
        f.write(f"\n**Overall**: alt={np.mean(all_alt):+.3f}, joint={np.mean(all_jt):+.3f}\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

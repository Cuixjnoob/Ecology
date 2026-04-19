"""Supervised Eco-GNRD on Beninca: same architecture, but h is directly supervised.

This gives the empirical ceiling: "best our architecture can do WITH labels."
Compare with unsupervised Eco-GNRD (0.162 all, 0.117 val) to measure
how much performance is lost by being unsupervised.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.load_beninca import load_beninca

SEEDS = [42, 123, 456]
EPOCHS = 500
SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                 "Picophyto", "Filam_diatoms", "Ostracods",
                 "Harpacticoids", "Bacteria"]


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N, encoder_d=96, encoder_blocks=3, encoder_heads=4,
        takens_lags=(1, 2, 4, 8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp", use_formula_hints=True,
        use_G_field=True, num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def alpha_sched(ep, total):
    f = ep / max(1, total)
    if f <= 0.5: return 1.0
    if f >= 0.95: return 0.0
    return 1.0 - (f - 0.5) / (0.95 - 0.5)


def train_supervised(visible, hidden, seed, device, epochs=EPOCHS):
    """Train with hidden_true supervision on h."""
    torch.manual_seed(seed)
    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    h_true = torch.tensor(hidden, dtype=torch.float32, device=device)

    model = make_model(N, device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0006, weight_decay=1e-4)
    warmup = int(0.2 * epochs)
    ramp = max(1, int(0.2 * epochs))
    train_end = int(0.75 * T)

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    best_val = float('inf')
    best_state = None

    for epoch in range(epochs):
        if hasattr(model, 'G_anchor_alpha'):
            model.G_anchor_alpha = alpha_sched(epoch, epochs)
        if epoch < warmup:
            h_w = 0.0
        else:
            h_w = min(1.0, (epoch - warmup) / ramp)

        model.train()
        opt.zero_grad()

        out = model(x_full, n_samples=2, rollout_K=0)
        tr_out = model.slice_out(out, 0, train_end)
        tr_out["visible"] = out["visible"][:, :train_end]
        tr_out["G"] = out["G"][:, :train_end]

        # Standard unsupervised losses
        losses = model.loss(
            tr_out, beta_kl=0.017, free_bits=0.02,
            margin_null=0.002, margin_shuf=0.001,
            lam_necessary=9.5, lam_shuffle=9.5 * 0.6,
            lam_energy=2.0, min_energy=0.14,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=0.0,
            lam_hf=0.2, lowpass_sigma=6.0,
            lam_rmse_log=0.1,
        )

        # SUPERVISED LOSS: MSE between h and hidden_true (TRAIN PORTION ONLY)
        h_mu = out["mu"][:, :train_end]  # (B, T_train)
        h_target = h_true[:train_end]
        # Normalize hidden to match h scale
        h_target_norm = (h_target - h_target.mean()) / (h_target.std() + 1e-8)
        sup_loss = F.mse_loss(h_mu.squeeze(0), h_target_norm)

        total = losses["total"] + h_w * 5.0 * sup_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # Val (supervised on val too for model selection)
        with torch.no_grad():
            h_val = out["mu"][:, train_end:].squeeze(0)
            h_val_target = h_true[train_end:]
            h_val_norm = (h_val_target - h_target.mean()) / (h_target.std() + 1e-8)
            val_loss = F.mse_loss(h_val, h_val_norm).item()
        if epoch > warmup + 15 and val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    if best_state:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=0)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

    # Pearson: train-fit, val-eval
    L = min(len(h_mean), len(hidden))
    X_tr = np.column_stack([h_mean[:train_end], np.ones(train_end)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:train_end], rcond=None)
    X_all = np.column_stack([h_mean[:L], np.ones(L)])
    h_sc = X_all @ coef
    r_all = float(np.corrcoef(h_sc, hidden[:L])[0, 1])
    r_val = float(np.corrcoef(h_sc[train_end:L], hidden[train_end:L])[0, 1])

    del model
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return r_all, r_val


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_beninca_supervised")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full, species, _ = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]

    print("SUPERVISED Eco-GNRD on Beninca (uses hidden_true in training)")
    print(f"Seeds: {SEEDS}, Epochs: {EPOCHS}\n")

    results = {sp: [] for sp in SPECIES_ORDER}
    total = len(SPECIES_ORDER) * len(SEEDS)
    ri = 0

    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        print(f"--- hidden={h_name} ---")
        for seed in SEEDS:
            ri += 1
            t0 = datetime.now()
            r_all, r_val = train_supervised(visible, hidden, seed, device)
            dt = (datetime.now() - t0).total_seconds()
            print(f"  [{ri}/{total}] seed={seed}  all={r_all:+.3f}  val={r_val:+.3f}  ({dt:.1f}s)")
            results[h_name].append({"r_all": r_all, "r_val": r_val})

    print(f"\n{'='*70}")
    print(f"{'Species':<16} {'Sup(all)':>10} {'Sup(val)':>10} {'Unsup(val)':>12}")
    print('-' * 70)
    # Unsupervised reference from beninca_valonly
    unsup_ref = {'Cyclopoids': -0.200, 'Calanoids': 0.037, 'Rotifers': 0.495,
                 'Nanophyto': -0.075, 'Picophyto': 0.240, 'Filam_diatoms': 0.071,
                 'Ostracods': 0.390, 'Harpacticoids': 0.358, 'Bacteria': -0.260}
    all_sa = []; all_sv = []; all_uv = []
    for sp in SPECIES_ORDER:
        rs = results[sp]
        sa = np.mean([r["r_all"] for r in rs])
        sv = np.mean([r["r_val"] for r in rs])
        uv = unsup_ref.get(sp, 0)
        all_sa.append(sa); all_sv.append(sv); all_uv.append(uv)
        print(f"{sp:<16} {sa:>+10.3f} {sv:>+10.3f} {uv:>+12.3f}")
    print('-' * 70)
    print(f"{'Overall':<16} {np.mean(all_sa):>+10.3f} {np.mean(all_sv):>+10.3f} {np.mean(all_uv):>+12.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Supervised Eco-GNRD on Beninca\n\n")
        f.write("| Species | Sup(all) | Sup(val) | Unsup(val) |\n|---|---|---|---|\n")
        for sp in SPECIES_ORDER:
            rs = results[sp]
            sa = np.mean([r["r_all"] for r in rs])
            sv = np.mean([r["r_val"] for r in rs])
            uv = unsup_ref.get(sp, 0)
            f.write(f"| {sp} | {sa:+.3f} | {sv:+.3f} | {uv:+.3f} |\n")
        f.write(f"\n**Overall**: Sup(val)={np.mean(all_sv):+.3f}, Unsup(val)={np.mean(all_uv):+.3f}\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

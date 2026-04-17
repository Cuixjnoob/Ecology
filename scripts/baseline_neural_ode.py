"""Neural ODE baselines for hidden species recovery (fixed version).

Baseline 1: Neural ODE with multi-step rollout
  - dv/dt = f_theta(v), integrate over windows of W steps
  - Residual PCA as hidden proxy

Baseline 2: Latent ODE (Rubanova 2019 style)
  - RNN encoder -> z0 (initial latent)
  - d[v,z]/dt = f_theta([v,z])
  - z trajectory PCA as hidden proxy

Reference: Chen et al. 2018 "Neural Ordinary Differential Equations"
           Rubanova et al. 2019 "Latent ODEs for Irregularly-Sampled Time Series"
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from pathlib import Path
from datetime import datetime


def pearson(a, b):
    L = min(len(a), len(b))
    a, b = a[:L], b[:L]
    X = np.column_stack([a, np.ones(L)])
    coef, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
    a_sc = X @ coef
    ac = a_sc - a_sc.mean()
    bc = b - b.mean()
    return float(np.sum(ac * bc) / (np.sqrt(np.sum(ac**2) * np.sum(bc**2)) + 1e-8))


class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )
    def forward(self, t, z):
        return self.net(z)


# ===== Baseline 1: Neural ODE with multi-step rollout =====
def neural_ode_baseline(visible, hidden, seeds=[42, 123, 456],
                         epochs=200, lr=0.003, window=20, device='cuda'):
    """Neural ODE: integrate over windows, residual PCA."""
    T, N = visible.shape
    train_end = int(0.75 * T)
    x_all = torch.tensor(visible, dtype=torch.float32, device=device)

    best_pear = -1
    for seed in seeds:
        torch.manual_seed(seed)
        func = ODEFunc(N, 64).to(device)
        opt = torch.optim.Adam(func.parameters(), lr=lr)

        for ep in range(epochs):
            func.train(); opt.zero_grad()
            total_loss = 0
            # Sample random starting points within training set
            n_starts = min(16, train_end - window)
            starts = np.random.choice(train_end - window, n_starts, replace=False)
            for s in starts:
                t_w = torch.linspace(0, 1, window, device=device)
                x0 = x_all[s]
                pred = odeint(func, x0, t_w, method='rk4')  # (W, N)
                target = x_all[s:s+window]
                total_loss = total_loss + ((pred - target) ** 2).mean()
            loss = total_loss / n_starts
            loss.backward()
            nn.utils.clip_grad_norm_(func.parameters(), 1.0)
            opt.step()

        # Evaluate: one-step residual over full trajectory
        func.eval()
        with torch.no_grad():
            residuals = []
            for t in range(T - 1):
                t_w = torch.tensor([0.0, 1.0], device=device)
                pred = odeint(func, x_all[t], t_w, method='rk4')[1]
                residuals.append((x_all[t+1] - pred).cpu().numpy())
            residual = np.array(residuals)  # (T-1, N)

        if residual.std() > 1e-8:
            pca = PCA(n_components=1)
            h_est = pca.fit_transform(residual).flatten()
            p = pearson(h_est, hidden[1:])
            if p > best_pear: best_pear = p

    return best_pear


# ===== Baseline 2: Latent ODE (encoder-based) =====
class GRUEncoder(nn.Module):
    """GRU encoder: visible sequence -> latent z0."""
    def __init__(self, n_vis, latent_dim, hidden=32):
        super().__init__()
        self.gru = nn.GRU(n_vis, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, latent_dim)
    def forward(self, x):
        # x: (1, T, N)
        _, h = self.gru(x)  # h: (1, 1, hidden)
        return self.fc(h.squeeze(0))  # (1, latent_dim)


def latent_ode_baseline(visible, hidden, latent_dim=4, seeds=[42, 123, 456],
                         epochs=200, lr=0.003, window=20, device='cuda'):
    """Latent ODE: GRU encoder -> z0, ODE integration, decode visible."""
    T, N = visible.shape
    train_end = int(0.75 * T)
    aug_dim = N + latent_dim
    x_all = torch.tensor(visible, dtype=torch.float32, device=device)

    best_pear = -1
    for seed in seeds:
        torch.manual_seed(seed)
        func = ODEFunc(aug_dim, 64).to(device)
        encoder = GRUEncoder(N, latent_dim, hidden=32).to(device)
        decoder = nn.Linear(aug_dim, N).to(device)
        params = list(func.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
        opt = torch.optim.Adam(params, lr=lr)

        for ep in range(epochs):
            func.train(); encoder.train(); decoder.train()
            opt.zero_grad()

            total_loss = 0
            n_starts = min(8, train_end - window)
            starts = np.random.choice(train_end - window, n_starts, replace=False)

            for s in starts:
                # Encode: use visible up to start point to get z0
                context_len = min(s + 1, 50)  # max 50 steps context
                context = x_all[max(0, s - context_len + 1):s + 1].unsqueeze(0)
                z0_latent = encoder(context).squeeze(0)  # (latent_dim,)
                z0 = torch.cat([x_all[s], z0_latent])  # (aug_dim,)

                # Integrate
                t_w = torch.linspace(0, 1, window, device=device)
                z_traj = odeint(func, z0, t_w, method='rk4')  # (W, aug_dim)

                # Decode visible
                x_pred = decoder(z_traj)  # (W, N)
                target = x_all[s:s + window]
                total_loss = total_loss + ((x_pred - target) ** 2).mean()

            loss = total_loss / n_starts
            loss.backward()
            nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()

        # Extract latent trajectory over full time series
        func.eval(); encoder.eval()
        with torch.no_grad():
            z_trajs = []
            for s in range(0, T - window + 1, window // 2):
                context_len = min(s + 1, 50)
                context = x_all[max(0, s - context_len + 1):s + 1].unsqueeze(0)
                z0_latent = encoder(context).squeeze(0)
                z0 = torch.cat([x_all[s], z0_latent])
                t_w = torch.linspace(0, 1, min(window, T - s), device=device)
                z_traj = odeint(func, z0, t_w, method='rk4')
                z_trajs.append(z_traj[:, N:].cpu().numpy())  # latent part

            # Stitch: use overlapping windows, take middle portions
            latent_full = np.zeros((T, latent_dim))
            counts = np.zeros(T)
            idx = 0
            for s in range(0, T - window + 1, window // 2):
                w = min(window, T - s)
                latent_full[s:s+w] += z_trajs[idx][:w]
                counts[s:s+w] += 1
                idx += 1
            counts = np.maximum(counts, 1)
            latent_full /= counts[:, None]

        if latent_full.std() > 1e-8:
            pca = PCA(n_components=1)
            h_est = pca.fit_transform(latent_full).flatten()
            p = pearson(h_est, hidden)
            if p > best_pear: best_pear = p

    return best_pear


# ===== Data loaders =====
def load_huisman():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = d["N_all"]; R_all = d["resources"]
    results = []
    for sp_idx in range(6):
        vis = np.concatenate([np.delete(N_all, sp_idx, axis=1), R_all], axis=1)
        vis = (vis + 0.01) / (vis.mean(axis=0, keepdims=True) + 1e-3)
        hid = N_all[:, sp_idx]
        hid = (hid + 0.01) / (hid.mean() + 1e-3)
        results.append((vis.astype(np.float32), hid.astype(np.float32)))
    return results


def load_beninca():
    from scripts.load_beninca import load_beninca as _load
    full, species, _ = _load(include_nutrients=True)
    species = [str(s) for s in species]
    SP = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
          "Picophyto", "Filam_diatoms", "Ostracods", "Harpacticoids", "Bacteria"]
    results = []
    for h in SP:
        h_idx = species.index(h)
        vis = np.delete(full, h_idx, axis=1).astype(np.float32)
        hid = full[:, h_idx].astype(np.float32)
        results.append((vis, hid, h))
    return results


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_neural_ode_baseline")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("=" * 80)
    print(f"NEURAL ODE BASELINES v2 (multi-step rollout, device={device})")
    print("=" * 80)

    # Huisman
    print("\nHuisman (6->1):")
    dh = load_huisman()
    np1 = []; np2 = []
    for i, (v, h) in enumerate(dh):
        p1 = neural_ode_baseline(v, h, device=device)
        p2 = latent_ode_baseline(v, h, device=device)
        np1.append(p1); np2.append(p2)
        print(f"  sp{i+1}: NODE={p1:+.3f}  LatentODE={p2:+.3f}")
    print(f"  Overall: NODE={np.mean(np1):+.3f}  LatentODE={np.mean(np2):+.3f}")
    print(f"  Ref: Eco-GNRD=+0.502, LSTM=+0.535")

    # Beninca
    print("\nBeninca (9->1):")
    db = load_beninca()
    bp1 = []; bp2 = []
    for v, h, name in db:
        p1 = neural_ode_baseline(v, h, device=device)
        p2 = latent_ode_baseline(v, h, device=device)
        bp1.append(p1); bp2.append(p2)
        print(f"  {name:<16}: NODE={p1:+.3f}  LatentODE={p2:+.3f}")
    print(f"  Overall: NODE={np.mean(bp1):+.3f}  LatentODE={np.mean(bp2):+.3f}")
    print(f"  Ref: Eco-GNRD=+0.162, LSTM=+0.108")

    # Summary
    print(f"\n{'='*80}")
    print(f"{'Method':<14} {'Huisman':>10} {'Beninca':>10}")
    print("-" * 40)
    print(f"{'VAR+PCA':<14} {'+0.027':>10} {'+0.022':>10}")
    print(f"{'MLP+PCA':<14} {'+0.042':>10} {'+0.030':>10}")
    print(f"{'Neural ODE':<14} {np.mean(np1):>+10.3f} {np.mean(bp1):>+10.3f}")
    print(f"{'Latent ODE':<14} {np.mean(np2):>+10.3f} {np.mean(bp2):>+10.3f}")
    print(f"{'LSTM':<14} {'+0.535':>10} {'+0.108':>10}")
    print(f"{'Eco-GNRD':<14} {'+0.502':>10} {'+0.162':>10}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Neural ODE Baselines v2 (multi-step rollout)\n\n")
        f.write("| Method | Huisman | Beninca |\n|---|---|---|\n")
        f.write(f"| VAR+PCA | +0.027 | +0.022 |\n")
        f.write(f"| MLP+PCA | +0.042 | +0.030 |\n")
        f.write(f"| Neural ODE | {np.mean(np1):+.3f} | {np.mean(bp1):+.3f} |\n")
        f.write(f"| Latent ODE | {np.mean(np2):+.3f} | {np.mean(bp2):+.3f} |\n")
        f.write(f"| LSTM | +0.535 | +0.108 |\n")
        f.write(f"| **Eco-GNRD** | **+0.502** | **+0.162** |\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

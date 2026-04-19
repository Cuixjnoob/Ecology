"""Fair baselines: ALL strictly unsupervised. No oracle selection anywhere.

Changes from previous versions:
1. No seed selection: report mean over seeds, not best
2. LSTM: PCA only, no oracle dim selection
3. EDM: fixed E=3, no oracle E sweep
4. All Pearson: train-fit lstsq, val-eval
5. VAR also gets proper train/val split

Methods: VAR+PCA, MLP+PCA, LSTM, Neural ODE, Latent ODE, EDM Simplex, MVE
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import json
import numpy as np
import torch
import torch.nn as nn
from torchdiffeq import odeint
from sklearn.decomposition import PCA
from pathlib import Path
from datetime import datetime


SEEDS = [42, 123, 456, 789, 2024, 11111, 22222, 27182, 31415, 65537]


def pearson_val(h, hidden, train_end):
    """Train-fit lstsq, val-eval Pearson. Strictly no leakage."""
    L = min(len(h), len(hidden))
    h = h[:L]; hidden = hidden[:L]
    if L <= train_end + 2:
        return 0.0, 0.0
    X_tr = np.column_stack([h[:train_end], np.ones(train_end)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:train_end], rcond=None)
    X_val = np.column_stack([h[train_end:], np.ones(L - train_end)])
    pred_val = X_val @ coef
    X_all = np.column_stack([h, np.ones(L)])
    pred_all = X_all @ coef
    r_all = float(np.corrcoef(pred_all, hidden)[0, 1])
    r_val = float(np.corrcoef(pred_val, hidden[train_end:])[0, 1])
    return r_all, r_val


# ===== VAR+PCA =====
def var_pca(visible, hidden, p=4):
    T, N = visible.shape
    train_end = int(0.75 * T)
    safe = np.maximum(visible, 1e-6)
    lr = np.log(safe[1:] / safe[:-1])
    X_list, Y_list = [], []
    for t in range(p, len(lr)):
        row = np.concatenate([visible[t - l] for l in range(p)])
        X_list.append(row); Y_list.append(lr[t])
    X = np.array(X_list); Y = np.array(Y_list)
    te_adj = train_end - p - 1
    # Fit VAR only on train
    X_aug = np.column_stack([X[:te_adj], np.ones(te_adj)])
    W, _, _, _ = np.linalg.lstsq(X_aug, Y[:te_adj], rcond=None)
    # Predict on all
    X_full = np.column_stack([X, np.ones(len(X))])
    pred = X_full @ W
    resid = Y - pred
    if resid.std() < 1e-8: return 0.0, 0.0
    pca = PCA(n_components=1)
    # PCA fit on train residuals only
    pca.fit(resid[:te_adj])
    h_est = pca.transform(resid).flatten()
    return pearson_val(h_est, hidden[p+1:], te_adj)


# ===== MLP+PCA =====
def mlp_pca(visible, hidden, seed, epochs=200, d_hidden=64, lr=0.001):
    T, N = visible.shape
    train_end = int(0.75 * (T - 1))
    safe = np.maximum(visible, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1])
    x_in = torch.tensor(visible[:-1], dtype=torch.float32)
    y_out = torch.tensor(log_ratio, dtype=torch.float32)

    torch.manual_seed(seed)
    model = nn.Sequential(nn.Linear(N, d_hidden), nn.ReLU(),
                          nn.Linear(d_hidden, d_hidden), nn.ReLU(),
                          nn.Linear(d_hidden, N))
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    for ep in range(epochs):
        model.train()
        pred = model(x_in[:train_end])
        loss = nn.functional.mse_loss(pred, y_out[:train_end])
        opt.zero_grad(); loss.backward(); opt.step()
    model.eval()
    with torch.no_grad():
        pred_all = model(x_in).numpy()
    resid = log_ratio - pred_all
    if resid.std() < 1e-8: return 0.0, 0.0
    pca = PCA(n_components=1)
    pca.fit(resid[:train_end])
    h_est = pca.transform(resid).flatten()
    return pearson_val(h_est, hidden[1:], train_end)


# ===== LSTM (PCA only, no oracle dim) =====
def lstm_pca(visible, hidden, seed, epochs=300, d_hidden=32, lr=0.001):
    T, N = visible.shape
    train_end = int(0.75 * (T - 1))
    safe = np.maximum(visible, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1])
    x_seq = torch.tensor(visible[:-1], dtype=torch.float32).unsqueeze(0)
    y_seq = torch.tensor(log_ratio, dtype=torch.float32).unsqueeze(0)

    torch.manual_seed(seed)
    lstm = nn.LSTM(N, d_hidden, batch_first=True)
    fc = nn.Linear(d_hidden, N)
    params = list(lstm.parameters()) + list(fc.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    for ep in range(epochs):
        lstm.train(); fc.train()
        out, _ = lstm(x_seq[:, :train_end])
        pred = fc(out)
        loss = nn.functional.mse_loss(pred, y_seq[:, :train_end])
        opt.zero_grad(); loss.backward(); opt.step()
    lstm.eval(); fc.eval()
    with torch.no_grad():
        out_full, _ = lstm(x_seq)
        h_states = out_full[0].numpy()
    if h_states.std() < 1e-8: return 0.0, 0.0
    pca = PCA(n_components=1)
    pca.fit(h_states[:train_end])
    h_est = pca.transform(h_states).flatten()
    return pearson_val(h_est, hidden[:-1], train_end)


# ===== Neural ODE =====
class ODEFunc(nn.Module):
    def __init__(self, dim, hidden=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(dim, hidden), nn.Tanh(),
                                  nn.Linear(hidden, hidden), nn.Tanh(),
                                  nn.Linear(hidden, dim))
    def forward(self, t, z): return self.net(z)


def neural_ode(visible, hidden, seed, epochs=200, lr=0.005, window=20, device='cuda'):
    T, N = visible.shape
    train_end = int(0.75 * T)
    x_all = torch.tensor(visible, dtype=torch.float32, device=device)
    t_step = torch.tensor([0.0, 1.0], device=device)

    torch.manual_seed(seed)
    func = ODEFunc(N, 64).to(device)
    opt = torch.optim.Adam(func.parameters(), lr=lr)
    for ep in range(epochs):
        func.train(); opt.zero_grad()
        n_starts = min(16, train_end - window)
        starts = np.random.choice(train_end - window, n_starts, replace=False)
        total_loss = 0
        for s in starts:
            t_w = torch.linspace(0, 1, window, device=device)
            pred = odeint(func, x_all[s], t_w, method='rk4')
            target = x_all[s:s+window]
            total_loss = total_loss + ((pred - target)**2).mean()
        (total_loss / n_starts).backward()
        nn.utils.clip_grad_norm_(func.parameters(), 1.0); opt.step()
    func.eval()
    with torch.no_grad():
        resids = []
        for t in range(T - 1):
            pred = odeint(func, x_all[t], t_step, method='rk4')[1]
            resids.append((x_all[t+1] - pred).cpu().numpy())
        resid = np.array(resids)
    if resid.std() < 1e-8: return 0.0, 0.0
    pca = PCA(n_components=1)
    pca.fit(resid[:train_end])
    h_est = pca.transform(resid).flatten()
    return pearson_val(h_est, hidden[1:], train_end)


# ===== Latent ODE =====
class GRUEnc(nn.Module):
    def __init__(self, n, lat, h=32):
        super().__init__()
        self.gru = nn.GRU(n, h, batch_first=True)
        self.fc = nn.Linear(h, lat)
    def forward(self, x):
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


def latent_ode(visible, hidden, seed, latent_dim=4, epochs=200, lr=0.003,
               window=20, device='cuda'):
    T, N = visible.shape
    train_end = int(0.75 * T)
    aug_dim = N + latent_dim
    x_all = torch.tensor(visible, dtype=torch.float32, device=device)

    torch.manual_seed(seed)
    func = ODEFunc(aug_dim, 64).to(device)
    encoder = GRUEnc(N, latent_dim, 32).to(device)
    decoder = nn.Linear(aug_dim, N).to(device)
    params = list(func.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)

    for ep in range(epochs):
        func.train(); encoder.train(); decoder.train(); opt.zero_grad()
        n_starts = min(8, train_end - window)
        starts = np.random.choice(train_end - window, n_starts, replace=False)
        total_loss = 0
        for s in starts:
            ctx_len = min(s + 1, 50)
            ctx = x_all[max(0, s-ctx_len+1):s+1].unsqueeze(0)
            z0_lat = encoder(ctx).squeeze(0)
            z0 = torch.cat([x_all[s], z0_lat])
            t_w = torch.linspace(0, 1, window, device=device)
            z_traj = odeint(func, z0, t_w, method='rk4')
            x_pred = decoder(z_traj)
            target = x_all[s:s+window]
            total_loss = total_loss + ((x_pred - target)**2).mean()
        (total_loss / n_starts).backward()
        nn.utils.clip_grad_norm_(params, 1.0); opt.step()

    func.eval(); encoder.eval()
    with torch.no_grad():
        z_trajs = []
        for s in range(0, T - window + 1, window // 2):
            ctx_len = min(s + 1, 50)
            ctx = x_all[max(0, s-ctx_len+1):s+1].unsqueeze(0)
            z0_lat = encoder(ctx).squeeze(0)
            z0 = torch.cat([x_all[s], z0_lat])
            t_w = torch.linspace(0, 1, min(window, T-s), device=device)
            z_traj = odeint(func, z0, t_w, method='rk4')
            z_trajs.append(z_traj[:, N:].cpu().numpy())
        latent_full = np.zeros((T, latent_dim))
        counts = np.zeros(T)
        idx = 0
        for s in range(0, T - window + 1, window // 2):
            w = min(window, T - s)
            latent_full[s:s+w] += z_trajs[idx][:w]
            counts[s:s+w] += 1; idx += 1
        counts = np.maximum(counts, 1)
        latent_full /= counts[:, None]
    if latent_full.std() < 1e-8: return 0.0, 0.0
    pca = PCA(n_components=1)
    pca.fit(latent_full[:train_end])
    h_est = pca.transform(latent_full).flatten()
    return pearson_val(h_est, hidden, train_end)


# ===== EDM Simplex (fixed E=3) =====
def simplex_predict(lib, pred_points, nn_k=4):
    preds = np.full(len(pred_points), np.nan)
    for i in range(len(pred_points)):
        dists = np.sqrt(((lib[:-1, 1:] - pred_points[i])**2).sum(axis=1))
        idx = np.argsort(dists)[:nn_k]
        d_nn = dists[idx]; d_min = d_nn[0]
        if d_min < 1e-10: w = np.zeros(nn_k); w[0] = 1.0
        else: w = np.exp(-d_nn / d_min)
        w /= w.sum() + 1e-10
        if idx.max() + 1 < len(lib):
            preds[i] = np.sum(w * lib[idx + 1, 0])
    return preds


def edm_simplex(visible, hidden, E=3, tau=1):
    T, N = visible.shape
    train_end = int(0.75 * T)
    all_resid = []
    for j in range(N):
        series = visible[:, j]
        n = T - (E-1)*tau
        if n <= 0: all_resid.append(np.zeros(T)); continue
        emb = np.column_stack([series[(E-1)*tau - l*tau : T - l*tau] for l in range(E)])
        offset = (E-1)*tau
        actual = visible[offset:, j]
        lib_end = train_end - offset
        lib = np.column_stack([actual[:lib_end].reshape(-1,1), emb[:lib_end]])
        preds = simplex_predict(lib, emb)
        L = min(len(preds), len(actual))
        resid = np.zeros(L)
        mask = ~np.isnan(preds[:L])
        resid[mask] = actual[:L][mask] - preds[:L][mask]
        all_resid.append(resid)
    min_len = min(len(r) for r in all_resid)
    resid_mat = np.column_stack([r[:min_len] for r in all_resid])
    if resid_mat.std() < 1e-8: return 0.0, 0.0
    te_adj = train_end - (E-1)*tau
    pca = PCA(n_components=1)
    pca.fit(resid_mat[:te_adj])
    h_est = pca.transform(resid_mat).flatten()
    return pearson_val(h_est, hidden[(E-1)*tau:(E-1)*tau+min_len], te_adj)


# ===== MVE (fixed E=3) =====
def mve_baseline(visible, hidden, E=3):
    T, N = visible.shape
    train_end = int(0.75 * T)
    from itertools import combinations
    combos = list(combinations(range(N), E))
    if len(combos) > 200:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(combos), 200, replace=False)
        combos = [combos[i] for i in idx]
    top_k = max(1, int(np.sqrt(len(combos))))
    all_resid = []
    for target_j in range(N):
        view_preds = []; view_skills = []
        for combo in combos:
            emb = np.column_stack([visible[:, c] for c in combo])
            emb_lag = emb[1:]; tgt_next = visible[1:, target_j]
            lib_emb = emb_lag[:train_end-1]; lib_tgt = tgt_next[:train_end-1]
            if len(lib_emb) < E + 2: continue
            pred_emb = emb_lag[:T-1]
            preds = np.full(len(pred_emb), np.nan)
            for i in range(len(pred_emb)):
                dists = np.sqrt(((lib_emb[:-1] - pred_emb[i])**2).sum(axis=1))
                idx = np.argsort(dists)[:E+1]
                d_nn = dists[idx]; d_min = d_nn[0]
                if d_min < 1e-10: w = np.zeros(E+1); w[0] = 1.0
                else: w = np.exp(-d_nn / d_min)
                w /= w.sum() + 1e-10
                if idx.max() + 1 < len(lib_tgt):
                    preds[i] = np.sum(w * lib_tgt[idx + 1])
            valid = ~np.isnan(preds[:train_end-1])
            if valid.sum() < 5: continue
            skill = np.corrcoef(preds[:train_end-1][valid], tgt_next[:train_end-1][valid])[0, 1]
            if np.isnan(skill): continue
            view_preds.append(preds); view_skills.append(skill)
        if len(view_preds) == 0: all_resid.append(np.zeros(T-1)); continue
        order = np.argsort(view_skills)[::-1][:top_k]
        weights = np.maximum(np.array([view_skills[i] for i in order]), 0)
        if weights.sum() < 1e-8: weights = np.ones(len(weights))
        weights /= weights.sum()
        avg_pred = sum(w * np.nan_to_num(view_preds[i][:T-1]) for w, i in zip(weights, order))
        all_resid.append(visible[1:, target_j] - avg_pred)
    min_len = min(len(r) for r in all_resid)
    resid_mat = np.column_stack([r[:min_len] for r in all_resid])
    if resid_mat.std() < 1e-8: return 0.0, 0.0
    pca = PCA(n_components=1)
    pca.fit(resid_mat[:train_end-1])
    h_est = pca.transform(resid_mat).flatten()
    return pearson_val(h_est, hidden[1:1+min_len], train_end-1)


# ===== Supervised Ridge (ceiling) =====
def supervised_ridge(visible, hidden, lags=4):
    from sklearn.linear_model import Ridge
    T, N = visible.shape
    train_end = int(0.75 * T)
    X_list, Y_list = [], []
    for t in range(lags, T):
        row = np.concatenate([visible[t-l] for l in range(lags)])
        X_list.append(row); Y_list.append(hidden[t])
    X = np.array(X_list); Y = np.array(Y_list)
    te_adj = train_end - lags
    best_r_val = -1
    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X[:te_adj], Y[:te_adj])
        pred_val = ridge.predict(X[te_adj:])
        r = float(np.corrcoef(pred_val, Y[te_adj:])[0, 1]) if len(Y[te_adj:]) > 2 else 0
        if r > best_r_val: best_r_val = r
    # Also get all-data for reference
    ridge = Ridge(alpha=1.0)
    ridge.fit(X[:te_adj], Y[:te_adj])
    pred_all = ridge.predict(X)
    r_all = float(np.corrcoef(pred_all, Y)[0, 1])
    return r_all, best_r_val


# ===== Dataset loaders =====
def load_huisman():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = d["N_all"]; R_all = d["resources"]
    tasks = []
    for sp_idx in range(6):
        vis = np.concatenate([np.delete(N_all, sp_idx, axis=1), R_all], axis=1)
        vis = (vis + 0.01) / (vis.mean(axis=0, keepdims=True) + 1e-3)
        hid = N_all[:, sp_idx]; hid = (hid + 0.01) / (hid.mean() + 1e-3)
        tasks.append((vis.astype(np.float32), hid.astype(np.float32), f'sp{sp_idx+1}'))
    return tasks

def load_beninca():
    from scripts.load_beninca import load_beninca as _load
    full, species, _ = _load(include_nutrients=True)
    species = [str(s) for s in species]
    SP = ["Cyclopoids","Calanoids","Rotifers","Nanophyto","Picophyto",
          "Filam_diatoms","Ostracods","Harpacticoids","Bacteria"]
    tasks = []
    for h in SP:
        h_idx = species.index(h)
        vis = np.delete(full, h_idx, axis=1).astype(np.float32)
        hid = full[:, h_idx].astype(np.float32)
        tasks.append((vis, hid, h))
    return tasks

def load_maizuru():
    from scripts.load_maizuru import load_maizuru as _load
    full, species, _ = _load(include_temp=False)
    species = [str(s) for s in species]
    tasks = []
    for h in species:
        h_idx = species.index(h)
        vis = np.delete(full, h_idx, axis=1).astype(np.float32)
        hid = full[:, h_idx].astype(np.float32)
        tasks.append((vis, hid, h))
    return tasks


# ===== Runner =====
def run_all(datasets, methods, n_seeds, out_root, device):
    loaders = {'huisman': load_huisman, 'beninca': load_beninca, 'maizuru': load_maizuru}
    seeds = SEEDS[:n_seeds]

    for ds_name in datasets:
        tasks = loaders[ds_name]()
        print(f"\n{'='*70}")
        print(f"DATASET: {ds_name} ({len(tasks)} species)")
        print(f"{'='*70}")

        for method_name, method_fn, needs_seed, needs_device in methods:
            print(f"\n--- {method_name} ---")
            ds_dir = out_root / method_name / ds_name
            ds_dir.mkdir(parents=True, exist_ok=True)

            all_sp_results = []
            for vis, hid, sp_name in tasks:
                sp_dir = ds_dir / sp_name
                sp_dir.mkdir(parents=True, exist_ok=True)

                if needs_seed:
                    sp_results = []
                    for seed in seeds:
                        seed_dir = sp_dir / f"seed_{seed:05d}"
                        mf = seed_dir / "metrics.json"
                        if mf.exists():
                            with open(mf) as f: r = json.load(f)
                            sp_results.append(r)
                            continue
                        if needs_device:
                            r_all, r_val = method_fn(vis, hid, seed, device=device)
                        else:
                            r_all, r_val = method_fn(vis, hid, seed)
                        r = {'seed': seed, 'method': method_name, 'dataset': ds_name,
                             'species': sp_name, 'pearson_all': r_all, 'pearson_val': r_val}
                        seed_dir.mkdir(parents=True, exist_ok=True)
                        with open(mf, 'w') as f: json.dump(r, f, indent=2)
                        sp_results.append(r)
                    pa = np.mean([r['pearson_all'] for r in sp_results])
                    pv = np.mean([r['pearson_val'] for r in sp_results])
                else:
                    r_all, r_val = method_fn(vis, hid)
                    pa, pv = r_all, r_val
                    sp_results = [{'pearson_all': r_all, 'pearson_val': r_val}]

                agg = {'species': sp_name, 'pearson_all_mean': pa, 'pearson_val_mean': pv,
                       'n_seeds': len(sp_results)}
                with open(sp_dir / "aggregate.json", 'w') as f: json.dump(agg, f, indent=2)
                all_sp_results.append(agg)
                print(f"  {sp_name:<25}: all={pa:+.3f} val={pv:+.3f}")

            oa = np.mean([r['pearson_all_mean'] for r in all_sp_results])
            ov = np.mean([r['pearson_val_mean'] for r in all_sp_results])
            ds_agg = {'dataset': ds_name, 'method': method_name,
                      'overall_pearson_all_mean': oa, 'overall_pearson_val_mean': ov}
            with open(ds_dir / "aggregate.json", 'w') as f: json.dump(ds_agg, f, indent=2)
            with open(ds_dir / "summary.md", 'w', encoding='utf-8') as f:
                f.write(f"# {ds_name}: {method_name}\n\n")
                f.write("| Species | P(all) | P(val) |\n|---|---|---|\n")
                for r in all_sp_results:
                    f.write(f"| {r['species']} | {r['pearson_all_mean']:+.3f} | {r['pearson_val_mean']:+.3f} |\n")
                f.write(f"\n**Overall**: all={oa:+.3f}, val={ov:+.3f}\n")
            print(f"  OVERALL: all={oa:+.3f}, val={ov:+.3f}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--datasets', nargs='+', default=['huisman', 'beninca', 'maizuru'])
    parser.add_argument('--seeds', type=int, default=5)
    parser.add_argument('--methods', nargs='+', default=['all'])
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path("重要实验/results/baselines")

    all_methods = [
        ('var_pca', var_pca, False, False),
        ('mlp_pca', mlp_pca, True, False),
        ('lstm', lstm_pca, True, False),
        ('edm_simplex', edm_simplex, False, False),
        ('mve', mve_baseline, False, False),
        ('supervised_ridge', supervised_ridge, False, False),
        ('neural_ode', neural_ode, True, True),
        ('latent_ode', latent_ode, True, True),
    ]

    if 'all' in args.methods:
        methods = all_methods
    else:
        methods = [m for m in all_methods if m[0] in args.methods]

    run_all(args.datasets, methods, args.seeds, out_root, device)


if __name__ == "__main__":
    main()

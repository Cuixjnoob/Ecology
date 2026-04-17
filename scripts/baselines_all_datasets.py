"""Baselines on all 4 datasets: LV, Holling, Huisman, Beninca.

Baseline 1: VAR(p) + PCA residual (linear, no deep learning)
  - Fit linear VAR on visible species
  - PCA on prediction residuals
  - First PC = hidden species estimate

Baseline 2: MLP predictor + PCA residual (simple neural)
  - Train MLP to predict log(x_{t+1}/x_t) from x_t
  - PCA on prediction residuals
  - First PC = hidden species estimate

Compare with CVHI results.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from datetime import datetime
from sklearn.decomposition import PCA

# ===== Evaluation =====
def pearson(a, b):
    L = min(len(a), len(b))
    a, b = a[:L], b[:L]
    # lstsq scaling (same as evaluate())
    X = np.column_stack([a, np.ones(L)])
    coef, _, _, _ = np.linalg.lstsq(X, b, rcond=None)
    a_sc = X @ coef
    ac = a_sc - a_sc.mean()
    bc = b - b.mean()
    return float(np.sum(ac * bc) / (np.sqrt(np.sum(ac**2) * np.sum(bc**2)) + 1e-8))


# ===== Baseline 1: VAR + PCA residual =====
def var_pca_baseline(visible, hidden, p=4):
    """VAR(p) on visible, PCA on residuals."""
    T, N = visible.shape
    safe = np.maximum(visible, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1])  # (T-1, N)

    # Build VAR(p) input: [x_t, x_{t-1}, ..., x_{t-p+1}]
    X_list, Y_list = [], []
    for t in range(p, len(log_ratio)):
        row = []
        for lag in range(p):
            row.append(visible[t - lag])  # visible state at t, t-1, ..., t-p+1
        X_list.append(np.concatenate(row))
        Y_list.append(log_ratio[t])
    X = np.array(X_list)  # (T-p, N*p)
    Y = np.array(Y_list)  # (T-p, N)

    # Fit linear model: Y = X @ W + b
    X_aug = np.column_stack([X, np.ones(len(X))])
    W, _, _, _ = np.linalg.lstsq(X_aug, Y, rcond=None)
    Y_pred = X_aug @ W
    residuals = Y - Y_pred  # (T-p, N)

    # PCA on residuals
    if residuals.std() < 1e-8:
        return 0.0
    pca = PCA(n_components=1)
    h_est = pca.fit_transform(residuals).flatten()
    hidden_aligned = hidden[p+1:]  # align with residuals
    return pearson(h_est, hidden_aligned)


# ===== Baseline 2: MLP predictor + PCA residual =====
def mlp_pca_baseline(visible, hidden, epochs=200, d_hidden=64, lr=0.001, seeds=[42, 123, 456]):
    """Train MLP to predict log-ratio, PCA on residuals."""
    T, N = visible.shape
    safe = np.maximum(visible, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1])

    train_end = int(0.75 * (T - 1))
    x_in = torch.tensor(visible[:-1], dtype=torch.float32)  # (T-1, N)
    y_out = torch.tensor(log_ratio, dtype=torch.float32)     # (T-1, N)

    best_pear = -1
    for seed in seeds:
        torch.manual_seed(seed)
        model = nn.Sequential(
            nn.Linear(N, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, N),
        )
        opt = torch.optim.Adam(model.parameters(), lr=lr)

        for ep in range(epochs):
            model.train()
            pred = model(x_in[:train_end])
            loss = nn.functional.mse_loss(pred, y_out[:train_end])
            opt.zero_grad(); loss.backward(); opt.step()

        model.eval()
        with torch.no_grad():
            pred_all = model(x_in).numpy()
        residuals = log_ratio - pred_all  # (T-1, N)

        if residuals.std() < 1e-8:
            continue
        pca = PCA(n_components=1)
        h_est = pca.fit_transform(residuals).flatten()
        p = pearson(h_est, hidden[1:])
        if p > best_pear:
            best_pear = p
    return best_pear


# ===== Baseline 3: LSTM hidden state =====
def lstm_baseline(visible, hidden, epochs=300, d_hidden=32, lr=0.001, seeds=[42, 123, 456]):
    """Train LSTM, use hidden state's first dim as h estimate."""
    T, N = visible.shape
    safe = np.maximum(visible, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1])
    train_end = int(0.75 * (T - 1))

    x_seq = torch.tensor(visible[:-1], dtype=torch.float32).unsqueeze(0)  # (1, T-1, N)
    y_seq = torch.tensor(log_ratio, dtype=torch.float32).unsqueeze(0)     # (1, T-1, N)

    best_pear = -1
    for seed in seeds:
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
            out_full, _ = lstm(x_seq)  # (1, T-1, d_hidden)
            h_states = out_full[0].numpy()  # (T-1, d_hidden)

        # Try each dim of LSTM hidden state
        for dim in range(min(d_hidden, 8)):
            p = pearson(h_states[:, dim], hidden[:-1])
            if p > best_pear:
                best_pear = p

        # Also try PCA on LSTM hidden states
        if h_states.std() > 1e-8:
            pca = PCA(n_components=1)
            h_pc1 = pca.fit_transform(h_states).flatten()
            p_pca = pearson(h_pc1, hidden[:-1])
            if p_pca > best_pear:
                best_pear = p_pca

    return best_pear


# ===== Dataset loaders =====
def load_lv():
    """Generate LV 5+1 data."""
    from data.partial_lv_mvp import generate_partial_lv_mvp_system
    data = generate_partial_lv_mvp_system(seed=42)
    visible = data.visible_states.numpy()      # (T, 5)
    hidden = data.hidden_states.numpy()[:, 0]  # (T,)
    return visible.astype(np.float32), hidden.astype(np.float32), "LV (5+1)"


def load_holling():
    """Generate Holling II 5+1 data."""
    from data.partial_nonlinear_mvp import generate_partial_nonlinear_mvp_system
    data = generate_partial_nonlinear_mvp_system(seed=42)
    visible = data.visible_states.numpy()
    hidden = data.hidden_states.numpy()[:, 0]
    return visible.astype(np.float32), hidden.astype(np.float32), "Holling (5+1)"


def load_huisman():
    """Load Huisman chaos data."""
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    N_all = d["N_all"]      # (T, 6)
    R_all = d["resources"]   # (T, 5)
    # 6->1: average over all species as hidden
    results = []
    for sp_idx in range(6):
        vis_sp = np.delete(N_all, sp_idx, axis=1)
        visible = np.concatenate([vis_sp, R_all], axis=1)
        visible = (visible + 0.01) / (visible.mean(axis=0, keepdims=True) + 1e-3)
        hidden = N_all[:, sp_idx]
        hidden = (hidden + 0.01) / (hidden.mean() + 1e-3)
        results.append((visible.astype(np.float32), hidden.astype(np.float32)))
    return results, "Huisman (6->1 rotation)"


def load_beninca():
    """Load Beninca data."""
    from scripts.load_beninca import load_beninca as _load
    full, species, _ = _load(include_nutrients=True)
    species = [str(s) for s in species]
    SPECIES_ORDER = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                     "Picophyto", "Filam_diatoms", "Ostracods",
                     "Harpacticoids", "Bacteria"]
    results = []
    for h_name in SPECIES_ORDER:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        results.append((visible, hidden, h_name))
    return results, "Beninca (9->1 rotation)"


# ===== Main =====
def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_baselines")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("BASELINES: VAR+PCA and MLP+PCA on all datasets")
    print("=" * 80)

    all_results = {}

    # --- LV ---
    try:
        vis, hid, name = load_lv()
        p_var = var_pca_baseline(vis, hid)
        p_mlp = mlp_pca_baseline(vis, hid)
        p_lstm = lstm_baseline(vis, hid)
        print(f"\n{name}:")
        print(f"  VAR+PCA:  {p_var:+.3f}")
        print(f"  MLP+PCA:  {p_mlp:+.3f}")
        print(f"  LSTM:     {p_lstm:+.3f}")
        print(f"  Ref CVHI: ~0.88")
        all_results["LV"] = {"var": p_var, "mlp": p_mlp, "lstm": p_lstm, "cvhi": 0.88}
    except Exception as e:
        print(f"LV failed: {e}")
        import traceback; traceback.print_exc()

    # --- Holling ---
    try:
        vis, hid, name = load_holling()
        p_var = var_pca_baseline(vis, hid)
        p_mlp = mlp_pca_baseline(vis, hid)
        p_lstm = lstm_baseline(vis, hid)
        print(f"\n{name}:")
        print(f"  VAR+PCA:  {p_var:+.3f}")
        print(f"  MLP+PCA:  {p_mlp:+.3f}")
        print(f"  LSTM:     {p_lstm:+.3f}")
        print(f"  Ref CVHI: ~0.85")
        all_results["Holling"] = {"var": p_var, "mlp": p_mlp, "lstm": p_lstm, "cvhi": 0.85}
    except Exception as e:
        print(f"Holling failed: {e}")
        import traceback; traceback.print_exc()

    # --- Huisman ---
    try:
        datasets, name = load_huisman()
        print(f"\n{name}:")
        var_ps = []; mlp_ps = []; lstm_ps = []
        for i, (vis, hid) in enumerate(datasets):
            pv = var_pca_baseline(vis, hid)
            pm = mlp_pca_baseline(vis, hid)
            pl = lstm_baseline(vis, hid)
            var_ps.append(pv); mlp_ps.append(pm); lstm_ps.append(pl)
            print(f"  sp{i+1}: VAR={pv:+.3f}  MLP={pm:+.3f}  LSTM={pl:+.3f}")
        print(f"  Overall VAR:  {np.mean(var_ps):+.3f}")
        print(f"  Overall MLP:  {np.mean(mlp_ps):+.3f}")
        print(f"  Overall LSTM: {np.mean(lstm_ps):+.3f}")
        print(f"  Ref CVHI:     +0.506")
        all_results["Huisman"] = {"var": np.mean(var_ps), "mlp": np.mean(mlp_ps),
                                   "lstm": np.mean(lstm_ps), "cvhi": 0.506}
    except Exception as e:
        print(f"Huisman failed: {e}")
        import traceback; traceback.print_exc()

    # --- Beninca ---
    try:
        datasets, name = load_beninca()
        print(f"\n{name}:")
        var_ps = []; mlp_ps = []; lstm_ps = []
        for vis, hid, sp_name in datasets:
            pv = var_pca_baseline(vis, hid)
            pm = mlp_pca_baseline(vis, hid)
            pl = lstm_baseline(vis, hid)
            var_ps.append(pv); mlp_ps.append(pm); lstm_ps.append(pl)
            print(f"  {sp_name:<16}: VAR={pv:+.3f}  MLP={pm:+.3f}  LSTM={pl:+.3f}")
        print(f"  Overall VAR:  {np.mean(var_ps):+.3f}")
        print(f"  Overall MLP:  {np.mean(mlp_ps):+.3f}")
        print(f"  Overall LSTM: {np.mean(lstm_ps):+.3f}")
        print(f"  Ref CVHI:     +0.162")
        all_results["Beninca"] = {"var": np.mean(var_ps), "mlp": np.mean(mlp_ps),
                                   "lstm": np.mean(lstm_ps), "cvhi": 0.162}
    except Exception as e:
        print(f"Beninca failed: {e}")
        import traceback; traceback.print_exc()

    # --- Summary table ---
    print(f"\n{'=' * 80}")
    print("SUMMARY")
    print(f"{'=' * 80}")
    print(f"{'Dataset':<16} {'VAR+PCA':>10} {'MLP+PCA':>10} {'LSTM':>10} {'CVHI':>10} {'CVHI vs best':>14}")
    print("-" * 90)
    for ds in ["LV", "Holling", "Huisman", "Beninca"]:
        if ds in all_results:
            r = all_results[ds]
            lstm_v = r.get("lstm", 0)
            best_bl = max(r["var"], r["mlp"], lstm_v)
            delta = r["cvhi"] - best_bl
            print(f"{ds:<16} {r['var']:>+10.3f} {r['mlp']:>+10.3f} {lstm_v:>+10.3f} {r['cvhi']:>+10.3f} {delta:>+14.3f}")

    # Save
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Baselines: VAR+PCA and MLP+PCA\n\n")
        f.write("| Dataset | VAR+PCA | MLP+PCA | CVHI | CVHI advantage |\n")
        f.write("|---|---|---|---|---|\n")
        for ds in ["LV", "Holling", "Huisman", "Beninca"]:
            if ds in all_results:
                r = all_results[ds]
                best_bl = max(r["var"], r["mlp"])
                f.write(f"| {ds} | {r['var']:+.3f} | {r['mlp']:+.3f} | {r['cvhi']:+.3f} | {r['cvhi']-best_bl:+.3f} |\n")
        if "Huisman" in all_results:
            f.write("\n## Huisman per-species\n\n| Species | VAR | MLP |\n|---|---|---|\n")
            for i, (v, m) in enumerate(zip(all_results["Huisman"]["var_per"],
                                            all_results["Huisman"]["mlp_per"])):
                f.write(f"| sp{i+1} | {v:+.3f} | {m:+.3f} |\n")
        if "Beninca" in all_results:
            f.write("\n## Beninca per-species\n\n| Species | VAR | MLP |\n|---|---|---|\n")
            sp_names = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
                        "Picophyto", "Filam_diatoms", "Ostracods", "Harpacticoids", "Bacteria"]
            for n, v, m in zip(sp_names, all_results["Beninca"]["var_per"],
                                all_results["Beninca"]["mlp_per"]):
                f.write(f"| {n} | {v:+.3f} | {m:+.3f} |\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

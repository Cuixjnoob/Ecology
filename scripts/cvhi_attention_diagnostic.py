"""Part A: 可视化 encoder 的 species-attention 和 temporal-attention,
看现有 attention 到底 attend 到了什么.

步骤:
1. 构造 G_anchor_first + anneal_late 模型
2. 在 Portal 上训练 300 epochs
3. eval 时用 hook 捕获 species_attn 和 temporal_attn 的权重
4. 画图:
   - species_attn 热图 (每个 t 取 mean)
   - temporal_attn 热图 (每个 species 的 time×time)
   - 对比 attention peak vs 事件月份 (高 |Δx|) vs 耦合月份 (PCA top-K)
5. 定量: attention entropy, concentration, align with "event" metric
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import (
    load_portal, evaluate, _configure_matplotlib,
)


def alpha_schedule(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one(visible, hidden_eval, device, seed=42, epochs=300):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = CVHI_Residual(
        num_visible=N,
        encoder_d=48, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1, 2, 4, 8, 12), encoder_dropout=0.2,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=True, use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=0.0008, weight_decay=1e-4)
    warmup = int(0.2 * epochs); ramp = max(1, int(0.2 * epochs))
    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    train_end = int(0.75 * T)
    best_val = float("inf"); best_state = None
    m_null, m_shuf, min_e = 0.002, 0.001, 0.05

    for epoch in range(epochs):
        model.G_anchor_alpha = alpha_schedule(epoch, epochs)
        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / (epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w
        model.train(); opt.zero_grad()
        out = model(x, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end)
        losses = model.loss(tr_out, beta_kl=0.03, free_bits=0.02,
            margin_null=m_null, margin_shuf=m_shuf,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=min_e,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=(1.0, 0.5, 0.25),
            lam_hf=0.0, lowpass_sigma=6.0)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sched.step()
        with torch.no_grad():
            val_out = model.slice_out(out, train_end, T)
            val_losses = model.loss(val_out, h_weight=1.0,
                margin_null=m_null, margin_shuf=m_shuf,
                lam_necessary=5.0, lam_shuffle=3.0,
                lam_energy=2.0, min_energy=min_e,
                lam_rollout=lam_r, rollout_weights=(1.0, 0.5, 0.25),
                lam_hf=0.0, lowpass_sigma=6.0)
            val_recon = val_losses["recon_full"].item()
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, x


def capture_attention(model, x):
    """Hook encoder's species_attn and temporal_attn, capture weights."""
    # forward model to get encoder's internal attention weights
    # We need to patch nn.MultiheadAttention forward to return weights

    model.eval()
    captured_sp_attn = []   # list of (B*T, N, N) per block
    captured_tp_attn = []   # list of (B*N, T, T) per block

    # Access encoder directly
    enc = model.encoder.encoders[0]

    # Manually forward the encoder, patching attention calls
    with torch.no_grad():
        visible = x
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape

        features = enc._build_features(visible)
        h = enc.input_proj(features) + enc.species_emb.view(1, 1, N, -1)
        from models.cvhi import sinusoidal_pe
        pe = sinusoidal_pe(T, enc.d_model, h.device)
        h = h + pe.view(1, T, 1, enc.d_model)

        # Pre-compute attention biases if needed (不启用此时)
        species_mask, temporal_mask = None, None

        for block in enc.blocks:
            # Species attention
            B_, T_, N_, D = h.shape
            x_s = h.reshape(B_ * T_, N_, D)
            a, sp_weights = block["species_attn"](
                x_s, x_s, x_s, attn_mask=species_mask,
                need_weights=True, average_attn_weights=True)
            captured_sp_attn.append(sp_weights.cpu().numpy())  # (B*T, N, N)
            x_s = block["norm1"](x_s + a)
            h = x_s.reshape(B_, T_, N_, D)
            # Temporal attention
            x_t = h.permute(0, 2, 1, 3).contiguous().reshape(B_ * N_, T_, D)
            a, tp_weights = block["temporal_attn"](
                x_t, x_t, x_t, attn_mask=temporal_mask,
                need_weights=True, average_attn_weights=True)
            captured_tp_attn.append(tp_weights.cpu().numpy())  # (B*N, T, T)
            x_t = block["norm2"](x_t + a)
            h = x_t.reshape(B_, N_, T_, D).permute(0, 2, 1, 3).contiguous()
            h = block["norm3"](h + block["ffn"](h))

    return captured_sp_attn, captured_tp_attn


def compute_event_metrics(x):
    """给出每个 (t, j) 的 'event 强度' 度量."""
    # x: torch (1, T, N)
    visible = x.cpu().numpy()[0]  # (T, N)
    T, N = visible.shape
    eps = 1e-6
    safe = np.clip(visible, eps, None)
    dx = np.log(safe[1:] / safe[:-1])  # (T-1, N)
    # pad to T
    dx_pad = np.concatenate([dx, dx[-1:]], axis=0)  # (T, N)
    # per-timestep |Δx| magnitude (collective event indicator)
    dx_mag = np.linalg.norm(dx_pad, axis=1)  # (T,)
    # PCA coupling weight (Option 3 style)
    U, S, Vt = np.linalg.svd(dx_pad, full_matrices=False)
    K = 3
    dx_coll = U[:, :K] @ np.diag(S[:K]) @ Vt[:K]
    w_collective = (dx_coll ** 2) / (dx_pad ** 2 + eps)
    w_collective = np.clip(w_collective, 0, 1)
    # overall coupling = mean across species
    coupling_per_t = w_collective.mean(axis=1)  # (T,)
    return dx_mag, coupling_per_t, w_collective


def main():
    _configure_matplotlib()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_attn_diagnostic")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vis, hid = load_portal("OT")

    print("Training G_anchor_first + anneal_late (seed 42, 300 epochs)...")
    model, x_tensor = train_one(vis, hid, device)
    model.eval()

    with torch.no_grad():
        out_eval = model(x_tensor, n_samples=30, rollout_K=3)
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    pearson, _ = evaluate(h_mean, hid)
    print(f"Trained Pearson = {pearson:+.3f}")

    # Capture attention
    print("\nCapturing attention weights...")
    sp_attns, tp_attns = capture_attention(model, x_tensor)
    print(f"  species_attn layers: {len(sp_attns)}")
    print(f"  species_attn shape (per layer): {sp_attns[0].shape}  (= B*T × N × N)")
    print(f"  temporal_attn layers: {len(tp_attns)}")
    print(f"  temporal_attn shape (per layer): {tp_attns[0].shape}  (= B*N × T × T)")

    T = vis.shape[0]; N = vis.shape[1]

    # Unpack dimensions
    # species: (B*T, N, N) → (T, N, N) for B=1
    sp_avg = np.mean([a.reshape(1, T, N, N)[0] for a in sp_attns], axis=0)  # avg over blocks: (T, N, N)
    # temporal: (B*N, T, T) → (N, T, T) for B=1
    tp_avg = np.mean([a.reshape(1, N, T, T)[0] for a in tp_attns], axis=0)  # (N, T, T)

    # Event metrics
    dx_mag, coupling_per_t, w_collective = compute_event_metrics(x_tensor)

    # ============ Analysis ============
    # 1. Species attention: mean over time, show N×N matrix
    sp_mean_over_t = sp_avg.mean(axis=0)  # (N, N)
    # 2. Temporal attention concentration: entropy per (n, t_q)
    # H(attention)_t_q = -sum_t_k p log p. Low = focused, high = spread.
    # Attention weights sum to 1 across t_k. entropy ∈ [0, log(T)].
    tp_entropy = -np.sum(tp_avg * np.log(tp_avg + 1e-12), axis=-1)  # (N, T)
    mean_entropy_per_species = tp_entropy.mean(axis=-1)  # (N,)
    max_entropy = np.log(T)
    print(f"\nTemporal attention entropy per species (normalized, 0=focused, 1=uniform):")
    for i in range(N):
        print(f"  species {i}: {mean_entropy_per_species[i]/max_entropy:.3f}")

    # 3. correlate attention "attention to time t_k" with event magnitude at t_k
    # For each species n, get per-t_k attention totals (summed over queries)
    tp_per_species_tk = tp_avg.sum(axis=1) / T  # (N, T), normalized attention received per time step
    # Correlate with dx_mag
    from scipy.stats import pearsonr, spearmanr
    print(f"\nCorrelation of temporal-attn(t_k) with |Δx|(t_k), per species:")
    for n in range(N):
        r_p, _ = pearsonr(tp_per_species_tk[n], dx_mag)
        r_s, _ = spearmanr(tp_per_species_tk[n], dx_mag)
        print(f"  species {n}: Pearson={r_p:+.3f}  Spearman={r_s:+.3f}")
    # overall (all species combined)
    all_attn = tp_per_species_tk.mean(axis=0)
    r_p, _ = pearsonr(all_attn, dx_mag)
    r_s, _ = spearmanr(all_attn, dx_mag)
    print(f"  ALL avg:   Pearson={r_p:+.3f}  Spearman={r_s:+.3f}")

    # Correlate with coupling
    r_c_p, _ = pearsonr(all_attn, coupling_per_t)
    r_c_s, _ = spearmanr(all_attn, coupling_per_t)
    print(f"  attention vs coupling: Pearson={r_c_p:+.3f}  Spearman={r_c_s:+.3f}")

    # ============ Plots ============
    fig, axes = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)

    # 1. Species attention heatmap (mean over time)
    ax = axes[0, 0]
    im = ax.imshow(sp_mean_over_t, cmap="hot", aspect="auto")
    ax.set_title("Species attention (mean over time)\nrow=query, col=key")
    ax.set_xlabel("species key"); ax.set_ylabel("species query")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # 2. Temporal attention "received" per (species, time)
    ax = axes[0, 1]
    im = ax.imshow(tp_per_species_tk, cmap="viridis", aspect="auto")
    ax.set_title("Temporal attention received per (species, time)")
    ax.set_xlabel("time t_k"); ax.set_ylabel("species")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # 3. |Δx| vs average attention
    ax = axes[0, 2]
    tt = np.arange(T)
    ax.plot(tt, all_attn / all_attn.max(), label="avg attn (norm)", color="tab:blue", alpha=0.7)
    ax.plot(tt, dx_mag / dx_mag.max(), label="|Δx| (norm)", color="tab:red", alpha=0.7)
    ax.plot(tt, coupling_per_t, label="coupling w (PCA top-3)", color="tab:green", alpha=0.7)
    ax.set_title(f"Avg attn vs event metrics\nPearson(attn, |Δx|)={r_p:+.2f}, Pearson(attn, coupling)={r_c_p:+.2f}")
    ax.legend(); ax.grid(alpha=0.25)
    ax.set_xlabel("time"); ax.set_ylabel("normalized")

    # 4. Species attention entropy per species (bar)
    ax = axes[1, 0]
    ent_norm = mean_entropy_per_species / max_entropy
    ax.bar(range(N), ent_norm, color=["tab:blue" if e > 0.8 else "tab:orange" for e in ent_norm])
    ax.axhline(1.0, color="gray", ls="--", alpha=0.5, label="uniform")
    ax.set_title("Per-species temporal attention entropy\n(low=focused, high=uniform)")
    ax.set_xlabel("species"); ax.set_ylabel("entropy / log(T)")
    ax.legend(); ax.grid(alpha=0.25)

    # 5. Temporal attention heatmap for first species
    ax = axes[1, 1]
    im = ax.imshow(tp_avg[0], cmap="hot", aspect="auto")
    ax.set_title(f"Temporal attention for species 0\n(t_q × t_k)")
    ax.set_xlabel("t_k"); ax.set_ylabel("t_q")
    plt.colorbar(im, ax=ax, fraction=0.04)

    # 6. Correlation distribution: |Δx| vs attn for each species
    ax = axes[1, 2]
    corrs = []
    for n in range(N):
        r, _ = pearsonr(tp_per_species_tk[n], dx_mag)
        corrs.append(r)
    ax.bar(range(N), corrs, color="tab:blue")
    ax.axhline(0, color="gray", ls="--", alpha=0.5)
    ax.set_title("Per-species: corr(attn, |Δx|)")
    ax.set_xlabel("species"); ax.set_ylabel("Pearson")
    ax.grid(alpha=0.25)

    fig.savefig(out_dir / "fig_attention_diagnostic.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

    # Save diagnostic summary
    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Attention Diagnostic\n\n")
        f.write(f"- Portal OT, 20-seed style config (seed=42, 300 epochs)\n")
        f.write(f"- Trained Pearson: {pearson:+.3f}\n\n")
        f.write(f"## Temporal attention summary\n\n")
        f.write(f"- Correlation (avg_attn, |Δx|): Pearson {r_p:+.3f}, Spearman {r_s:+.3f}\n")
        f.write(f"- Correlation (avg_attn, coupling_w): Pearson {r_c_p:+.3f}, Spearman {r_c_s:+.3f}\n\n")
        f.write(f"## Per-species entropy (normalized)\n\n")
        for n in range(N):
            f.write(f"- species {n}: {ent_norm[n]:.3f}\n")

    np.savez(out_dir / "raw.npz",
              sp_avg=sp_avg, tp_avg=tp_avg,
              dx_mag=dx_mag, coupling_per_t=coupling_per_t, w_collective=w_collective,
              tp_per_species_tk=tp_per_species_tk,
              mean_entropy_per_species=mean_entropy_per_species)

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

"""CVHI-R 训练工具 — 性能优化版.

安全优化 (0 accuracy 影响):
  1. torch.compile(model) — JIT 编译同 op 更快
  2. opt.zero_grad(set_to_none=True) — 置 None 更快
  3. 避免不必要的 .cpu() 调用
  4. best_state 只在真正需要 CPU 拷贝时转

经典 NN 改进 (几乎无副作用):
  5. EMA of weights — 权重指数移动平均, 泛化更好
  6. Snapshot ensemble — 末期 N 个 checkpoint 的 h 平均
"""
from __future__ import annotations

import copy
import numpy as np
import torch
from typing import Dict, Optional, List


class ModelEMA:
    """经典 EMA: ema_w = decay·ema_w + (1-decay)·current_w, decay~0.999."""
    def __init__(self, model, decay=0.999):
        self.ema = copy.deepcopy(model)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.decay = decay
        self.step_count = 0

    @torch.no_grad()
    def update(self, model):
        self.step_count += 1
        # Warmup: 早期训练, 当前权重比例更高
        eff_decay = min(self.decay, (1 + self.step_count) / (10 + self.step_count))
        # Handle torch.compile wrapper
        src_model = model._orig_mod if hasattr(model, "_orig_mod") else model
        msd = src_model.state_dict()
        for k, v in self.ema.state_dict().items():
            if v.dtype.is_floating_point:
                v.mul_(eff_decay).add_(msd[k].to(v.dtype), alpha=1 - eff_decay)

    def state_dict(self):
        return self.ema.state_dict()


def alpha_schedule_default(epoch, epochs, start=0.5, end=0.95):
    f = epoch / max(1, epochs)
    if f <= start: return 1.0
    if f >= end: return 0.0
    return 1.0 - (f - start) / (end - start)


def train_one_fast(
    model,
    visible: np.ndarray,
    hidden_eval: np.ndarray,
    device: str = "cuda",
    epochs: int = 300,
    lr: float = 0.0008,
    warmup_frac: float = 0.2,
    ramp_frac: float = 0.2,
    train_end_frac: float = 0.75,
    best_state_track: bool = True,
    # Loss params
    beta_kl: float = 0.03, free_bits: float = 0.02,
    margin_null: float = 0.002, margin_shuf: float = 0.001,
    lam_necessary: float = 5.0, lam_shuffle: float = 3.0,
    lam_energy: float = 2.0, min_energy: float = 0.05,
    lam_smooth: float = 0.02, lam_sparse: float = 0.02,
    lam_rollout: float = 0.5,
    rollout_weights: tuple = (1.0, 0.5, 0.25),
    lam_hf: float = 0.0, lowpass_sigma: float = 6.0,
    # Stage 1 ecology priors
    lam_rmse_log: float = 0.0,
    lam_mte_prior: float = 0.0,
    mte_prior_target: Optional[torch.Tensor] = None,
    # Stage 1c corrected MTE
    lam_mte_shape: float = 0.0,
    mte_target_log_r: Optional[torch.Tensor] = None,
    # Stage 2 Klausmeier sign priors
    lam_stoich_sign: float = 0.0,
    stoich_pos_pairs: tuple = (),
    stoich_neg_pairs: tuple = (),
    # Augmentation
    input_dropout_prob: float = 0.0,
    # Anchor schedule
    alpha_schedule_fn=None,
    # torch.compile
    use_compile: bool = True,
    # 经典 NN 改进
    use_ema: bool = True, ema_decay: float = 0.999,
    use_snapshot_ensemble: bool = True, snapshot_last_frac: float = 0.15, n_snapshots: int = 5,
) -> Dict:
    """通用 CVHI_Residual 单 seed 训练 (加速版).

    调用前先:
      torch.manual_seed(seed)
    model 必须已经 .to(device).

    返回: dict with pearson, val_recon, d_ratio, h_mean
    """
    from scripts.cvhi_residual_L1L3_diagnostics import evaluate, hidden_true_substitution

    if alpha_schedule_fn is None:
        alpha_schedule_fn = alpha_schedule_default

    T, N = visible.shape
    x_full = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)

    # ✓ Optimization 1: torch.compile — only if Triton is available
    if use_compile:
        try:
            import triton  # type: ignore
            model = torch.compile(model, mode="default")
        except ImportError:
            pass  # Triton not installed (common on Windows); skip silently
        except Exception as e:
            print(f"  [WARN] torch.compile failed: {e}, proceeding without")

    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # ✓ Optimization 5: EMA of weights
    ema = None
    if use_ema:
        underlying = model._orig_mod if hasattr(model, "_orig_mod") else model
        ema = ModelEMA(underlying, decay=ema_decay)

    # ✓ Optimization 6: Snapshot ensemble (末期 n_snapshots 个)
    snapshots: List[Dict] = []
    snapshot_start_epoch = int((1.0 - snapshot_last_frac) * epochs)
    snapshot_interval = max(1, int(snapshot_last_frac * epochs / max(1, n_snapshots)))
    warmup = int(warmup_frac * epochs)
    ramp = max(1, int(ramp_frac * epochs))

    def lr_lambda(step):
        if step < 50: return step / 50
        p = (step - 50) / max(1, epochs - 50)
        return 0.5 * (1 + np.cos(np.pi * p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)

    train_end = int(train_end_frac * T)
    best_val = float("inf")
    best_state = None

    mte_target_dev = mte_prior_target.to(device) if mte_prior_target is not None else None
    mte_target_log_r_dev = mte_target_log_r.to(device) if mte_target_log_r is not None else None

    for epoch in range(epochs):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule_fn(epoch, epochs)
        elif hasattr(model, "_orig_mod") and hasattr(model._orig_mod, "G_anchor_alpha"):
            model._orig_mod.G_anchor_alpha = alpha_schedule_fn(epoch, epochs)

        if epoch < warmup:
            h_w, K_r, lam_r = 0.0, 0, 0.0
        else:
            post = epoch - warmup
            h_w = min(1.0, post / ramp)
            k_ramp = min(1.0, post / max(1, epochs - warmup) * 2)
            K_r = max(1 if h_w > 0 else 0, int(round(k_ramp * 3)))
            lam_r = 0.5 * h_w

        # Input dropout augmentation
        if input_dropout_prob > 0 and epoch > warmup:
            mask = (torch.rand(1, T, 1, device=device) > input_dropout_prob).float()
            x_train = x_full * mask + (1 - mask) * x_full.mean(dim=1, keepdim=True)
        else:
            x_train = x_full

        model.train()
        # ✓ Optimization 2: set_to_none
        opt.zero_grad(set_to_none=True)

        out = model(x_train, n_samples=2, rollout_K=K_r)
        tr_out = model.slice_out(out, 0, train_end) if hasattr(model, 'slice_out') \
                 else model._orig_mod.slice_out(out, 0, train_end)
        # Forward visible + G (for RMSE log / MTE priors)
        tr_out["visible"] = out["visible"][:, 0:train_end]
        tr_out["G"] = out["G"][:, 0:train_end]

        losses = _call_loss(
            model, tr_out,
            beta_kl=beta_kl, free_bits=free_bits,
            margin_null=margin_null, margin_shuf=margin_shuf,
            lam_necessary=lam_necessary, lam_shuffle=lam_shuffle,
            lam_energy=lam_energy, min_energy=min_energy,
            lam_smooth=lam_smooth, lam_sparse=lam_sparse,
            h_weight=h_w, lam_rollout=lam_r,
            rollout_weights=rollout_weights,
            lam_hf=lam_hf, lowpass_sigma=lowpass_sigma,
            lam_rmse_log=lam_rmse_log,
            lam_mte_prior=lam_mte_prior,
            mte_prior_target=mte_target_dev,
            lam_mte_shape=lam_mte_shape,
            mte_target_log_r=mte_target_log_r_dev,
            lam_stoich_sign=lam_stoich_sign,
            stoich_pos_pairs=stoich_pos_pairs,
            stoich_neg_pairs=stoich_neg_pairs,
        )
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        # EMA update after optim step
        if ema is not None:
            ema.update(model)

        # Snapshot ensemble (末期)
        if use_snapshot_ensemble and epoch >= snapshot_start_epoch \
           and (epoch - snapshot_start_epoch) % snapshot_interval == 0:
            sd_src = model._orig_mod.state_dict() if hasattr(model, "_orig_mod") else model.state_dict()
            snapshots.append({k: v.detach().cpu().clone() for k, v in sd_src.items()})

        # Validation
        if best_state_track:
            with torch.no_grad():
                # If no input dropout, reuse 'out' for val (fast path)
                if input_dropout_prob == 0:
                    val_out = model.slice_out(out, train_end, T) if hasattr(model, 'slice_out') \
                              else model._orig_mod.slice_out(out, train_end, T)
                    val_out["visible"] = out["visible"][:, train_end:T]
                    val_out["G"] = out["G"][:, train_end:T]
                else:
                    # with augmentation, need clean forward for val
                    out_val = model(x_full, n_samples=2, rollout_K=K_r)
                    val_out = model.slice_out(out_val, train_end, T) if hasattr(model, 'slice_out') \
                              else model._orig_mod.slice_out(out_val, train_end, T)
                    val_out["visible"] = out_val["visible"][:, train_end:T]
                    val_out["G"] = out_val["G"][:, train_end:T]

                val_losses = _call_loss(
                    model, val_out, h_weight=1.0,
                    margin_null=margin_null, margin_shuf=margin_shuf,
                    lam_necessary=lam_necessary, lam_shuffle=lam_shuffle,
                    lam_energy=lam_energy, min_energy=min_energy,
                    lam_rollout=lam_r, rollout_weights=rollout_weights,
                    lam_hf=lam_hf, lowpass_sigma=lowpass_sigma,
                    beta_kl=beta_kl, free_bits=free_bits,
                    lam_smooth=lam_smooth, lam_sparse=lam_sparse,
                    lam_rmse_log=0.0, lam_mte_prior=0.0,
                    lam_mte_shape=0.0, lam_stoich_sign=0.0,
                )
                val_recon = val_losses["recon_full"].item()   # unavoidable sync
            if epoch > warmup + 15 and val_recon < best_val:
                best_val = val_recon
                # State copy (CPU)
                sd = model.state_dict() if hasattr(model, "state_dict") else model._orig_mod.state_dict()
                best_state = {k: v.detach().cpu().clone() for k, v in sd.items()}

    # === 评测阶段: 组合 best_state / EMA / snapshots ===
    underlying = model._orig_mod if hasattr(model, "_orig_mod") else model

    def eval_h(sd):
        """给定 state_dict, 返回 h_mean (numpy)."""
        underlying.load_state_dict(sd)
        underlying.eval()
        with torch.no_grad():
            out_eval = underlying(x_full, n_samples=30, rollout_K=3)
            return out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()

    results_per_method = {}

    # 1) best_state (基线, val-selected)
    if best_state is not None:
        h_best = eval_h(best_state)
        pear_best, _ = evaluate(h_best, hidden_eval)
        results_per_method["best_val"] = {"pearson": float(pear_best), "h": h_best}

    # 2) EMA weights
    if ema is not None:
        h_ema = eval_h(ema.state_dict())
        pear_ema, _ = evaluate(h_ema, hidden_eval)
        results_per_method["ema"] = {"pearson": float(pear_ema), "h": h_ema}

    # 3) Snapshot ensemble: 各 snapshot 的 h 平均
    if use_snapshot_ensemble and len(snapshots) >= 2:
        h_list = [eval_h(sd) for sd in snapshots]
        h_snap = np.mean(h_list, axis=0)
        pear_snap, _ = evaluate(h_snap, hidden_eval)
        results_per_method["snapshot"] = {"pearson": float(pear_snap), "h": h_snap}

    # 4) 联合: best_val + EMA + snapshot 平均 (如都可用)
    all_h = []
    for name in ["best_val", "ema", "snapshot"]:
        if name in results_per_method:
            all_h.append(results_per_method[name]["h"])
    if len(all_h) >= 2:
        h_combined = np.mean(all_h, axis=0)
        pear_combined, _ = evaluate(h_combined, hidden_eval)
        results_per_method["combined"] = {"pearson": float(pear_combined), "h": h_combined}

    # 选 best_val 作主返回 (向后兼容), 其余作 extra
    primary = results_per_method.get("best_val", results_per_method.get(
        list(results_per_method.keys())[0] if results_per_method else "best_val", None))

    if primary is None:
        # 极端 fallback: 直接用 current model
        h_mean = eval_h(underlying.state_dict())
        pear, _ = evaluate(h_mean, hidden_eval)
        primary = {"pearson": float(pear), "h": h_mean}

    h_mean = primary["h"]
    pear = primary["pearson"]
    # d_ratio (用 best_state 下的 model 算)
    if best_state is not None:
        underlying.load_state_dict(best_state)
    diag = hidden_true_substitution(underlying, visible, hidden_eval, device)
    d_ratio = diag["recon_true_scaled"] / diag["recon_encoder"]

    return {
        "pearson": pear, "val_recon": best_val, "d_ratio": d_ratio,
        "h_mean": h_mean,
        # Extra: per-method Pearsons for ablation
        "per_method": {name: r["pearson"] for name, r in results_per_method.items()},
        "n_snapshots": len(snapshots),
    }


def _call_loss(model, out, **kwargs):
    """Call loss, handling torch.compile wrapper."""
    underlying = model._orig_mod if hasattr(model, "_orig_mod") else model
    return underlying.loss(out, **kwargs)

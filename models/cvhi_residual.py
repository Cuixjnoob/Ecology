"""CVHI-Residual: 强制残差分解 + 反事实必要性, 纯无监督 (无 anchor, 无 hidden 标签).

架构:
  log(x_{t+1}/x_t) = f_visible(x_t) + h_t * G(x_t)
    - f_visible: Species-GNN SoftForms (num_hidden=0, 纯 visible 动力学)
    - G:         Species-GNN SoftForms (num_hidden=0, 输出 per-species "h 敏感度场")
    - h * G:     逐元素, 保证 h=0 时 h 分支恒为 0 (硬约束, 消除 architect-away)

训练 (纯无监督, 无 anchor):
  L = MSE(pred_full, actual)
    + β · KL[q(h|x) || N(0, σ_prior²)]
    + λ_nec  · ReLU(m_null - (MSE_null - MSE_full))   # h 必要 (无 h 时预测必须变差)
    + λ_shuf · ReLU(m_shuf - (MSE_shuf - MSE_full))   # h 时序结构 (打乱时必须变差)
    + λ_energy · ReLU(min_energy - var(h))            # 防 h 塌陷成常数
    + λ_smooth · ||Δ²h||²                              # h 平滑
    + λ_sparse · (L1_gates + L1_coefs)                # 稀疏化

红线: 训练中绝不碰 hidden_true. 只在最终 eval 时用.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.cvhi_ncd import MultiLayerSpeciesGNN, MultiChannelPosteriorEncoder


class CVHI_Residual(nn.Module):
    """CVHI with forced residual decomposition + counterfactual necessity.

    n→1 任务 (k_hidden=1 固定).
    """

    def __init__(
        self,
        num_visible: int,
        # Encoder
        encoder_d: int = 64,
        encoder_blocks: int = 2,
        encoder_heads: int = 4,
        takens_lags: Tuple[int, ...] = (1, 2, 4, 8, 12),
        encoder_dropout: float = 0.15,
        # f_visible dynamics
        d_species_f: int = 24,
        f_visible_layers: int = 2,
        f_visible_top_k: int = 4,
        f_visible_use_free_nn: bool = False,
        # G sensitivity field
        d_species_G: int = 16,
        G_field_layers: int = 1,
        G_field_top_k: int = 3,
        G_field_use_free_nn: bool = False,
        # prior
        prior_std: float = 1.0,
        clamp_min: float = -2.5,
        clamp_max: float = 2.5,
        # backbone: "softforms" (5 preset forms) or "mlp" (pure MLP messages, no presets)
        gnn_backbone: str = "softforms",
    ):
        super().__init__()
        self.num_visible = num_visible
        self.prior_std = prior_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Encoder (GNN + Takens, k=1 scalar hidden)
        self.encoder = MultiChannelPosteriorEncoder(
            k_hidden=1, num_visible=num_visible,
            takens_lags=list(takens_lags),
            d_model=encoder_d, num_heads=encoder_heads,
            num_blocks=encoder_blocks, dropout=encoder_dropout,
        )

        self.gnn_backbone = gnn_backbone

        # f_visible: visible-only dynamics baseline (no h input)
        self.f_visible = MultiLayerSpeciesGNN(
            num_layers=f_visible_layers,
            backbone=gnn_backbone,
            num_visible=num_visible, num_hidden=0,
            d_species=d_species_f, top_k=f_visible_top_k,
            use_free_nn=f_visible_use_free_nn,
        )

        # G: visible-only, outputs per-species h sensitivity
        self.G_field = MultiLayerSpeciesGNN(
            num_layers=G_field_layers,
            backbone=gnn_backbone,
            num_visible=num_visible, num_hidden=0,
            d_species=d_species_G, top_k=G_field_top_k,
            use_free_nn=G_field_use_free_nn,
        )

    def compute_f_visible(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) → log_ratio (B, T, N)."""
        pred, _ = self.f_visible(x, temporal_feat=None)
        return pred

    def compute_G(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) → G (B, T, N) per-species h sensitivity."""
        G, _ = self.G_field(x, temporal_feat=None)
        return G

    def forward_rollout(self, visible: torch.Tensor, mu: torch.Tensor,
                         K: int = 3) -> Tuple[torch.Tensor, torch.Tensor]:
        """L1: K-step rollout from every valid starting point.

        visible: (B, T, N), mu: (B, T) — encoder's posterior mean (deterministic)
        Returns:
          rollout_log_states: (B, num_starts, K, N) — log of predicted x̂ at rollout step k=1..K
          target_log_states:  (B, num_starts, K, N) — log of true x_{t+k}
        """
        B, T, N = visible.shape
        num_starts = T - K
        if num_starts <= 0:
            return None, None

        log_x = torch.log(torch.clamp(visible, min=1e-6))
        # Start at each valid t: x_curr = x[:, :num_starts, :]
        x_curr = visible[:, :num_starts, :]  # (B, num_starts, N)

        rollout_logs = []
        for k in range(K):
            # Compute f_visible and G at current state (time-independent, per-step)
            base = self.compute_f_visible(x_curr)   # (B, num_starts, N)
            G_val = self.compute_G(x_curr)          # (B, num_starts, N)
            # h at step k for starting point t: mu[t+k-1] = mu[k : k+num_starts]
            # (using mu at the step BEFORE the predicted state, matching log_ratio framing)
            h_step = mu[:, k : k + num_starts]      # (B, num_starts)
            log_ratio = base + h_step.unsqueeze(-1) * G_val
            log_ratio = torch.clamp(log_ratio, self.clamp_min, self.clamp_max)
            x_next = torch.clamp(x_curr * torch.exp(log_ratio), min=1e-6)
            rollout_logs.append(torch.log(x_next))
            x_curr = x_next

        rollout_log_states = torch.stack(rollout_logs, dim=-2)  # (B, num_starts, K, N)
        target_list = [log_x[:, k + 1 : k + 1 + num_starts, :] for k in range(K)]
        target_log_states = torch.stack(target_list, dim=-2)     # (B, num_starts, K, N)

        return rollout_log_states, target_log_states

    @staticmethod
    def lowpass_gaussian(h: torch.Tensor, sigma: float) -> torch.Tensor:
        """L3: 1D Gaussian low-pass filter on time axis.

        h: (..., T). Returns h_lowpass: same shape.
        """
        # Build Gaussian kernel
        kernel_size = int(4 * sigma + 1)
        kernel_size = max(3, kernel_size | 1)  # odd
        xk = torch.arange(kernel_size, device=h.device, dtype=h.dtype)
        xk = xk - (kernel_size - 1) / 2
        gauss = torch.exp(-xk ** 2 / (2 * sigma ** 2))
        gauss = gauss / gauss.sum()
        kernel = gauss.view(1, 1, -1)
        # Reshape h to (batch, 1, T) for conv1d
        orig_shape = h.shape
        h_flat = h.reshape(-1, 1, orig_shape[-1])
        pad = kernel_size // 2
        h_padded = F.pad(h_flat, (pad, pad), mode='reflect')
        h_lp = F.conv1d(h_padded, kernel)
        return h_lp.reshape(orig_shape)

    def forward(self, visible: torch.Tensor, n_samples: int = 1,
                 rollout_K: int = 0) -> Dict[str, torch.Tensor]:
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape

        # Encoder → (mu, log_sigma) for k=1 scalar hidden
        mu_k, log_sigma_k = self.encoder(visible)  # (B, T, 1)
        mu = mu_k[..., 0]        # (B, T)
        log_sigma = log_sigma_k[..., 0]  # (B, T)

        # Sample h
        sigma = log_sigma.exp()
        eps = torch.randn(n_samples, B, T, device=visible.device)
        h_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, B, T)

        # f_visible and G (shared across samples, visible only)
        base = self.compute_f_visible(visible)  # (B, T, N)
        G = self.compute_G(visible)              # (B, T, N)

        # Full prediction: pred_full[s, b, t, n] = base[b,t,n] + h_samples[s,b,t] * G[b,t,n]
        pred_full = base.unsqueeze(0) + h_samples.unsqueeze(-1) * G.unsqueeze(0)
        # (S, B, T, N)

        # Null: no h contribution
        pred_null = base  # (B, T, N)

        # Shuffled h (permute time axis) — tests h's time-structure necessity
        perm = torch.randperm(T, device=visible.device)
        h_shuf = h_samples[:, :, perm]  # (S, B, T)
        pred_shuf = base.unsqueeze(0) + h_shuf.unsqueeze(-1) * G.unsqueeze(0)

        # Actual log_ratio (target)
        safe = torch.clamp(visible, min=1e-6)
        actual = torch.log(safe[:, 1:] / safe[:, :-1])
        actual = torch.clamp(actual, self.clamp_min, self.clamp_max)

        # Drop last time step (no target)
        pred_full = pred_full[:, :, :-1, :]  # (S, B, T-1, N)
        pred_null_trimmed = pred_null[:, :-1, :]     # (B, T-1, N)
        pred_shuf = pred_shuf[:, :, :-1, :]

        out = {
            "mu": mu,
            "log_sigma": log_sigma,
            "h_samples": h_samples,
            "pred_full": pred_full,
            "pred_null": pred_null_trimmed,
            "pred_shuf": pred_shuf,
            "actual": actual,
            "G": G,
            "base": base,
        }

        # L1: rollout (deterministic, uses mu only)
        if rollout_K > 0 and T > rollout_K + 1:
            rollout_log_states, target_log_states = self.forward_rollout(
                visible, mu, K=rollout_K
            )
            out["rollout_log_states"] = rollout_log_states
            out["target_log_states"] = target_log_states
            out["rollout_K"] = rollout_K

        return out

    def loss(
        self,
        out: Dict[str, torch.Tensor],
        beta_kl: float = 0.05,
        free_bits: float = 0.02,
        margin_null: float = 0.003,
        margin_shuf: float = 0.002,
        lam_necessary: float = 5.0,
        lam_shuffle: float = 3.0,
        lam_energy: float = 2.0,
        min_energy: float = 0.02,
        lam_smooth: float = 0.05,
        lam_sparse: float = 0.02,
        h_weight: float = 1.0,   # scales all h-related losses, 0 during warmup
        # L1: multi-step rollout
        lam_rollout: float = 0.5,
        rollout_weights: Tuple[float, ...] = (1.0, 0.5, 0.25),
        # L3: low-frequency prior
        lam_hf: float = 0.5,
        lowpass_sigma: float = 6.0,
    ) -> Dict[str, torch.Tensor]:
        mu = out["mu"]
        log_sigma = out["log_sigma"]
        pred_full = out["pred_full"]
        pred_null = out["pred_null"]
        pred_shuf = out["pred_shuf"]
        actual = out["actual"]
        h_samples = out["h_samples"]

        # Reconstructions
        recon_full = F.mse_loss(pred_full, actual.unsqueeze(0).expand_as(pred_full))
        recon_null = F.mse_loss(pred_null, actual)
        recon_shuf = F.mse_loss(pred_shuf, actual.unsqueeze(0).expand_as(pred_shuf))

        # During warmup (h_weight<1): blend recon_null (pure f_visible) with recon_full
        recon_loss = h_weight * recon_full + (1.0 - h_weight) * recon_null

        # Counterfactual margins (observed)
        margin_null_obs = recon_null - recon_full
        margin_shuf_obs = recon_shuf - recon_full

        # Counterfactual losses (active only when h_weight > 0)
        loss_necessary = F.relu(margin_null - margin_null_obs)
        loss_shuffle = F.relu(margin_shuf - margin_shuf_obs)

        # KL to prior N(0, σ_prior²) -- encoder's mu/log_sigma regularization
        prior_var = self.prior_std ** 2
        sigma_sq = torch.exp(2 * log_sigma)
        kl_per_step = 0.5 * (
            torch.log(torch.tensor(prior_var, device=mu.device)) - 2 * log_sigma
            + (sigma_sq + mu ** 2) / prior_var - 1
        )
        kl = torch.clamp(kl_per_step - free_bits, min=0).mean()

        # Energy: prevent h collapse to constant (must have variance over time)
        h_var = h_samples.var(dim=-1).mean()  # mean across samples, batches
        loss_energy = F.relu(min_energy - h_var)

        # Smoothness (second-order diff on h over time)
        dh = h_samples[:, :, 2:] - 2 * h_samples[:, :, 1:-1] + h_samples[:, :, :-2]
        loss_smooth = (dh ** 2).mean()

        # Sparsity on dynamics gates/coefs
        loss_sparse = (self.f_visible.l1_gates() + self.f_visible.l1_coefs()
                       + self.G_field.l1_gates() + self.G_field.l1_coefs())

        # L1: Multi-step rollout (if forward was called with rollout_K > 0)
        if "rollout_log_states" in out:
            rls = out["rollout_log_states"]   # (B, num_starts, K, N)
            tls = out["target_log_states"]
            K_out = rls.shape[-2]
            # per-step MSE
            step_mse = ((rls - tls) ** 2).mean(dim=(0, 1, 3))  # (K,)
            # weights (decreasing)
            w = torch.tensor(list(rollout_weights)[:K_out],
                              device=rls.device, dtype=rls.dtype)
            w = w / w.sum()
            loss_rollout = (step_mse * w).sum()
        else:
            loss_rollout = torch.tensor(0.0, device=mu.device)
            step_mse = None

        # L3: Low-frequency prior on h (penalize high-frequency energy)
        h_lp = self.lowpass_gaussian(h_samples, sigma=lowpass_sigma)  # (S, B, T)
        h_hf = h_samples - h_lp
        loss_hf = (h_hf ** 2).mean()
        # Diagnostic: fraction of h-variance from high-frequency
        with torch.no_grad():
            hf_var = h_hf.var(dim=-1).mean()
            total_var = h_samples.var(dim=-1).mean() + 1e-8
            hf_frac = hf_var / total_var

        total = (recon_loss
                 + beta_kl * kl
                 + h_weight * lam_necessary * loss_necessary
                 + h_weight * lam_shuffle * loss_shuffle
                 + lam_energy * loss_energy
                 + lam_smooth * loss_smooth
                 + lam_sparse * loss_sparse
                 + h_weight * lam_rollout * loss_rollout   # L1
                 + h_weight * lam_hf * loss_hf)             # L3

        return {
            "total": total,
            "recon_full": recon_full,
            "recon_null": recon_null,
            "recon_shuf": recon_shuf,
            "kl": kl,
            "necessary": loss_necessary,
            "shuffle": loss_shuffle,
            "energy": loss_energy,
            "smooth": loss_smooth,
            "sparse": loss_sparse,
            "margin_null_obs": margin_null_obs.detach(),
            "margin_shuf_obs": margin_shuf_obs.detach(),
            "h_var": h_var.detach(),
            "sigma_mean": log_sigma.exp().mean().detach(),
            # L1 / L3 diagnostics
            "rollout": loss_rollout.detach() if torch.is_tensor(loss_rollout) else torch.tensor(0.0),
            "rollout_per_step": step_mse.detach() if step_mse is not None else None,
            "hf": loss_hf.detach(),
            "hf_frac": hf_frac.detach(),
        }

    def slice_out(self, out: Dict[str, torch.Tensor], t_start: int, t_end: int) -> Dict[str, torch.Tensor]:
        """Return a sub-dict sliced to [t_start, t_end] — for train/val split loss."""
        p_end = t_end - 1
        p_start = max(0, t_start - 1) if t_start > 0 else 0
        sliced = {
            "mu": out["mu"][:, t_start:t_end],
            "log_sigma": out["log_sigma"][:, t_start:t_end],
            "h_samples": out["h_samples"][:, :, t_start:t_end],
            "pred_full": out["pred_full"][:, :, p_start:p_end, :],
            "pred_null": out["pred_null"][:, p_start:p_end, :],
            "pred_shuf": out["pred_shuf"][:, :, p_start:p_end, :],
            "actual": out["actual"][:, p_start:p_end, :],
            "G": out["G"],
            "base": out["base"],
        }
        # Slice rollout fields if present (num_starts is T-K, slice to train segment)
        if "rollout_log_states" in out:
            rls = out["rollout_log_states"]  # (B, T-K, K, N)
            tls = out["target_log_states"]
            # rollout starting points t in [0, T-K); we want those in [t_start, t_end-K]
            K_r = rls.shape[-2]
            r_end = max(0, t_end - K_r)
            r_start = t_start
            if r_end > r_start:
                sliced["rollout_log_states"] = rls[:, r_start:r_end, :, :]
                sliced["target_log_states"] = tls[:, r_start:r_end, :, :]
                sliced["rollout_K"] = K_r
        return sliced

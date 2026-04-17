"""Eco-GNRD (Ecological Graph Neural Residual Dynamics).

A fully unsupervised framework for inferring the dynamical influence of
unobserved species in partially-observed ecological communities.

Architecture:
  log(x_{t+1}/x_t)_i = f_visible_i(x_t) + h(t) * G_i(x_t)

  - f_visible: Species-GNN modeling observable community dynamics
  - G:         Species-GNN outputting per-species sensitivity to hidden influence
  - h(t):     Scalar latent variable representing the hidden species' effect
  - h * G:    Element-wise product ensuring zero contribution when h=0

Training objective (strictly unsupervised -- hidden_true never used):
  L = MSE(pred_full, actual)                              # visible reconstruction
    + beta * KL[q(h|x) || N(0, sigma_prior^2)]            # posterior regularization
    + lam_nec  * ReLU(m_null - (MSE_null - MSE_full))     # counterfactual necessity
    + lam_shuf * ReLU(m_shuf - (MSE_shuf - MSE_full))     # temporal structure
    + lam_energy * ReLU(min_energy - var(h))               # anti-collapse
    + lam_smooth * ||d^2 h / dt^2||^2                     # smoothness
    + lam_sparse * (L1_gates + L1_coefs)                   # interaction sparsity

Reference:
  "Inferring Unobserved Species Influence in Chaotic Ecological
   Dynamics: A Data-Driven Unsupervised Approach"
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.cvhi_ncd import MultiLayerSpeciesGNN, MultiChannelPosteriorEncoder


class EcoGNRD(nn.Module):
    """Eco-GNRD: Graph Neural Residual Dynamics for hidden species inference.

    Decomposes ecological dynamics into an observable baseline f_visible(x)
    and a latent residual closure h(t)*G(x), where h(t) is inferred by a
    variational encoder without any supervision from the hidden species.

    Designed for the n-to-1 partial observation setting (single hidden species).
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
        # ablation flags
        use_formula_hints: bool = True,   # 控制 MLP backbone 是否使用 formula hints 输入
        use_G_field: bool = True,         # 控制残差分解是否使用 G(x) 敏感度场 (False: h 直接均匀加, 不学 G)
        # event-focused loss reweighting (Exp1, 2026-04-14)
        use_event_weighting: bool = False,  # 打开后 recon loss 每个 t 乘上 w_t ∝ ||Δlog x_t||^α
        event_alpha: float = 1.0,           # α 越大越强调事件时刻 (0 = 均匀)
        # MoG posterior (Exp2, 2026-04-14)
        num_mixture_components: int = 1,    # K 个高斯分量 (K=1 等价原版); 通过 Gumbel-softmax 采样
        gumbel_tau: float = 1.0,            # Gumbel-softmax 温度
        # Path A: G(x) ≥ 0 约束, 破 ±h 对称 (2026-04-14)
        G_positive: bool = False,           # True: 用 softplus 强制 *全部* G ≥ 0, 消除 sign ambiguity
        # Path A'': 只 pin 一个物种 (idx=0) 的 G ≥ 0, 其它自由  — 更温和破对称
        G_anchor_first: bool = False,
        # Path A''': sign convention of anchored species. +1: pin ≥ 0, -1: pin ≤ 0.
        # 配合 "双向训练 + val 选方向" 消除 convention 偏见.
        G_anchor_sign: int = +1,
        # Option 3: encoder 的 PCA 耦合权重 (per-(t,j) dynamic attention feature)
        use_coupling_weight: bool = False,
        coupling_top_k: int = 3,
        # Option 3 B+C: 用 log(w) 作为 attention bias (species & temporal)
        use_coupling_attn: bool = False,
        # Part B: residual-driven attention (R = Δlog x - f_visible(x))
        use_residual_attn: bool = False,
        # Path B multi-hidden: 总 species 数 (encoder 用 ID 索引)
        num_total_species: int = None,
        # Hierarchical h: 2 个 channel (slow + fast) 求和作最终 h
        hierarchical_h: bool = False,
        # Point estimate mode: encoder outputs h directly, no sampling, no KL
        point_estimate: bool = False,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.prior_std = prior_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.point_estimate = point_estimate
        self.use_event_weighting = use_event_weighting
        self.event_alpha = event_alpha
        self.K = num_mixture_components
        self.gumbel_tau = gumbel_tau
        self.G_positive = G_positive
        self.G_anchor_first = G_anchor_first
        self.G_anchor_sign = int(G_anchor_sign)
        assert self.G_anchor_sign in (-1, +1), f"G_anchor_sign must be ±1, got {G_anchor_sign}"
        # annealing schedule: 1.0 = hard softplus (原 G_anchor_first), 0.0 = identity (无约束).
        # 训练循环通过 model.G_anchor_alpha = ... 逐 epoch 更新
        self.G_anchor_alpha = 1.0

        self.use_residual_attn = use_residual_attn
        self.hierarchical_h = hierarchical_h

        # Encoder: k_hidden=2 if hierarchical (one slow + one fast channel)
        k_hidden = 2 if hierarchical_h else 1
        self.encoder = MultiChannelPosteriorEncoder(
            k_hidden=k_hidden, num_visible=num_visible,
            takens_lags=list(takens_lags),
            d_model=encoder_d, num_heads=encoder_heads,
            num_blocks=encoder_blocks, dropout=encoder_dropout,
            num_mixture_components=num_mixture_components,
            use_coupling_weight=use_coupling_weight,
            coupling_top_k=coupling_top_k,
            use_coupling_attn=use_coupling_attn,
            use_residual_attn=use_residual_attn,
            num_total_species=num_total_species,
        )

        self.gnn_backbone = gnn_backbone
        self.use_G_field = use_G_field

        # ablation: formula hints 仅对 mlp backbone 生效
        extra_kw = {}
        if gnn_backbone == "mlp":
            extra_kw["use_formula_hints"] = use_formula_hints

        # f_visible: visible-only dynamics baseline (no h input)
        self.f_visible = MultiLayerSpeciesGNN(
            num_layers=f_visible_layers,
            backbone=gnn_backbone,
            num_visible=num_visible, num_hidden=0,
            d_species=d_species_f, top_k=f_visible_top_k,
            use_free_nn=f_visible_use_free_nn,
            **extra_kw,
        )

        # G: visible-only, outputs per-species h sensitivity (仅在 use_G_field=True 时构建)
        if use_G_field:
            self.G_field = MultiLayerSpeciesGNN(
                num_layers=G_field_layers,
                backbone=gnn_backbone,
                num_visible=num_visible, num_hidden=0,
                d_species=d_species_G, top_k=G_field_top_k,
                use_free_nn=G_field_use_free_nn,
                **extra_kw,
            )
        else:
            self.G_field = None

    def compute_f_visible(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) → log_ratio (B, T, N)."""
        pred, _ = self.f_visible(x, temporal_feat=None)
        return pred

    def compute_G(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, N) → G (B, T, N) per-species h sensitivity.

        Ablation: 当 use_G_field=False 时, 返回 ones (h 均匀加到所有物种, 无敏感度差异).
        Path A: 当 G_positive=True 时, 对 *全部* G 输出做 softplus, 保证 G ≥ 0.
        Path A'': 当 G_anchor_first=True 时, **只** 对 G[..., 0] 做 softplus (pin 参考物种),
          其它物种 G 完全自由 — 仅破 sign 对称, 保留表达力.
        """
        if not self.use_G_field:
            return torch.ones(x.shape[0], x.shape[1], x.shape[2], device=x.device)
        G, _ = self.G_field(x, temporal_feat=None)
        if self.G_positive:
            G = F.softplus(G)
        elif self.G_anchor_first:
            # 软约束插值: alpha*(sign·softplus) + (1-alpha)*identity
            raw_anchor = G[..., :1]
            alpha = float(self.G_anchor_alpha)
            sign = float(self.G_anchor_sign)
            if alpha >= 1.0:
                G_anchor = sign * F.softplus(raw_anchor)
            elif alpha <= 0.0:
                G_anchor = raw_anchor
            else:
                G_anchor = alpha * (sign * F.softplus(raw_anchor)) + (1 - alpha) * raw_anchor
            G_rest = G[..., 1:]
            G = torch.cat([G_anchor, G_rest], dim=-1)
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
                 rollout_K: int = 0, species_ids: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape

        # Compute residual R = Δlog x - f_visible(x) if needed for attention
        residual = None
        if self.use_residual_attn:
            with torch.no_grad():
                base_for_R = self.compute_f_visible(visible).detach()  # (B, T, N)
                safe = torch.clamp(visible, min=1e-6)
                dx = torch.log(safe[:, 1:] / safe[:, :-1])   # (B, T-1, N)
                dx_pad = torch.cat([dx, dx[:, -1:]], dim=1)   # (B, T, N)
                residual = (dx_pad - base_for_R).detach()     # (B, T, N)

        logits = None   # set below when K>1
        if self.K == 1:
            # Encoder → (mu, log_sigma) for k_hidden=1 or 2
            mu_k, log_sigma_k = self.encoder(visible, residual=residual, species_ids=species_ids)
            # mu_k shape: (B, T, k_hidden)

            if self.hierarchical_h:
                # k_hidden=2: channel 0 = slow, channel 1 = fast
                mu_slow = mu_k[..., 0]           # (B, T)
                mu_fast = mu_k[..., 1]
                log_sigma_slow = log_sigma_k[..., 0]
                log_sigma_fast = log_sigma_k[..., 1]

                sigma_slow = log_sigma_slow.exp()
                sigma_fast = log_sigma_fast.exp()

                # Independent samples per channel
                eps1 = torch.randn(n_samples, B, T, device=visible.device)
                eps2 = torch.randn(n_samples, B, T, device=visible.device)
                h_slow = mu_slow.unsqueeze(0) + sigma_slow.unsqueeze(0) * eps1
                h_fast = mu_fast.unsqueeze(0) + sigma_fast.unsqueeze(0) * eps2

                # Sum = final h
                h_samples = h_slow + h_fast    # (S, B, T)

                # For loss: expose combined mu, log_sigma (use sum mu, combined variance)
                mu = mu_slow + mu_fast          # (B, T)
                # Combined log_sigma (independent Gaussians sum → variance adds)
                sigma_combined_sq = sigma_slow**2 + sigma_fast**2
                log_sigma = 0.5 * torch.log(sigma_combined_sq + 1e-10)
                # Keep per-channel for separate smooth prior
                _mu_slow, _mu_fast = mu_slow, mu_fast
            else:
                mu = mu_k[..., 0]        # (B, T)
                log_sigma = log_sigma_k[..., 0]  # (B, T)
                _mu_slow, _mu_fast = None, None

                if self.point_estimate:
                    # Point estimate: h = mu directly, no sampling noise
                    h_samples = mu.unsqueeze(0).expand(n_samples, B, T)  # (S, B, T)
                    log_sigma = torch.zeros_like(log_sigma)  # dummy for loss compatibility
                else:
                    # Sample h (single Gaussian)
                    sigma = log_sigma.exp()
                    eps = torch.randn(n_samples, B, T, device=visible.device)
                    h_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, B, T)
        else:
            # MoG posterior: encoder returns (mu, log_sigma, logits) each (B, T, K, 1)
            mu_k, log_sigma_k, logits_k = self.encoder(visible, residual=residual, species_ids=species_ids)
            mu = mu_k[..., 0]            # (B, T, K)
            log_sigma = log_sigma_k[..., 0]
            logits = logits_k[..., 0]    # (B, T, K)
            sigma = log_sigma.exp()

            # Gumbel-softmax straight-through, per-sample per (b, t):
            #   sample y (n_samples, B, T, K), one-hot in forward, soft gradient
            S = n_samples
            gumbel = -torch.log(-torch.log(
                torch.rand(S, B, T, self.K, device=visible.device) + 1e-20) + 1e-20)
            logits_exp = logits.unsqueeze(0).expand(S, B, T, self.K)
            y_soft = F.softmax((logits_exp + gumbel) / self.gumbel_tau, dim=-1)
            # straight-through: hard onehot forward, soft gradient backward
            idx = y_soft.argmax(dim=-1, keepdim=True)
            y_hard = torch.zeros_like(y_soft).scatter_(-1, idx, 1.0)
            y = (y_hard - y_soft).detach() + y_soft                    # (S, B, T, K)

            # Gaussian reparameterize per-component, then pick by y
            eps = torch.randn(S, B, T, self.K, device=visible.device)
            z_comp = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps        # (S, B, T, K)
            h_samples = (z_comp * y).sum(-1)                            # (S, B, T)

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
            "visible": visible,      # Stage 1: 给 loss 算 log(x) 重构
        }
        # Hierarchical mu slow/fast for separate smoothness priors
        if self.hierarchical_h and _mu_slow is not None:
            out["mu_slow"] = _mu_slow
            out["mu_fast"] = _mu_fast
        if logits is not None:
            out["logits"] = logits   # (B, T, K) MoG mixture logits

        # For rollout we need a scalar mu per (b, t). With K>1 pick π-weighted mean.
        if self.K == 1:
            mu_for_rollout = mu
        else:
            pi = F.softmax(logits, dim=-1)                          # (B, T, K)
            mu_for_rollout = (pi * mu).sum(-1)                      # (B, T)

        # L1: rollout (deterministic, uses mu only)
        if rollout_K > 0 and T > rollout_K + 1:
            rollout_log_states, target_log_states = self.forward_rollout(
                visible, mu_for_rollout, K=rollout_K
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
        # MoG entropy regularizer on π (only active when K>1)
        lam_entropy: float = 0.1,
        # Stage 1 (2026-04-14): ecology priors
        lam_rmse_log: float = 0.0,         # log(x) 重构损失 (amplitude 感知)
        lam_mte_prior: float = 0.0,        # [DEPRECATED] MTE-based G magnitude prior (wrong target)
        mte_prior_target: torch.Tensor = None,  # (N,) target |G| magnitudes from body mass
        # Stage 1c (2026-04-15): corrected MTE — shape-only prior on f_visible intrinsic rate
        lam_mte_shape: float = 0.0,        # soft correlation prior, only shape not absolute
        mte_target_log_r: torch.Tensor = None,   # (N,) target log intrinsic rate ~ (b-1)*log10(M)
        # Stage 2 (2026-04-15): Klausmeier stoichiometric sign priors
        lam_stoich_sign: float = 0.0,
        # Robust recon: Huber loss for bursty data (Beninca etc.)
        use_huber_recon: bool = False,
        huber_delta: float = 0.1,
        # Nutrients-as-input: only compute recon loss on first n channels (species)
        n_recon_channels: int = None,   # None = all channels
        stoich_pos_pairs: tuple = (),  # list of (i_target, j_source) where d base_i / d x_j should be > 0
        stoich_neg_pairs: tuple = (),  # d base_i / d x_j should be < 0
    ) -> Dict[str, torch.Tensor]:
        mu = out["mu"]
        log_sigma = out["log_sigma"]
        pred_full = out["pred_full"]
        pred_null = out["pred_null"]
        pred_shuf = out["pred_shuf"]
        actual = out["actual"]
        h_samples = out["h_samples"]
        logits = out.get("logits", None)   # (B, T, K) when K>1, else None

        # Nutrients-as-input: slice to species-only channels for recon loss
        nc = n_recon_channels
        if nc is not None and nc < actual.shape[-1]:
            pred_full = pred_full[..., :nc]
            pred_null = pred_null[..., :nc]
            pred_shuf = pred_shuf[..., :nc]
            actual = actual[..., :nc]

        # Reconstructions — optionally reweight per timestep by event magnitude
        if self.use_event_weighting:
            # Event salience: L2 norm of actual log-ratio per (b, t) -> (B, T_slice)
            with torch.no_grad():
                mag = actual.pow(2).sum(-1).sqrt()            # (B, T)
                w = (mag + 1e-6) ** self.event_alpha
                w = w / w.mean(dim=-1, keepdim=True).clamp(min=1e-8)   # mean=1 per batch
                w = w.detach()
            # weighted MSE helpers
            def _weighted_mse(pred, tgt):
                # pred: (..., T, N);  tgt: (..., T, N);  w: (B, T)
                sq = (pred - tgt) ** 2                         # (..., T, N)
                per_t = sq.mean(dim=-1)                         # (..., T)
                return (per_t * w).mean()
            recon_full = _weighted_mse(pred_full, actual.unsqueeze(0).expand_as(pred_full))
            recon_null = _weighted_mse(pred_null, actual)
            recon_shuf = _weighted_mse(pred_shuf, actual.unsqueeze(0).expand_as(pred_shuf))
        else:
            if use_huber_recon:
                recon_full = F.huber_loss(pred_full, actual.unsqueeze(0).expand_as(pred_full),
                                            delta=huber_delta)
                recon_null = F.huber_loss(pred_null, actual, delta=huber_delta)
                recon_shuf = F.huber_loss(pred_shuf, actual.unsqueeze(0).expand_as(pred_shuf),
                                            delta=huber_delta)
            else:
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

        # KL to prior N(0, σ_prior²)
        prior_var = self.prior_std ** 2
        if logits is None:
            # 原版解析 KL (K=1)
            sigma_sq = torch.exp(2 * log_sigma)
            kl_per_step = 0.5 * (
                torch.log(torch.tensor(prior_var, device=mu.device)) - 2 * log_sigma
                + (sigma_sq + mu ** 2) / prior_var - 1
            )
            kl = torch.clamp(kl_per_step - free_bits, min=0).mean()
            loss_entropy_reg = torch.tensor(0.0, device=mu.device)
            pi_entropy = torch.tensor(0.0, device=mu.device)
        else:
            # MoG MC KL: 对每个 h_samples s, 计算 log q(h|x) - log p(h) 再平均
            # q(h|x) = Σ_k π_k N(h; μ_k, σ_k²)
            sigma = log_sigma.exp()                                 # (B, T, K)
            log_pi = F.log_softmax(logits, dim=-1)                   # (B, T, K)
            # h_samples (S, B, T); broadcast across K
            h = h_samples.unsqueeze(-1)                              # (S, B, T, 1)
            mu_b    = mu.unsqueeze(0)                                # (1, B, T, K)
            ls_b    = log_sigma.unsqueeze(0)                         # (1, B, T, K)
            sig_b   = sigma.unsqueeze(0)                             # (1, B, T, K)
            # log N(h; μ_k, σ_k²)  (S, B, T, K)
            log_N_k = -0.5 * (
                torch.log(torch.tensor(2 * torch.pi, device=mu.device))
                + 2 * ls_b + ((h - mu_b) / sig_b) ** 2
            )
            log_q = torch.logsumexp(log_pi.unsqueeze(0) + log_N_k, dim=-1)   # (S, B, T)
            # log p(h) = log N(h; 0, σ_prior²)
            log_p = -0.5 * (
                torch.log(torch.tensor(2 * torch.pi * prior_var, device=mu.device))
                + h_samples ** 2 / prior_var
            )                                                        # (S, B, T)
            kl_per_step_sample = log_q - log_p                       # (S, B, T)
            # 等效的 per-step KL 估计: 在样本维度做平均先
            kl_per_step = kl_per_step_sample.mean(dim=0)             # (B, T)
            kl = torch.clamp(kl_per_step - free_bits, min=0).mean()
            # Entropy reg on π (反 collapse): 最大化 H(π) → 减 loss
            pi = log_pi.exp()
            pi_entropy = -(pi * log_pi).sum(-1).mean()               # scalar
            loss_entropy_reg = -pi_entropy                            # 负的, 乘 lam_entropy 做减

        # Energy: prevent h collapse to constant (must have variance over time)
        h_var = h_samples.var(dim=-1).mean()  # mean across samples, batches
        loss_energy = F.relu(min_energy - h_var)

        # Smoothness (second-order diff on h over time)
        dh = h_samples[:, :, 2:] - 2 * h_samples[:, :, 1:-1] + h_samples[:, :, :-2]
        loss_smooth = (dh ** 2).mean()

        # Sparsity on dynamics gates/coefs (G_field 可能被 ablation 关掉)
        loss_sparse = self.f_visible.l1_gates() + self.f_visible.l1_coefs()
        if self.G_field is not None:
            loss_sparse = loss_sparse + self.G_field.l1_gates() + self.G_field.l1_coefs()

        # L1: Multi-step rollout (if forward was called with rollout_K > 0)
        if "rollout_log_states" in out:
            rls = out["rollout_log_states"]   # (B, num_starts, K, N)
            tls = out["target_log_states"]
            if nc is not None and nc < rls.shape[-1]:
                rls = rls[..., :nc]
                tls = tls[..., :nc]
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

        # Stage 1: log(x) reconstruction (amplitude-aware)
        if lam_rmse_log > 0.0 and "visible" in out:
            vis = out["visible"]                                 # (B, T, N_all)
            if nc is not None and nc < vis.shape[-1]:
                vis = vis[..., :nc]
            safe_vis = torch.clamp(vis, min=1e-6)
            log_x_actual = torch.log(safe_vis)[:, 1:, :]         # (B, T-1, N)
            # pred log(x_{t+1}) = log(x_t) + Δlog_pred = log(x_t) + pred_full[s,b,t,n]
            log_x_prev = torch.log(safe_vis)[:, :-1, :]          # (B, T-1, N)
            pred_log_x = log_x_prev.unsqueeze(0) + pred_full     # (S, B, T-1, N)
            loss_rmse_log = ((pred_log_x - log_x_actual.unsqueeze(0)) ** 2).mean()
        else:
            loss_rmse_log = torch.tensor(0.0, device=mu.device)

        # Hierarchical h: 额外 smoothness 给 slow channel
        loss_hier_smooth = torch.tensor(0.0, device=mu.device)
        if "mu_slow" in out and "mu_fast" in out:
            mu_slow = out["mu_slow"]
            mu_fast = out["mu_fast"]
            # Slow channel: 强 smoothness (second-order diff)
            d2_slow = mu_slow[:, 2:] - 2 * mu_slow[:, 1:-1] + mu_slow[:, :-2]
            loss_hier_smooth = (d2_slow ** 2).mean()
            # Fast channel: 弱 smoothness (放任高频)
            # 可选: 也可以加 ||mu_fast||² 让 fast 小

        # Stage 1: MTE-based G magnitude prior [DEPRECATED — applied to wrong target]
        if lam_mte_prior > 0.0 and mte_prior_target is not None and "G" in out:
            G_field = out["G"]                                   # (B, T, N)
            G_mag = G_field.abs().mean(dim=(0, 1))               # (N,)
            G_mag_norm = G_mag / (G_mag.sum() + 1e-8)
            mte_norm = mte_prior_target / (mte_prior_target.sum() + 1e-8)
            loss_mte = ((G_mag_norm - mte_norm) ** 2).sum()
        else:
            loss_mte = torch.tensor(0.0, device=mu.device)

        # Stage 1c (2026-04-15): corrected MTE shape prior on f_visible ONLY
        # Per Clarke 2025: MTE constrains intrinsic-growth shape/exponent, not absolute rate.
        # Per Glazier 2005 / Kremer 2017: pelagic phyto + zooplankton have b≈0.88-0.95, weaker scaling.
        # Apply as Pearson-correlation distance: 1 - corr(log|base|_i, target_log_r_i).
        # Only direction is constrained — absolute rate stays free.
        if (lam_mte_shape > 0.0 and mte_target_log_r is not None
                and "base" in out):
            base = out["base"]                                   # (B, T, N)
            base_mag = base.abs().mean(dim=(0, 1))               # (N,)
            log_base_mag = torch.log(base_mag + 1e-6)
            target = mte_target_log_r.to(base.device)            # (N,) — NaN entries are skipped
            mask = ~torch.isnan(target)
            if mask.sum().item() >= 2:
                x_m = log_base_mag[mask]
                y_m = target[mask]
                x_c = x_m - x_m.mean()
                y_c = y_m - y_m.mean()
                denom = torch.sqrt((x_c ** 2).mean() * (y_c ** 2).mean() + 1e-10)
                corr = (x_c * y_c).mean() / denom
                loss_mte_shape = (1.0 - corr).clamp(min=0.0)
            else:
                loss_mte_shape = torch.tensor(0.0, device=mu.device)
        else:
            loss_mte_shape = torch.tensor(0.0, device=mu.device)

        # Stage 2 (2026-04-15): Klausmeier sign prior on f_visible partial derivatives
        # 使用 finite-difference on x: ε-扰 one species j, 看 base[i] 变化方向.
        # soft sign: penalize ReLU(-dbase_i/dx_j) for positive pairs, ReLU(dbase_i/dx_j) for negative pairs
        if lam_stoich_sign > 0.0 and "base" in out and "visible" in out \
                and (len(stoich_pos_pairs) > 0 or len(stoich_neg_pairs) > 0):
            vis = out["visible"]                                 # (B, T_vis, N)
            base_cur = out["base"]                               # (B, T_base, N)
            T_sync = min(vis.shape[1], base_cur.shape[1])
            vis = vis[:, :T_sync]
            base_cur = base_cur[:, :T_sync]
            # 效率: 56 pairs × 500 epochs 全算太慢. 每次 loss 只对 **随机采样的 1 个 j_src**
            # 做一次 perturbed forward, 覆盖所有共享该 j_src 的 pairs. 500 epochs 自然轮询所有 j_src.
            # 每 loss 调用只 +1 次 compute_f_visible, overhead 从 +56x 降到 +1x.
            unique_sources = list({j for (_, j) in stoich_pos_pairs} |
                                   {j for (_, j) in stoich_neg_pairs})
            j_src_sample = unique_sources[torch.randint(len(unique_sources), (1,)).item()]
            vis_p = vis.clone()
            vis_p[..., j_src_sample] = vis_p[..., j_src_sample] * (1.0 + 0.05) + 1e-8
            base_p = self.compute_f_visible(vis_p)
            loss_stoich = torch.zeros((), device=mu.device)
            n_active = 0
            for (i_tgt, j_src) in stoich_pos_pairs:
                if j_src == j_src_sample:
                    dbase = base_p[..., i_tgt] - base_cur[..., i_tgt]
                    loss_stoich = loss_stoich + F.relu(-dbase).mean()
                    n_active += 1
            for (i_tgt, j_src) in stoich_neg_pairs:
                if j_src == j_src_sample:
                    dbase = base_p[..., i_tgt] - base_cur[..., i_tgt]
                    loss_stoich = loss_stoich + F.relu(dbase).mean()
                    n_active += 1
            if n_active > 0:
                loss_stoich = loss_stoich / n_active
        else:
            loss_stoich = torch.tensor(0.0, device=mu.device)

        total = (recon_loss
                 + beta_kl * kl
                 + h_weight * lam_necessary * loss_necessary
                 + h_weight * lam_shuffle * loss_shuffle
                 + lam_energy * loss_energy
                 + lam_smooth * loss_smooth
                 + lam_sparse * loss_sparse
                 + h_weight * lam_rollout * loss_rollout   # L1
                 + h_weight * lam_hf * loss_hf              # L3
                 + lam_entropy * loss_entropy_reg           # MoG anti-collapse
                 + lam_rmse_log * loss_rmse_log             # Stage 1: amplitude
                 + lam_mte_prior * loss_mte                  # Stage 1 [DEPRECATED]
                 + lam_mte_shape * loss_mte_shape            # Stage 1c: MTE shape on f_visible
                 + lam_stoich_sign * loss_stoich             # Stage 2: Klausmeier sign prior
                 + 5.0 * lam_smooth * loss_hier_smooth)       # Hierarchical: slow-channel 强 smooth

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
            # MoG diagnostics
            "pi_entropy": pi_entropy.detach() if torch.is_tensor(pi_entropy) else torch.tensor(0.0),
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
        if "logits" in out:
            sliced["logits"] = out["logits"][:, t_start:t_end]
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


# Backward-compatible alias so existing imports still work
CVHI_Residual = EcoGNRD

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
    ):
        super().__init__()
        self.num_visible = num_visible
        self.prior_std = prior_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
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

        # Encoder (GNN + Takens, k=1 scalar hidden, K 个分量)
        self.encoder = MultiChannelPosteriorEncoder(
            k_hidden=1, num_visible=num_visible,
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
            # Encoder → (mu, log_sigma) for k=1 scalar hidden
            mu_k, log_sigma_k = self.encoder(visible, residual=residual, species_ids=species_ids)
            mu = mu_k[..., 0]        # (B, T)
            log_sigma = log_sigma_k[..., 0]  # (B, T)

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
        }
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
    ) -> Dict[str, torch.Tensor]:
        mu = out["mu"]
        log_sigma = out["log_sigma"]
        pred_full = out["pred_full"]
        pred_null = out["pred_null"]
        pred_shuf = out["pred_shuf"]
        actual = out["actual"]
        h_samples = out["h_samples"]
        logits = out.get("logits", None)   # (B, T, K) when K>1, else None

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
                 + h_weight * lam_hf * loss_hf              # L3
                 + lam_entropy * loss_entropy_reg)           # MoG anti-collapse

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

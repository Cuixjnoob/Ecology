"""Conditional Variational Hidden Inference (CVHI) MVP.

核心思想:
  1. Posterior Encoder q(h|x_visible): GNN + Takens → (μ, log σ) per time step
  2. Data-Driven Dynamics Operator: Sparse GAT, 无预设公式
     软约束: L1 on attention weights, Lipschitz penalty
  3. ELBO Training:
     L = visible_reconstruction + β·KL[q(h|x)‖p(h)] + λ·sparsity + γ·lipschitz
     β warm-up 防 posterior collapse
  4. Multi-hypothesis: 多次采样 → 多个 plausible hidden

严格无 hidden 监督。Hidden_true 只在评估时使用。
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_pe(length, d_model, device):
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =============================================================================
# Module 1: Posterior Encoder q(h|x_visible)
# =============================================================================
class PosteriorEncoder(nn.Module):
    """GNN-based encoder for q(h|x_visible).

    Input: visible_states (B, T, N)
    Output: μ(B,T), log_σ(B,T) — per-time-step Gaussian parameters
    """
    def __init__(
        self,
        num_visible: int = 5,
        takens_lags: List[int] = (1, 2, 4, 8),
        d_model: int = 96,
        num_heads: int = 4,
        num_blocks: int = 3,
        dropout: float = 0.1,
        num_mixture_components: int = 1,   # K=1 (单高斯, 原版); K>1 = MoG posterior
        use_coupling_weight: bool = False, # True: 添加 PCA 耦合 per-(t,j) 权重为额外特征
        coupling_top_k: int = 3,            # top-K PC 作为"集体模式"
        use_coupling_attn: bool = False,   # True: log(w) 作为 species+temporal attn 加性 bias (路 B+C)
        use_residual_attn: bool = False,   # True: log|R| 作为 attn bias (R=Δlog x - f_visible(x))
        # Path B: multi-hidden joint training — species_emb 按 ID 索引, 允许不同 hidden choice 共享 encoder
        num_total_species: int = None,     # 若 None 则与 num_visible 相等 (原版行为)
    ):
        super().__init__()
        self.num_visible = num_visible
        self.takens_lags = list(takens_lags)
        self.d_model = d_model
        self.num_heads = num_heads
        self.K = num_mixture_components
        self.use_coupling_weight = use_coupling_weight
        self.coupling_top_k = coupling_top_k
        self.use_coupling_attn = use_coupling_attn
        self.use_residual_attn = use_residual_attn
        self.num_total_species = num_total_species if num_total_species is not None else num_visible
        self.use_species_id_emb = (num_total_species is not None and num_total_species != num_visible)

        # Feature dim per (t, n): [x, log_x, Takens x, Takens log_x] (+ coupling_weight if enabled)
        feat_dim = 2 + 2 * len(self.takens_lags) + (1 if use_coupling_weight else 0)

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        # Path B: 若开启 species ID embedding, emb 按 total-species ID 索引 (forward 时 gather)
        # 否则按 position 索引 (原版)
        if self.use_species_id_emb:
            self.species_emb_table = nn.Parameter(torch.randn(self.num_total_species, d_model) * 0.1)
            self.species_emb = None
        else:
            self.species_emb = nn.Parameter(torch.randn(num_visible, d_model) * 0.1)
            self.species_emb_table = None

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                "species_attn": nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
                "norm1": nn.LayerNorm(d_model),
                "temporal_attn": nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True),
                "norm2": nn.LayerNorm(d_model),
                "ffn": nn.Sequential(
                    nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
                    nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
                ),
                "norm3": nn.LayerNorm(d_model),
            })
            for _ in range(num_blocks)
        ])

        # Readout to (μ, log σ) per time step. K=1 = scalar Gaussian (原版);
        # K>1 = 混合 K 个分量, 每步输出 (μ_k, log σ_k, logit_k).
        self.readout_mu = nn.Sequential(
            nn.Linear(d_model * num_visible, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.K),
        )
        self.readout_logsigma = nn.Sequential(
            nn.Linear(d_model * num_visible, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, self.K),
        )
        if self.K > 1:
            self.readout_logits = nn.Sequential(
                nn.Linear(d_model * num_visible, d_model),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(d_model, self.K),
            )
        else:
            self.readout_logits = None

    def _compute_coupling_weight(self, x: torch.Tensor) -> torch.Tensor:
        """PCA 耦合权重: w_{t,j} = 集体成分² / 原 dx², ∈ [0, 1].

        动机: w 大 = 这个 (t,j) 变化被群落 top-K PCs 解释 = 真事件;
              w 小 = 孤立跳动, 大概率噪声.
        x: (B, T, N) → w: (B, T, N)
        """
        with torch.no_grad():
            B, T, N = x.shape
            eps = 1e-6
            safe = torch.clamp(x, min=eps)
            dx = torch.log(safe[:, 1:] / safe[:, :-1])   # (B, T-1, N)
            # 尾部复制最后一个时间点到 T 长度 (avoid shape mismatch)
            dx_padded = torch.cat([dx, dx[:, -1:]], dim=1)  # (B, T, N)
            # batch SVD
            U, S, Vh = torch.linalg.svd(dx_padded, full_matrices=False)   # (B, T, r), (B, r), (B, r, N)
            K = min(self.coupling_top_k, S.shape[-1])
            S_k = S[..., :K]                                 # (B, K)
            U_k = U[..., :K]                                 # (B, T, K)
            Vh_k = Vh[..., :K, :]                             # (B, K, N)
            dx_coll = U_k @ torch.diag_embed(S_k) @ Vh_k    # (B, T, N)
            w = (dx_coll ** 2) / (dx_padded ** 2 + eps)
            w = torch.clamp(w, min=0.0, max=1.0)
        return w.detach()

    def _build_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N)
        Returns: (B, T, N, feat_dim)
        """
        B, T, N = x.shape
        safe = torch.clamp(x, min=1e-6)
        log_x = torch.log(safe)
        feats = [x.unsqueeze(-1), log_x.unsqueeze(-1)]
        for lag in self.takens_lags:
            padded_x = F.pad(x.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            padded_log = F.pad(log_x.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            feats.append(padded_x.unsqueeze(-1))
            feats.append(padded_log.unsqueeze(-1))
        if self.use_coupling_weight:
            w = self._compute_coupling_weight(x)            # (B, T, N)
            feats.append(w.unsqueeze(-1))
        return torch.cat(feats, dim=-1)

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None,
                 species_ids: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode visible → posterior params.
        x: (B, T, N)
        residual: optional (B, T, N), used if use_residual_attn=True
        species_ids: optional (N,) long tensor of species IDs in [0, num_total_species) — used if use_species_id_emb
        Returns: μ (B, T), log_σ (B, T)
        """
        if x.dim() == 2:
            x = x.unsqueeze(0)
        B, T, N = x.shape

        features = self._build_features(x)
        if self.use_species_id_emb:
            if species_ids is None:
                species_ids = torch.arange(N, device=x.device)
            assert species_ids.shape[0] == N, \
                f"species_ids size {species_ids.shape[0]} != N={N}"
            sp_emb = self.species_emb_table[species_ids]   # (N, d_model)
        else:
            sp_emb = self.species_emb                       # (N, d_model)
        h = self.input_proj(features) + sp_emb.view(1, 1, N, -1)
        pe = sinusoidal_pe(T, self.d_model, h.device)
        h = h + pe.view(1, T, 1, self.d_model)

        # Pre-compute attention biases. Priority: residual_attn > coupling_attn > none.
        species_attn_mask = None
        temporal_attn_mask = None
        if self.use_residual_attn and residual is not None:
            # |R| 作 bias 源: log|R|_{t,j}, 按每 species 归一化
            R_mag = torch.abs(residual).clamp(min=1e-5)       # (B, T, N)
            R_norm = R_mag / R_mag.mean(dim=1, keepdim=True).clamp(min=1e-5)
            log_r = torch.log(R_norm.clamp(min=1e-4, max=100.0))   # (B, T, N)
            sp_bias = log_r.reshape(B * T, N).unsqueeze(1).expand(B * T, N, N)
            species_attn_mask = sp_bias.repeat_interleave(self.num_heads, dim=0).contiguous()
            log_r_bnt = log_r.permute(0, 2, 1).reshape(B * N, T)
            tp_bias = log_r_bnt.unsqueeze(1).expand(B * N, T, T)
            temporal_attn_mask = tp_bias.repeat_interleave(self.num_heads, dim=0).contiguous()
        elif self.use_coupling_attn:
            w = self._compute_coupling_weight(x)            # (B, T, N)
            log_w = torch.log(torch.clamp(w, min=1e-4))      # (B, T, N), ∈ [-∞~-4, 0]
            # Species attn mask: shape (B*T*num_heads, N, N)
            # bias[query i, key j] = log_w[b, t, j]
            sp_bias = log_w.reshape(B * T, N).unsqueeze(1).expand(B * T, N, N)
            species_attn_mask = sp_bias.repeat_interleave(self.num_heads, dim=0).contiguous()
            # Temporal attn mask: shape (B*N*num_heads, T, T)
            # bias[query t_q, key t_k] = log_w[b, t_k, n]
            log_w_bnt = log_w.permute(0, 2, 1).reshape(B * N, T)   # (B*N, T)
            tp_bias = log_w_bnt.unsqueeze(1).expand(B * N, T, T)
            temporal_attn_mask = tp_bias.repeat_interleave(self.num_heads, dim=0).contiguous()

        for block in self.blocks:
            # Species attention (per time)
            B_, T_, N_, D = h.shape
            x_s = h.reshape(B_ * T_, N_, D)
            a, _ = block["species_attn"](x_s, x_s, x_s, attn_mask=species_attn_mask)
            x_s = block["norm1"](x_s + a)
            h = x_s.reshape(B_, T_, N_, D)
            # Temporal attention
            x_t = h.permute(0, 2, 1, 3).contiguous().reshape(B_ * N_, T_, D)
            a, _ = block["temporal_attn"](x_t, x_t, x_t, attn_mask=temporal_attn_mask)
            x_t = block["norm2"](x_t + a)
            h = x_t.reshape(B_, N_, T_, D).permute(0, 2, 1, 3).contiguous()
            h = block["norm3"](h + block["ffn"](h))

        # Flatten species dim and readout
        h_flat = h.reshape(B, T, N * self.d_model)
        mu_K = self.readout_mu(h_flat)                     # (B, T, K)
        log_sigma_K = self.readout_logsigma(h_flat)
        log_sigma_K = torch.clamp(log_sigma_K, min=-3.0, max=1.0)
        if self.K == 1:
            # Backward-compat: return (B, T) 2-tuple
            return mu_K.squeeze(-1), log_sigma_K.squeeze(-1)
        # MoG path: return 3-tuple with logits per component
        logits_K = self.readout_logits(h_flat)             # (B, T, K)
        return mu_K, log_sigma_K, logits_K

    def sample(self, mu: torch.Tensor, log_sigma: torch.Tensor, n_samples: int = 1) -> torch.Tensor:
        """Sample h from posterior.
        mu, log_sigma: (B, T)
        Returns: h_samples (n_samples, B, T)
        """
        sigma = log_sigma.exp()
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        h = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        # Softplus to keep positive (hidden is abundance)
        return F.softplus(h) + 0.01


# =============================================================================
# Module 2: Data-Driven Dynamics Operator (Sparse GAT)
# =============================================================================
class SparseGAT(nn.Module):
    """Sparse GAT layer with top-k attention + L1 regularization."""
    def __init__(self, d_in: int, d_out: int, num_heads: int = 2, top_k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.top_k = top_k
        self.q = nn.Linear(d_in, d_out)
        self.k = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_out)
        self.out = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, N, d_in)
        Returns: (B, N, d_out), attention weights for L1
        """
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        k = self.k(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / (self.d_head ** 0.5)
        k_eff = min(self.top_k, N)
        topk_vals, topk_idx = scores.topk(k_eff, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        attn = F.softmax(mask, dim=-1)
        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.out(out), attn  # attn used for L1 reg


class DynamicsOperator(nn.Module):
    """Data-driven dynamics operator f_θ: (x_t, h_t) → log_ratio.

    Factorized design (防 GAT bypass hidden):
      log_ratio_visible(t) = GAT_visible(x_t) + b · h_t + c · h_t²
                              ↑ 只看 visible     ↑ hidden 必须 linear 参与
      log_ratio_hidden(t)   = GAT_hidden(x_t, h_t)  (hidden 自身 dynamics)

    No presumed LV/Holling form. Sparse GAT + L1 on attention.
    Hidden 对 visible 的影响**强制**通过 simple linear/quadratic coupling,
    防止 GAT 通过 visible-to-visible 吃掉 hidden signal。
    """
    def __init__(
        self,
        num_visible: int = 5,
        num_hidden: int = 1,
        d_model: int = 48,
        num_layers: int = 3,
        num_heads: int = 2,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_nodes = num_visible + num_hidden
        self.d_model = d_model

        # === Path 1: Visible baseline = Linear sparse (hard) + small GAT correction (soft) ===
        # 受约束的 visible baseline 防止 bypass hidden
        # Linear sparse part
        self.r_visible_sparse = nn.Parameter(torch.zeros(num_visible))
        A_init = 0.01 * torch.randn(num_visible, num_visible)
        A_init.fill_diagonal_(-0.2)
        self.A_visible_sparse = nn.Parameter(A_init)  # L1 regularized

        # Small GAT correction (capacity-limited)
        small_d = d_model // 2
        self.visible_input_proj = nn.Sequential(
            nn.Linear(2, small_d), nn.GELU(),
            nn.Linear(small_d, small_d),
        )
        self.visible_node_emb = nn.Parameter(torch.randn(num_visible, small_d) * 0.1)
        self.visible_gat_layers = nn.ModuleList([
            SparseGAT(small_d, small_d, num_heads=max(1, num_heads // 2), top_k=top_k, dropout=dropout)
            for _ in range(max(1, num_layers - 1))  # 浅一些
        ])
        self.visible_norms = nn.ModuleList([nn.LayerNorm(small_d) for _ in range(max(1, num_layers - 1))])
        self.visible_output_head = nn.Sequential(
            nn.Linear(small_d, small_d // 2), nn.GELU(),
            nn.Linear(small_d // 2, 1),
        )
        # GAT correction α-gate (start near 0, sigmoid(-5) ≈ 0.007)
        self.visible_gat_alpha_raw = nn.Parameter(torch.tensor(-5.0))

        # === Path 2: Hidden coupling (LINEAR, 强制 hidden 必须影响 visible) ===
        # hidden → visible coupling
        self.b_h2v = nn.Parameter(0.1 * torch.randn(num_visible))         # linear term
        self.c_h2v = nn.Parameter(0.02 * torch.randn(num_visible))        # quadratic term

        # === Path 3: Hidden self-dynamics (small GAT on full state) ===
        # 只用来预测 hidden 自身的 log_ratio，不影响 visible
        self.hidden_input_proj = nn.Sequential(
            nn.Linear(2, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.hidden_node_emb = nn.Parameter(torch.randn(self.num_nodes, d_model) * 0.1)
        self.hidden_gat_layers = nn.ModuleList([
            SparseGAT(d_model, d_model, num_heads=num_heads, top_k=top_k, dropout=dropout)
            for _ in range(2)  # 比 visible 浅
        ])
        self.hidden_norms = nn.ModuleList([nn.LayerNorm(d_model) for _ in range(2)])
        self.hidden_output_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2), nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Learnable intrinsic growth
        self.r_visible = nn.Parameter(torch.zeros(num_visible))
        self.r_hidden = nn.Parameter(torch.zeros(num_hidden))

    def forward(self, state: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        state: (B, T, num_visible + num_hidden) — concatenated [visible, hidden]
        Returns:
          log_ratio: (B, T, num_nodes)
          attn_weights: list
        """
        B, T, N_total = state.shape
        N = self.num_visible
        visible = state[:, :, :N]                    # (B, T, N_v)
        hidden = state[:, :, N:]                     # (B, T, N_h)

        # === Path 1: Visible baseline = Linear sparse (hard) + small GAT correction (soft) ===
        # Linear sparse part (main structure)
        visible_log_ratio_linear = self.r_visible_sparse.view(1, 1, -1) + visible @ self.A_visible_sparse.T

        # Small GAT correction (capacity-limited, α-gated)
        safe_v = torch.clamp(visible, min=1e-6)
        feats_v = torch.stack([visible, torch.log(safe_v)], dim=-1)
        hv = self.visible_input_proj(feats_v) + self.visible_node_emb.view(1, 1, N, -1)
        hv_flat = hv.reshape(B * T, N, -1)
        attn_maps = []
        for layer, norm in zip(self.visible_gat_layers, self.visible_norms):
            delta, attn = layer(hv_flat)
            hv_flat = norm(hv_flat + delta)
            attn_maps.append(attn)
        gat_correction = self.visible_output_head(hv_flat).squeeze(-1).reshape(B, T, N)
        alpha_gat = torch.sigmoid(self.visible_gat_alpha_raw)
        visible_log_ratio_base = visible_log_ratio_linear + alpha_gat * gat_correction

        # === Path 2: Hidden → Visible LINEAR coupling ===
        # For single hidden: just h_t · b + h_t² · c
        # For multiple hidden: h_t @ B_h2v, but we keep single-hidden for MVP
        h_t = hidden[:, :, 0]  # (B, T)   (assume num_hidden=1)
        hidden_coupling = h_t.unsqueeze(-1) * self.b_h2v.view(1, 1, -1) + \
                          (h_t.unsqueeze(-1) ** 2) * self.c_h2v.view(1, 1, -1)

        # Full visible log_ratio
        visible_log_ratio = visible_log_ratio_base + hidden_coupling

        # === Path 3: Hidden self-dynamics ===
        safe_all = torch.clamp(state, min=1e-6)
        feats_all = torch.stack([state, torch.log(safe_all)], dim=-1)  # (B, T, N_total, 2)
        hh = self.hidden_input_proj(feats_all) + self.hidden_node_emb.view(1, 1, N_total, -1)
        hh_flat = hh.reshape(B * T, N_total, -1)
        for layer, norm in zip(self.hidden_gat_layers, self.hidden_norms):
            delta, attn = layer(hh_flat)
            hh_flat = norm(hh_flat + delta)
            attn_maps.append(attn)
        hidden_log_ratio_full = self.hidden_output_head(hh_flat).squeeze(-1).reshape(B, T, N_total)
        # 只取 hidden 部分
        hidden_log_ratio = hidden_log_ratio_full[:, :, N:]  # (B, T, N_h)
        hidden_log_ratio = hidden_log_ratio + self.r_hidden.view(1, 1, -1)

        # Concat
        full_log_ratio = torch.cat([visible_log_ratio, hidden_log_ratio], dim=-1)
        return full_log_ratio, attn_maps

    def l1_on_attention(self, attn_maps: List[torch.Tensor]) -> torch.Tensor:
        """L1 on attention weights (encourages sparsity)."""
        reg = torch.tensor(0.0, device=attn_maps[0].device)
        for attn in attn_maps:
            # attn: (B*T, H, N, N); off-diagonal means real edges
            B_, H, N, _ = attn.shape
            mask = 1 - torch.eye(N, device=attn.device).view(1, 1, N, N)
            reg = reg + (attn * mask).abs().mean()
        return reg / len(attn_maps)

    def lipschitz_penalty(self, state: torch.Tensor, log_ratio: torch.Tensor, eps: float = 0.01) -> torch.Tensor:
        """Rough Lipschitz regularization: |df/dx| shouldn't be too large.

        Computed via finite differences."""
        with torch.no_grad():
            perturbation = eps * torch.randn_like(state)
        perturbed_state = state + perturbation
        perturbed_log_ratio, _ = self.forward(perturbed_state)
        # Compare
        diff = (perturbed_log_ratio - log_ratio).abs().mean()
        return diff / eps


# =============================================================================
# Module 3: CVHI — full model
# =============================================================================
class CVHI(nn.Module):
    """Conditional Variational Hidden Inference with h_coarse anchor.

    Anchored to h_coarse (from Linear Sparse + EM) as posterior mean initializer.
    Encoder only learns (delta_mu, log_sigma) — small corrections + uncertainty.
    This combines:
      - Linear Sparse + EM's strong point estimate (Pearson ~0.977 on LV)
      - CVHI's multi-hypothesis + uncertainty quantification
    """
    def __init__(
        self,
        num_visible: int = 5,
        num_hidden: int = 1,
        # Encoder
        encoder_d: int = 96,
        encoder_blocks: int = 3,
        encoder_heads: int = 4,
        takens_lags: List[int] = (1, 2, 4, 8),
        # Dynamics
        dynamics_d: int = 48,
        dynamics_layers: int = 3,
        dynamics_heads: int = 2,
        dynamics_top_k: int = 3,
        dropout: float = 0.1,
        # Prior
        prior_std: float = 2.0,
        clamp_min: float = -1.12,
        clamp_max: float = 0.92,
        use_h_anchor: bool = True,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.num_nodes = num_visible + num_hidden
        self.prior_std = prior_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.use_h_anchor = use_h_anchor

        self.encoder = PosteriorEncoder(
            num_visible=num_visible,
            takens_lags=takens_lags,
            d_model=encoder_d,
            num_heads=encoder_heads,
            num_blocks=encoder_blocks,
            dropout=dropout,
        )
        self.dynamics = DynamicsOperator(
            num_visible=num_visible,
            num_hidden=num_hidden,
            d_model=dynamics_d,
            num_layers=dynamics_layers,
            num_heads=dynamics_heads,
            top_k=dynamics_top_k,
            dropout=dropout,
        )

    def forward(self, visible: torch.Tensor, n_samples: int = 1,
                 h_anchor: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Encode → sample hidden → compute dynamics forward.

        visible: (B, T, N_visible) or (T, N_visible)
        h_anchor: (B, T) or (T,) optional — posterior mean anchor (from linear sparse + EM)
        Returns dict with mu, log_sigma, h_samples, predicted_log_ratio etc.
        """
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape

        # Step 1: Encode posterior parameters (delta_mu, log_sigma)
        delta_mu, log_sigma = self.encoder(visible)  # (B, T)

        # Step 2: Combine with anchor (additive in linear space, no softplus-inv hackery)
        if h_anchor is not None and self.use_h_anchor:
            if h_anchor.dim() == 1:
                h_anchor = h_anchor.unsqueeze(0)
            # mu = h_anchor + delta_mu (直接加法)
            # delta_mu 的 KL 是相对 prior N(0, prior_std²)
            mu = h_anchor + delta_mu
        else:
            mu = delta_mu

        # Step 3: Sample hidden — additive Gaussian, clamp positive
        sigma = log_sigma.exp()
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        h_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        # Clamp positive (hidden is abundance)
        h_samples = torch.clamp(h_samples, min=0.01)

        # Step 3: For each sample, compute dynamics
        # We'll process in a flattened way: (S*B, T, N_visible)
        visible_expanded = visible.unsqueeze(0).expand(n_samples, -1, -1, -1).reshape(n_samples * B, T, N)
        h_flat = h_samples.reshape(n_samples * B, T)
        # Concat visible + hidden to full state (..., T, N+1)
        state = torch.cat([visible_expanded, h_flat.unsqueeze(-1)], dim=-1)  # (S*B, T, N+1)

        log_ratio, attn_maps = self.dynamics(state)  # (S*B, T, N+1)

        # Compute actual log_ratio from visible (not using hidden)
        safe = torch.clamp(visible, min=1e-6)
        actual_log_ratio_visible = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio_visible = torch.clamp(actual_log_ratio_visible, self.clamp_min, self.clamp_max)

        # Predicted log_ratio for visible species (first N columns)
        predicted_log_ratio_visible = log_ratio[:, :-1, :N]  # (S*B, T-1, N)
        # Reshape back to (S, B, T-1, N)
        predicted_log_ratio_visible = predicted_log_ratio_visible.reshape(n_samples, B, T - 1, N)

        return {
            "mu": mu,
            "delta_mu": delta_mu,
            "log_sigma": log_sigma,
            "h_samples": h_samples,  # (S, B, T)
            "state": state.reshape(n_samples, B, T, N + 1),
            "predicted_log_ratio_visible": predicted_log_ratio_visible,  # (S, B, T-1, N)
            "actual_log_ratio_visible": actual_log_ratio_visible,  # (B, T-1, N)
            "full_log_ratio": log_ratio.reshape(n_samples, B, T, N + 1),
            "attn_maps": attn_maps,
        }

    def elbo_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        beta: float = 1.0,
        lam_sparse: float = 0.05,
        lam_lipschitz: float = 0.0,
        lam_smooth: float = 0.02,
        free_bits: float = 0.5,  # 每时间步 KL 至少这么多（防 collapse）
    ) -> Dict[str, torch.Tensor]:
        """ELBO: -log p(x|h) + β KL[q(h|x) ‖ p(h)] + regularizers.

        Free bits: per-time-step KL 不会低于 `free_bits`，
        防止 posterior 塌缩到 prior (encoder 什么也没学)。
        """
        mu = outputs["mu"]
        delta_mu = outputs.get("delta_mu", mu)  # anchor 后的偏差
        log_sigma = outputs["log_sigma"]
        pred = outputs["predicted_log_ratio_visible"]   # (S, B, T-1, N)
        actual = outputs["actual_log_ratio_visible"]    # (B, T-1, N)
        h_samples = outputs["h_samples"]                 # (S, B, T)

        # Reconstruction loss (averaged over samples)
        recon = F.mse_loss(pred, actual.unsqueeze(0).expand_as(pred))

        # KL: posterior q = N(mu, σ²) vs prior p = N(h_anchor_raw, prior_std²)
        # 即: KL measures how far delta_mu deviates from 0 and how narrow σ is
        # 用 delta_mu (deviation from anchor) 做 KL 计算
        prior_var = self.prior_std ** 2
        sigma_sq = torch.exp(2 * log_sigma)
        kl_per_step = 0.5 * (
            torch.log(torch.tensor(prior_var, device=mu.device)) - 2 * log_sigma
            + (sigma_sq + delta_mu ** 2) / prior_var - 1
        )  # (B, T)
        # Free bits: 只有超过 threshold 部分才惩罚
        kl_clipped = torch.clamp(kl_per_step - free_bits, min=0)
        kl = kl_clipped.mean()
        kl_raw = kl_per_step.mean()

        # Sparsity on attention + L1 on visible-only linear A
        sparse_attn = self.dynamics.l1_on_attention(outputs["attn_maps"])
        A_off = self.dynamics.A_visible_sparse - torch.diag(torch.diag(self.dynamics.A_visible_sparse))
        l1_A_visible = A_off.abs().mean()
        # 还加一个 anti-zero prior on coupling b (防 hidden bypass)
        b_penalty = F.relu(0.02 - self.dynamics.b_h2v.abs().mean())
        sparse = sparse_attn + 3.0 * l1_A_visible + 2.0 * b_penalty

        # Smoothness on sampled hidden (prevent sharp jumps)
        smooth = ((h_samples[:, :, 2:] - 2 * h_samples[:, :, 1:-1] + h_samples[:, :, :-2]) ** 2).mean()

        # Optional Lipschitz (expensive, disabled by default for speed)
        lipschitz = torch.tensor(0.0, device=mu.device)
        if lam_lipschitz > 0:
            state = outputs["state"][0]  # (B, T, N+1)
            lr_flat = outputs["full_log_ratio"][0]
            lipschitz = self.dynamics.lipschitz_penalty(state, lr_flat)

        total = recon + beta * kl + lam_sparse * sparse + lam_smooth * smooth + lam_lipschitz * lipschitz
        return {
            "total": total,
            "recon": recon,
            "kl": kl,
            "kl_raw": kl_raw,
            "sparse": sparse,
            "smooth": smooth,
            "lipschitz": lipschitz,
            "sigma_mean": log_sigma.exp().mean().detach(),
            "mu_std": mu.std().detach(),
        }

    @torch.no_grad()
    def generate_hypotheses(self, visible: torch.Tensor, n_hypotheses: int = 10,
                             h_anchor: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """Generate multiple plausible hidden hypotheses from posterior.

        Returns: h_hypotheses (K, B, T), reconstruction_scores (K, B)
        """
        outputs = self.forward(visible, n_samples=n_hypotheses, h_anchor=h_anchor)
        h_samples = outputs["h_samples"]  # (K, B, T)
        pred = outputs["predicted_log_ratio_visible"]  # (K, B, T-1, N)
        actual = outputs["actual_log_ratio_visible"]  # (B, T-1, N)

        errors = ((pred - actual.unsqueeze(0)) ** 2).mean(dim=(-1, -2)).sqrt()  # (K, B)
        return {
            "h_hypotheses": h_samples,
            "recon_rmse": errors,
            "mu": outputs["mu"],
            "log_sigma": outputs["log_sigma"],
        }

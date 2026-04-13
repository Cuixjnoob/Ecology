"""Residual-Signature GNN v2 — 加重生态残差与 Takens 双分支。

架构核心:
  [生态残差分支]  ← 重点
    LearnedBaseline: 线性 (r5 + A55·x) + shallow MLP nonlinear
    Multi-scale residual: k=1,2,3,5 步残差
    ResidualEncoder: 深度 MLP + self-attention
    → residual_embedding (B, T, N, D)

  [Takens 分支]   ← 加强
    Multi-τ delay embedding: τ=1,2,4,8
    每个 τ 独立 encoder
    → takens_embedding (B, T, N, D)

  [GNN 核心]
    Fusion: cross-attention between residual and Takens
    Multi-layer spatio-temporal attention (GNN core)
    → final_embedding (B, T, N, D)

  [Hidden Decoder]
    Aggregate over species with learned attention
    MLP → h(t)

  [Cycle Head]
    Predict next visible using learned (baseline + hidden coupling)
    Loss = MSE(predicted, actual visible)  ← 无 hidden 监督
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_pe(length: int, d_model: int, device) -> torch.Tensor:
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


class MLPBlock(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.norm(x + self.net(x))


# =============================================================================
# Branch 1: LearnedBaseline (visible-only dynamics estimator)
# =============================================================================
class LearnedBaseline(nn.Module):
    """学习的 visible-only 动力学 baseline。

    log_ratio_baseline(t) = r5 + A55·x_t + f_theta(x_t) + bias
    其中 f_theta 是一个 shallow MLP（非线性修正项）
    """
    def __init__(self, num_visible: int, mlp_hidden: int = 32, nonlinear_scale: float = 0.1):
        super().__init__()
        self.num_visible = num_visible
        self.r5 = nn.Parameter(0.1 * torch.ones(num_visible))
        A_init = 0.02 * torch.randn(num_visible, num_visible)
        A_init.fill_diagonal_(-0.2)
        self.A55 = nn.Parameter(A_init)
        self.bias = nn.Parameter(torch.zeros(num_visible))
        # 非线性修正项
        self.nonlinear = nn.Sequential(
            nn.Linear(num_visible, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, num_visible),
        )
        self.nonlinear_scale = nonlinear_scale

    def forward(self, x_t: torch.Tensor) -> torch.Tensor:
        """Compute baseline log-ratio.
        Args: x_t (B, T-1, N)
        Returns: (B, T-1, N)
        """
        linear_term = self.r5.view(1, 1, -1) + torch.einsum("btn,mn->btm", x_t, self.A55)
        nonlinear_term = self.nonlinear(x_t)
        return linear_term + self.nonlinear_scale * nonlinear_term + self.bias.view(1, 1, -1)


# =============================================================================
# Branch 2: Multi-scale Residual Extractor
# =============================================================================
class MultiScaleResidual(nn.Module):
    """计算多尺度残差：k-step log-ratio 的 baseline 预测误差。

    对 k ∈ scales:
      actual_k_log_ratio = log(x_{t+k} / x_t)
      baseline_k_log_ratio ≈ sum_{i=0}^{k-1} baseline(x_{t+i})  // 近似 k 步累积
      residual_k = actual - baseline_k

    Returns: (B, T, N, len(scales)) multi-scale residuals
    """
    def __init__(self, scales: List[int] = (1, 2, 3, 5), clamp_min: float = -2.5, clamp_max: float = 2.0):
        super().__init__()
        self.scales = tuple(scales)
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

    def forward(self, visible_states: torch.Tensor, baseline_model: LearnedBaseline) -> torch.Tensor:
        B, T, N = visible_states.shape
        safe = torch.clamp(visible_states, min=1e-6)

        # Compute 1-step baseline log-ratio once
        x_t = visible_states[:, :-1]                          # (B, T-1, N)
        baseline_1step = baseline_model(x_t)                  # (B, T-1, N)

        residuals = []
        for k in self.scales:
            if k == 1:
                actual = torch.log(safe[:, 1:] / safe[:, :-1])
                actual = torch.clamp(actual, self.clamp_min, self.clamp_max)
                res = actual - baseline_1step
                res_padded = F.pad(res, (0, 0, 0, 1), value=0.0)  # (B, T, N) 末尾 pad 0
            else:
                # k-step: actual = log(x[t+k]/x[t])
                if T - k <= 0:
                    res_padded = visible_states.new_zeros(B, T, N)
                else:
                    actual = torch.log(safe[:, k:] / safe[:, :-k])
                    actual = torch.clamp(actual, self.clamp_min * k, self.clamp_max * k)
                    # baseline_k = sum_{i=0}^{k-1} baseline(x[t+i])
                    baseline_k = torch.zeros_like(actual)
                    for i in range(k):
                        if T - k - 1 + i + 1 >= baseline_1step.shape[1]:
                            baseline_i = baseline_1step[:, i:i + (T - k), :]
                        else:
                            baseline_i = baseline_1step[:, i:i + (T - k), :]
                        # Make sure length matches
                        if baseline_i.shape[1] != T - k:
                            baseline_i = baseline_1step[:, :T - k, :]
                        baseline_k = baseline_k + baseline_i
                    res = actual - baseline_k
                    # Pad to T: pad the end
                    res_padded = F.pad(res, (0, 0, 0, k), value=0.0)
            residuals.append(res_padded)

        return torch.stack(residuals, dim=-1)  # (B, T, N, len(scales))


# =============================================================================
# Branch 3: Residual Encoder (deep MLP + self-attention)
# =============================================================================
class ResidualEncoder(nn.Module):
    """深度残差编码器。

    输入: residuals (B, T, N, num_scales) 多尺度残差
    流程:
      1. 每物种 per-scale → concat → MLP to d_model
      2. 时间维度 self-attention（看看不同时间点的残差信号）
      3. 输出: (B, T, N, d_model)
    """
    def __init__(
        self,
        num_visible: int,
        num_scales: int,
        takens_lags: List[int],
        d_model: int,
        num_heads: int = 4,
        num_mlp_layers: int = 2,
        num_attn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_scales = num_scales
        self.takens_lags = takens_lags
        self.d_model = d_model

        # 输入维度：num_scales * (1 + len(takens_lags))
        # 每个尺度的残差有 1 个当前值 + len(takens_lags) 个历史值
        input_dim = num_scales * (1 + len(takens_lags))
        layers = [nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_mlp_layers - 1):
            layers.extend([nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.LayerNorm(d_model))
        self.input_mlp = nn.Sequential(*layers)

        # Temporal self-attention (per-species)
        self.temporal_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            for _ in range(num_attn_layers)
        ])

    def _build_takens_residual(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        residuals: (B, T, N, num_scales)
        Returns: (B, T, N, num_scales * (1 + num_lags))
        """
        B, T, N, S = residuals.shape
        delayed_list = [residuals]  # 原始值
        for lag in self.takens_lags:
            padded = F.pad(residuals.permute(0, 2, 3, 1), (lag, 0), value=0.0)[..., :T]
            # (B, N, S, T) → back to (B, T, N, S)
            delayed = padded.permute(0, 3, 1, 2)
            delayed_list.append(delayed)
        # Concat along last dim
        stacked = torch.cat(delayed_list, dim=-1)  # (B, T, N, S * (1 + L))
        return stacked

    def forward(self, residuals: torch.Tensor) -> torch.Tensor:
        """
        Args: residuals (B, T, N, num_scales)
        Returns: (B, T, N, d_model)
        """
        B, T, N, S = residuals.shape

        # Takens of residuals
        x = self._build_takens_residual(residuals)  # (B, T, N, S * (1 + L))
        # Encode each (t, n) token
        x = self.input_mlp(x)  # (B, T, N, d_model)

        # Add temporal PE
        pe = sinusoidal_pe(T, self.d_model, x.device)
        x = x + pe.view(1, T, 1, self.d_model)

        # Temporal attention per species
        # Reshape to (B*N, T, d_model) for attention
        x_t = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, self.d_model)
        for layer in self.temporal_attn_layers:
            x_t = layer(x_t)
        x = x_t.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3).contiguous()
        return x


# =============================================================================
# Branch 4: State Encoder with Takens
# =============================================================================
class StateEncoder(nn.Module):
    """State encoder with multi-τ Takens delay embedding."""
    def __init__(
        self,
        num_visible: int,
        takens_lags: List[int],
        d_model: int,
        num_heads: int = 4,
        num_mlp_layers: int = 2,
        num_attn_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.takens_lags = takens_lags
        self.d_model = d_model

        # Features per (t, n): raw, log, Takens delays (each has raw & log)
        input_dim = 2 * (1 + len(takens_lags))  # raw + log × (current + lags)
        layers = [nn.Linear(input_dim, d_model), nn.GELU(), nn.Dropout(dropout)]
        for _ in range(num_mlp_layers - 1):
            layers.extend([nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout)])
        layers.append(nn.LayerNorm(d_model))
        self.input_mlp = nn.Sequential(*layers)

        self.temporal_attn_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=d_model, nhead=num_heads, dim_feedforward=d_model * 2,
                dropout=dropout, batch_first=True, activation="gelu",
            )
            for _ in range(num_attn_layers)
        ])

    def _build_takens(self, states: torch.Tensor) -> torch.Tensor:
        """
        states: (B, T, N)
        Returns: (B, T, N, 2 * (1 + num_lags))
        """
        B, T, N = states.shape
        safe = torch.clamp(states, min=1e-6)
        log_states = torch.log(safe)

        all_feats = [states.unsqueeze(-1), log_states.unsqueeze(-1)]  # [raw, log]
        for lag in self.takens_lags:
            padded_state = F.pad(states.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            padded_log = F.pad(log_states.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            all_feats.append(padded_state.unsqueeze(-1))
            all_feats.append(padded_log.unsqueeze(-1))
        return torch.cat(all_feats, dim=-1)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        B, T, N = states.shape
        x = self._build_takens(states)  # (B, T, N, feat_dim)
        x = self.input_mlp(x)  # (B, T, N, d_model)

        pe = sinusoidal_pe(T, self.d_model, x.device)
        x = x + pe.view(1, T, 1, self.d_model)

        x_t = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, self.d_model)
        for layer in self.temporal_attn_layers:
            x_t = layer(x_t)
        x = x_t.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3).contiguous()
        return x


# =============================================================================
# Cross-Attention Fusion
# =============================================================================
class CrossAttentionFusion(nn.Module):
    """State embedding queries Residual embedding; fuse."""
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, state_emb: torch.Tensor, residual_emb: torch.Tensor) -> torch.Tensor:
        """
        Both: (B, T, N, d_model)
        Returns: fused (B, T, N, d_model)
        """
        B, T, N, D = state_emb.shape
        # Flatten (T, N) for attention
        s_flat = state_emb.reshape(B, T * N, D)
        r_flat = residual_emb.reshape(B, T * N, D)
        # Cross attention: state queries, residual K/V
        attn_out, _ = self.cross_attn(s_flat, r_flat, r_flat)
        s_flat = self.norm1(s_flat + attn_out)
        # FFN on concat
        fused = self.ffn(torch.cat([s_flat, r_flat], dim=-1))
        fused = self.norm2(s_flat + fused)
        return fused.reshape(B, T, N, D)


# =============================================================================
# Spatio-Temporal GNN Block
# =============================================================================
class SpatioTemporalBlock(nn.Module):
    """Species attention → Temporal attention → FFN."""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.species_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        # Species attention (per time)
        x_s = x.reshape(B * T, N, D)
        a, _ = self.species_attn(x_s, x_s, x_s)
        x_s = self.norm1(x_s + a)
        x = x_s.reshape(B, T, N, D)
        # Temporal attention (per species)
        x_t = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D)
        a, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = self.norm2(x_t + a)
        x = x_t.reshape(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        # FFN
        x = self.norm3(x + self.ffn(x))
        return x


# =============================================================================
# Full RS-GNN v2
# =============================================================================
class RSGNNv2(nn.Module):
    def __init__(
        self,
        num_visible: int = 5,
        num_steps: int = 820,
        # Residual branch
        residual_scales: List[int] = (1, 2, 3, 5),
        residual_takens_lags: List[int] = (1, 2, 4, 8),
        residual_d_model: int = 96,
        residual_mlp_layers: int = 3,
        residual_attn_layers: int = 3,
        # State branch
        state_takens_lags: List[int] = (1, 2, 4, 8),
        state_d_model: int = 96,
        state_mlp_layers: int = 2,
        state_attn_layers: int = 2,
        # Fusion
        fusion_d_model: int = 128,
        # Core GNN
        num_core_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_steps = num_steps
        self.fusion_d_model = fusion_d_model

        # Baselines
        self.baseline = LearnedBaseline(num_visible)

        # Residual branch
        self.residual_extractor = MultiScaleResidual(residual_scales)
        self.residual_encoder = ResidualEncoder(
            num_visible=num_visible,
            num_scales=len(residual_scales),
            takens_lags=list(residual_takens_lags),
            d_model=residual_d_model,
            num_heads=num_heads // 2,
            num_mlp_layers=residual_mlp_layers,
            num_attn_layers=residual_attn_layers,
            dropout=dropout,
        )

        # State branch
        self.state_encoder = StateEncoder(
            num_visible=num_visible,
            takens_lags=list(state_takens_lags),
            d_model=state_d_model,
            num_heads=num_heads // 2,
            num_mlp_layers=state_mlp_layers,
            num_attn_layers=state_attn_layers,
            dropout=dropout,
        )

        # Project both to fusion_d_model if different
        self.proj_residual = nn.Linear(residual_d_model, fusion_d_model) if residual_d_model != fusion_d_model else nn.Identity()
        self.proj_state = nn.Linear(state_d_model, fusion_d_model) if state_d_model != fusion_d_model else nn.Identity()

        # Cross-attention fusion
        self.fusion = CrossAttentionFusion(fusion_d_model, num_heads=num_heads // 2, dropout=dropout)

        # Core spatio-temporal GNN blocks
        self.core_blocks = nn.ModuleList([
            SpatioTemporalBlock(fusion_d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_core_blocks)
        ])

        # Hidden decoder
        self.hidden_readout_attn = nn.Parameter(torch.ones(num_visible) / num_visible)
        self.hidden_decoder = nn.Sequential(
            nn.Linear(fusion_d_model * num_visible, fusion_d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_d_model * 2, fusion_d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_d_model, 1),
        )

        # Hidden → visible coupling (for cycle loss)
        self.coupling_b = nn.Parameter(0.1 * torch.randn(num_visible))
        self.coupling_c = nn.Parameter(0.01 * torch.randn(num_visible))

    def forward(self, visible_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        if visible_states.dim() == 2:
            visible_states = visible_states.unsqueeze(0)
        B, T, N = visible_states.shape

        # Branch 1: Extract multi-scale residuals
        residuals_multiscale = self.residual_extractor(visible_states, self.baseline)  # (B, T, N, S)
        # Encode residuals
        residual_emb = self.residual_encoder(residuals_multiscale)  # (B, T, N, D_res)
        residual_emb = self.proj_residual(residual_emb)  # (B, T, N, D_fusion)

        # Branch 2: State encoder
        state_emb = self.state_encoder(visible_states)  # (B, T, N, D_state)
        state_emb = self.proj_state(state_emb)  # (B, T, N, D_fusion)

        # Fusion
        fused = self.fusion(state_emb, residual_emb)  # (B, T, N, D_fusion)

        # Core GNN blocks
        h = fused
        for block in self.core_blocks:
            h = block(h)

        # Hidden decoder: flatten species dim
        h_flat = h.reshape(B, T, N * self.fusion_d_model)
        hidden_raw = self.hidden_decoder(h_flat).squeeze(-1)
        hidden = F.softplus(hidden_raw) + 0.01

        # Cycle reconstruction
        x_current = visible_states[:, :-1]
        h_current = hidden[:, :-1]
        baseline_log_ratio = self.baseline(x_current)
        hidden_linear = h_current.unsqueeze(-1) * self.coupling_b.view(1, 1, -1)
        hidden_quad = (h_current.unsqueeze(-1) ** 2) * self.coupling_c.view(1, 1, -1)
        reconstructed_log_ratio = baseline_log_ratio + hidden_linear + hidden_quad

        # Actual
        safe = torch.clamp(visible_states, min=1e-6)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, min=-1.12, max=0.92)

        return {
            "hidden": hidden,
            "reconstructed_log_ratio": reconstructed_log_ratio,
            "actual_log_ratio": actual_log_ratio,
            "baseline_log_ratio": baseline_log_ratio,
            "residuals_multiscale": residuals_multiscale,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        smooth_lambda: float = 0.01,
        sparse_lambda: float = 0.001,
        var_lambda: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        actual = outputs["actual_log_ratio"]
        reconstructed = outputs["reconstructed_log_ratio"]
        hidden = outputs["hidden"]

        fit_loss = F.mse_loss(reconstructed, actual)
        smooth_loss = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        A = self.baseline.A55
        A_offdiag = A - torch.diag(torch.diag(A))
        sparse_loss = A_offdiag.abs().mean()
        h_var = hidden.var(dim=-1).mean()
        var_loss = F.relu(0.05 - h_var)

        total = fit_loss + smooth_lambda * smooth_loss + sparse_lambda * sparse_loss + var_lambda * var_loss

        return {
            "total": total,
            "fit": fit_loss,
            "smooth": smooth_loss,
            "sparse": sparse_loss,
            "var": var_loss,
            "h_variance": h_var.detach(),
        }

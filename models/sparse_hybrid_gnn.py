"""SparseGNN Hybrid: 稀疏 BaselineGNN + 深度 HiddenDecoderGNN + Coupling。

核心设计原则（从线性实验学到的 insight）：
  BaselineGNN 必须受限，否则会吃掉 hidden signal。
  具体：小容量 + 稀疏注意力 + L1 权重正则。

架构：
  1. BaselineGNN（小，稀疏）：
     - 2-3 层 graph attention
     - top-k neighbor sparsity (k 小)
     - hidden dim 小 (32)
     - L1 on params
     - 不知道动力学形式，从数据学但受限制
     - 输出: baseline_log_ratio(t+1) per species

  2. HiddenDecoderGNN（大，深度）：
     - 输入: visible, Takens, baseline residual（关键！）
     - Multi-layer spatio-temporal attention
     - hidden dim 大 (128)
     - 输出: hidden(t) 时间序列

  3. Coupling（shallow）:
     - Hidden → visible effect
     - 简单线性+二次形式，防 shortcut

  4. Forward:
     predicted_log_ratio(t+1) = BaselineGNN(visible_t)
                              + Coupling(hidden(t))

  5. Loss（严格无 hidden 监督）:
     L = MSE(predicted, actual log_ratio)
       + λ_sparse · ||Baseline weights||_1
       + λ_smooth · smoothness(hidden)
       + λ_var · keep hidden variance non-trivial
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F


def sinusoidal_pe(length, d_model, device):
    pe = torch.zeros(length, d_model, device=device)
    position = torch.arange(0, length, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2, device=device).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =============================================================================
# BaselineGNN: low-capacity, sparse attention, learns visible-only dynamics
# =============================================================================
class SparseGATLayer(nn.Module):
    """Simplified GAT with top-k sparse attention.

    Each species attends only to top-k others (sparsity enforced).
    """
    def __init__(self, d_in: int, d_out: int, num_heads: int = 2, top_k: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.top_k = top_k
        self.q = nn.Linear(d_in, d_out)
        self.k = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_out)
        self.out = nn.Linear(d_out, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, N, d_in)  or (B*T, N, d_in)
        B, N, _ = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)  # (B, H, N, d)
        k = self.k(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)
        v = self.v(x).reshape(B, N, self.num_heads, self.d_head).transpose(1, 2)

        scores = torch.einsum("bhnd,bhmd->bhnm", q, k) / (self.d_head ** 0.5)  # (B, H, N, N)
        # Top-k masking: only keep top-k neighbors per node
        k_eff = min(self.top_k, N)
        topk_vals, topk_idx = scores.topk(k_eff, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        attn = F.softmax(mask, dim=-1)  # (B, H, N, N)

        out = torch.einsum("bhnm,bhmd->bhnd", attn, v)  # (B, H, N, d)
        out = out.transpose(1, 2).reshape(B, N, -1)  # (B, N, d_out)
        return self.out(out)


class BaselineGNN(nn.Module):
    """低容量 GNN baseline，学 visible-only 动力学但受限制。

    输入: x_t (B, T-1, N) — current visible state
    输出: baseline_log_ratio(t+1) (B, T-1, N)
    """
    def __init__(
        self,
        num_visible: int,
        d_hidden: int = 32,
        num_layers: int = 2,
        num_heads: int = 2,
        top_k: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_visible = num_visible

        # Embed each species's state (1 scalar + species one-hot)
        self.input_proj = nn.Sequential(
            nn.Linear(2, d_hidden),  # [raw, log]
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        # Species embedding (to distinguish nodes)
        self.species_emb = nn.Parameter(torch.randn(num_visible, d_hidden) * 0.1)

        self.layers = nn.ModuleList([
            SparseGATLayer(d_hidden, d_hidden, num_heads=num_heads, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)

        # Output: scalar per species = predicted log_ratio
        self.output_head = nn.Linear(d_hidden, 1)

        # Intrinsic growth (constant per species)
        self.r_intrinsic = nn.Parameter(torch.zeros(num_visible))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args: x (B, T-1, N) visible state
        Returns: baseline_log_ratio (B, T-1, N)
        """
        B, T, N = x.shape
        # Per (t, n) node feature: [raw_state, log_state]
        x_log = torch.log(torch.clamp(x, min=1e-6))
        node_features = torch.stack([x, x_log], dim=-1)  # (B, T, N, 2)
        h = self.input_proj(node_features)  # (B, T, N, d_hidden)
        h = h + self.species_emb.view(1, 1, N, -1)

        # Flatten time into batch for per-timestep graph
        h_flat = h.reshape(B * T, N, -1)  # (B*T, N, d_hidden)

        for layer, norm in zip(self.layers, self.norms):
            h_flat = norm(h_flat + self.dropout(layer(h_flat)))

        out = self.output_head(h_flat).squeeze(-1)  # (B*T, N)
        out = out.reshape(B, T, N)
        out = out + self.r_intrinsic.view(1, 1, -1)
        return out

    def l1_regularization(self) -> torch.Tensor:
        """L1 on all baseline linear weights (encourages sparse dynamics)."""
        reg = 0.0
        for m in [self.input_proj, self.output_head]:
            for p in m.parameters():
                if p.dim() > 1:  # only matrices
                    reg = reg + p.abs().mean()
        for layer in self.layers:
            for mod in [layer.q, layer.k, layer.v, layer.out]:
                reg = reg + mod.weight.abs().mean()
        return reg


# =============================================================================
# HiddenDecoderGNN: high-capacity, uses residual + Takens + visible
# =============================================================================
class HiddenSpatioTemporalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.species_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model), nn.Dropout(dropout),
        )
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, x):
        B, T, N, D = x.shape
        x_s = x.reshape(B * T, N, D)
        a, _ = self.species_attn(x_s, x_s, x_s)
        x_s = self.norm1(x_s + a)
        x = x_s.reshape(B, T, N, D)
        x_t = x.permute(0, 2, 1, 3).contiguous().reshape(B * N, T, D)
        a, _ = self.temporal_attn(x_t, x_t, x_t)
        x_t = self.norm2(x_t + a)
        x = x_t.reshape(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        x = self.norm3(x + self.ffn(x))
        return x


class HiddenDecoderGNN(nn.Module):
    """深度 GNN 从 visible + Takens + residual 推 hidden。

    输入:
      visible_states (B, T, N)
      baseline_residual (B, T, N) — 来自 BaselineGNN 的残差（关键信号）
    输出:
      hidden (B, T)
    """
    def __init__(
        self,
        num_visible: int,
        takens_lags: List[int] = (1, 2, 4, 8),
        d_model: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.takens_lags = takens_lags
        self.d_model = d_model

        # Features per (t, n):
        # [raw, log, raw Takens (L), log Takens (L), residual, residual Takens (L)]
        feat_dim = 2 + 2 * len(takens_lags) + 1 + len(takens_lags)
        self.feat_dim = feat_dim

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )
        self.species_emb = nn.Parameter(torch.randn(num_visible, d_model) * 0.1)

        self.blocks = nn.ModuleList([
            HiddenSpatioTemporalBlock(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Readout: per-timestep aggregate species embeddings → scalar hidden
        self.readout = nn.Sequential(
            nn.Linear(d_model * num_visible, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _build_features(self, visible: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        """
        visible: (B, T, N) raw state
        residual: (B, T, N) baseline residual (already computed)
        Returns: (B, T, N, feat_dim)
        """
        B, T, N = visible.shape
        safe = torch.clamp(visible, min=1e-6)
        log_v = torch.log(safe)

        feats = [visible.unsqueeze(-1), log_v.unsqueeze(-1)]
        # Takens for raw visible
        for lag in self.takens_lags:
            padded = F.pad(visible.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            feats.append(padded.unsqueeze(-1))
        # Takens for log visible
        for lag in self.takens_lags:
            padded = F.pad(log_v.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            feats.append(padded.unsqueeze(-1))
        # Residual
        feats.append(residual.unsqueeze(-1))
        # Takens for residual
        for lag in self.takens_lags:
            padded = F.pad(residual.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            feats.append(padded.unsqueeze(-1))
        return torch.cat(feats, dim=-1)  # (B, T, N, feat_dim)

    def forward(self, visible: torch.Tensor, residual: torch.Tensor) -> torch.Tensor:
        B, T, N = visible.shape
        features = self._build_features(visible, residual)
        h = self.input_proj(features)  # (B, T, N, d_model)
        h = h + self.species_emb.view(1, 1, N, -1)
        pe = sinusoidal_pe(T, self.d_model, h.device)
        h = h + pe.view(1, T, 1, self.d_model)

        for block in self.blocks:
            h = block(h)

        h_flat = h.reshape(B, T, N * self.d_model)
        hidden_raw = self.readout(h_flat).squeeze(-1)
        # Positive hidden
        hidden = F.softplus(hidden_raw) + 0.01
        return hidden


# =============================================================================
# Full Hybrid Model
# =============================================================================
class SparseHybridGNN(nn.Module):
    """完整架构: BaselineGNN (小, 稀疏) + HiddenDecoderGNN (大) + Coupling."""

    def __init__(
        self,
        num_visible: int = 5,
        # Baseline (小)
        baseline_d_hidden: int = 32,
        baseline_num_layers: int = 2,
        baseline_top_k: int = 3,
        baseline_num_heads: int = 2,
        # Hidden decoder (大)
        hidden_takens_lags: List[int] = (1, 2, 4, 8),
        hidden_d_model: int = 128,
        hidden_num_blocks: int = 4,
        hidden_num_heads: int = 8,
        dropout: float = 0.1,
        clamp_min: float = -1.12,
        clamp_max: float = 0.92,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.baseline_gnn = BaselineGNN(
            num_visible=num_visible,
            d_hidden=baseline_d_hidden,
            num_layers=baseline_num_layers,
            num_heads=baseline_num_heads,
            top_k=baseline_top_k,
            dropout=dropout,
        )
        self.hidden_decoder = HiddenDecoderGNN(
            num_visible=num_visible,
            takens_lags=list(hidden_takens_lags),
            d_model=hidden_d_model,
            num_heads=hidden_num_heads,
            num_blocks=hidden_num_blocks,
            dropout=dropout,
        )

        # Coupling: hidden → visible (simple, low capacity)
        self.coupling_b = nn.Parameter(0.1 * torch.randn(num_visible))
        self.coupling_c = nn.Parameter(0.01 * torch.randn(num_visible))

    def forward(self, visible_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args: visible_states (T, N) or (B, T, N)
        Returns:
          hidden (B, T)
          baseline_log_ratio (B, T-1, N)
          predicted_log_ratio (B, T-1, N)
          actual_log_ratio (B, T-1, N)
          residual (B, T, N) — padded to T for hidden decoder input
        """
        if visible_states.dim() == 2:
            visible_states = visible_states.unsqueeze(0)
        B, T, N = visible_states.shape

        # Step 1: BaselineGNN predicts log_ratio from visible state
        x_current = visible_states[:, :-1]  # (B, T-1, N)
        baseline_log_ratio = self.baseline_gnn(x_current)  # (B, T-1, N)

        # Step 2: Compute residual (actual - baseline)
        safe = torch.clamp(visible_states, min=1e-6)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, self.clamp_min, self.clamp_max)
        residual = actual_log_ratio - baseline_log_ratio  # (B, T-1, N)

        # Pad residual to T by zero-padding at t=0 (for feeding to hidden decoder)
        residual_padded = F.pad(residual, (0, 0, 1, 0), value=0.0)  # (B, T, N)

        # Step 3: HiddenDecoderGNN recovers hidden
        hidden = self.hidden_decoder(visible_states, residual_padded)  # (B, T)

        # Step 4: Coupling — hidden → visible effect
        h_current = hidden[:, :-1]  # (B, T-1)
        hidden_linear = h_current.unsqueeze(-1) * self.coupling_b.view(1, 1, -1)
        hidden_quad = (h_current.unsqueeze(-1) ** 2) * self.coupling_c.view(1, 1, -1)
        predicted_log_ratio = baseline_log_ratio + hidden_linear + hidden_quad

        return {
            "hidden": hidden,
            "baseline_log_ratio": baseline_log_ratio,
            "predicted_log_ratio": predicted_log_ratio,
            "actual_log_ratio": actual_log_ratio,
            "residual": residual,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        sparse_lambda: float = 0.5,
        smooth_lambda: float = 0.01,
        var_lambda: float = 0.2,
    ) -> Dict[str, torch.Tensor]:
        actual = outputs["actual_log_ratio"]
        pred = outputs["predicted_log_ratio"]
        hidden = outputs["hidden"]

        fit_loss = F.mse_loss(pred, actual)
        # Sparse baseline regularization (key insight from linear experiments)
        sparse_loss = self.baseline_gnn.l1_regularization()
        # Hidden smoothness
        smooth_loss = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        # Keep hidden variance non-trivial
        h_var = hidden.var(dim=-1).mean()
        var_loss = F.relu(0.05 - h_var)

        total = fit_loss + sparse_lambda * sparse_loss + smooth_lambda * smooth_loss + var_lambda * var_loss

        return {
            "total": total,
            "fit": fit_loss,
            "sparse": sparse_loss,
            "smooth": smooth_loss,
            "var": var_loss,
            "h_variance": h_var.detach(),
        }

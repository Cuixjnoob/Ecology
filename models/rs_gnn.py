"""Residual-Signature GNN (RS-GNN) for Hidden Species Recovery.

架构要点：
  1. Feature engineering（节点特征）：
     - Raw state + log state
     - Takens 延迟嵌入（捕获动力学结构）
     - Residual dynamics（visible-only baseline 的预测误差，作为 hidden 的"指纹线索"）
     - Residual 的 Takens 延迟嵌入

  2. GNN 核心（多层物种间 message passing）：
     - 多头 Graph Attention (nodes = 5 visible species, fully connected)
     - Residual connections + LayerNorm + FFN

  3. Temporal Self-Attention (时间维度):
     - Causal masked attention over time for each species
     - Positional encoding

  4. Alternating spatial-temporal blocks (多轮)

  5. Hidden Decoder:
     - 聚合所有 species 的 embedding → MLP → h(t)

  6. Cycle-consistency head:
     - 从 recovered hidden 预测下一步 visible（通过 learned coupling）
     - Loss = 与真实 visible 的 MSE（完全无 hidden 监督）

使用：
  from models.rs_gnn import RSGNN
  model = RSGNN(num_visible=5, num_steps=820, ...)
  h_pred, visible_next_pred, baseline_params = model(visible_states)
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
from torch import nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (固定版，支持任意长度)"""
    def __init__(self, d_model: int, max_len: int = 2000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (..., T, d_model)
        return x + self.pe[: x.shape[-2]]


class FeedForward(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpeciesAttention(nn.Module):
    """Multi-head attention across species at each time point.

    Input: (B, T, N, D) where N = num_visible
    Output: (B, T, N, D)
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * 2, dropout)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        # Flatten time into batch dim for per-timestep species attention
        x_flat = x.view(B * T, N, D)
        attn_out, _ = self.attn(x_flat, x_flat, x_flat)
        x_flat = self.norm(x_flat + attn_out)
        x_flat = self.norm2(x_flat + self.ffn(x_flat))
        return x_flat.view(B, T, N, D)


class TemporalAttention(nn.Module):
    """Multi-head self-attention across time for each species (causal optional).

    Input: (B, T, N, D)
    Output: (B, T, N, D)
    """
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_model * 2, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.causal = causal

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, N, D = x.shape
        # Transpose to (B, N, T, D), flatten N into batch
        x_t = x.permute(0, 2, 1, 3).contiguous().view(B * N, T, D)
        mask = None
        if self.causal:
            mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        attn_out, _ = self.attn(x_t, x_t, x_t, attn_mask=mask)
        x_t = self.norm(x_t + attn_out)
        x_t = self.norm2(x_t + self.ffn(x_t))
        x_t = x_t.view(B, N, T, D).permute(0, 2, 1, 3).contiguous()
        return x_t


class SpatioTemporalBlock(nn.Module):
    """One block: species attention → temporal attention."""
    def __init__(self, d_model: int, num_heads: int = 8, dropout: float = 0.1, causal: bool = False):
        super().__init__()
        self.spatial = SpeciesAttention(d_model, num_heads, dropout)
        self.temporal = TemporalAttention(d_model, num_heads, dropout, causal=causal)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.spatial(x)
        x = self.temporal(x)
        return x


class RSGNN(nn.Module):
    """Residual-Signature GNN for hidden species recovery."""

    def __init__(
        self,
        num_visible: int = 5,
        num_steps: int = 820,
        delay_steps: int = 6,
        d_model: int = 128,
        num_heads: int = 8,
        num_blocks: int = 4,
        dropout: float = 0.1,
        clamp_min: float = -1.12,
        clamp_max: float = 0.92,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_steps = num_steps
        self.delay_steps = delay_steps
        self.d_model = d_model
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Visible-only baseline (learnable, jointly optimized)
        self.r5 = nn.Parameter(0.1 * torch.ones(num_visible))
        A_init = 0.02 * torch.randn(num_visible, num_visible)
        A_init.fill_diagonal_(-0.2)
        self.A55 = nn.Parameter(A_init)
        # Hidden → visible coupling
        self.b5 = nn.Parameter(0.1 * torch.randn(num_visible))
        self.c5 = nn.Parameter(0.01 * torch.randn(num_visible))  # quadratic
        self.bias5 = nn.Parameter(torch.zeros(num_visible))

        # Feature dimension:
        # [state, log_state] = 2
        # Takens delay of state: delay_steps
        # Takens delay of residual: delay_steps
        # Position of species (one-hot): num_visible
        feat_dim = 2 + 2 * delay_steps + num_visible
        self.feat_dim = feat_dim

        # Feature encoder: project raw features to d_model
        self.feat_encoder = nn.Sequential(
            nn.Linear(feat_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
        )

        # Positional encoding for time
        self.pos_encoding = PositionalEncoding(d_model)

        # Multiple spatio-temporal blocks
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d_model, num_heads, dropout, causal=False)
            for _ in range(num_blocks)
        ])

        # Hidden decoder: aggregate all species → h(t)
        # Use learnable species weights + MLP
        self.species_weights = nn.Parameter(torch.ones(num_visible) / num_visible)
        self.hidden_decoder = nn.Sequential(
            nn.Linear(d_model * num_visible, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def _build_features(self, visible_states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build node features for each (time, species).

        Args:
          visible_states: (B, T, N) visible observations

        Returns:
          features: (B, T, N, feat_dim)
          residual: (B, T-1, N) residual log-ratio (used for feature and loss)
        """
        B, T, N = visible_states.shape
        device = visible_states.device

        # Log state
        safe = torch.clamp(visible_states, min=1e-6)
        log_state = torch.log(safe)

        # Log-ratio (actual)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, min=self.clamp_min, max=self.clamp_max)

        # Visible-only baseline prediction
        # log_ratio_baseline(t) = r5 + A55 @ x_t
        x_current = visible_states[:, :-1]  # (B, T-1, N)
        baseline_log_ratio = self.r5.view(1, 1, -1) + torch.einsum("btn,mn->btm", x_current, self.A55)
        # Add bias
        baseline_log_ratio = baseline_log_ratio + self.bias5.view(1, 1, -1)

        # Residual: signature of hidden
        residual = actual_log_ratio - baseline_log_ratio  # (B, T-1, N)
        # Pad residual with 0 at time 0 to align with T
        residual_padded = F.pad(residual, (0, 0, 1, 0), value=0.0)  # (B, T, N)

        # Takens delay embedding of state: lag 1, 2, ..., delay_steps
        state_delays = []
        for lag in range(1, self.delay_steps + 1):
            # lag=k: take x(t-k)
            padded = F.pad(visible_states, (0, 0, lag, 0), value=0.0)[:, :T]
            state_delays.append(padded)
        state_delay_stack = torch.stack(state_delays, dim=-1)  # (B, T, N, delay_steps)

        # Takens delay of residual
        residual_delays = []
        for lag in range(1, self.delay_steps + 1):
            padded = F.pad(residual_padded, (0, 0, lag, 0), value=0.0)[:, :T]
            residual_delays.append(padded)
        residual_delay_stack = torch.stack(residual_delays, dim=-1)  # (B, T, N, delay_steps)

        # Species one-hot (broadcast)
        species_id = torch.eye(N, device=device)  # (N, N)
        species_id = species_id.view(1, 1, N, N).expand(B, T, N, N)

        # Concatenate features
        # state: (B, T, N) → (B, T, N, 1)
        features = torch.cat([
            visible_states.unsqueeze(-1),
            log_state.unsqueeze(-1),
            state_delay_stack,                # (B, T, N, delay_steps)
            residual_delay_stack,              # (B, T, N, delay_steps)
            species_id,                        # (B, T, N, N)
        ], dim=-1)
        return features, residual

    def forward(self, visible_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Recover hidden from visible observations.

        Args:
          visible_states: (B, T, N) or (T, N)

        Returns:
          {
            "hidden": (B, T) recovered hidden time series
            "reconstructed_log_ratio": (B, T-1, N) predicted visible log-ratio
            "actual_log_ratio": (B, T-1, N)
            "baseline_log_ratio": (B, T-1, N)
            "residual": (B, T-1, N) baseline residual
          }
        """
        if visible_states.dim() == 2:
            visible_states = visible_states.unsqueeze(0)
        B, T, N = visible_states.shape

        # Build features
        features, residual = self._build_features(visible_states)

        # Encode features → d_model
        h = self.feat_encoder(features)  # (B, T, N, d_model)

        # Add positional encoding (along time)
        # h is (B, T, N, d_model), need to add pe on time axis
        pe = self.pos_encoding.pe[:T]  # (T, d_model)
        h = h + pe.view(1, T, 1, self.d_model)

        # Apply spatio-temporal blocks
        for block in self.blocks:
            h = block(h)  # (B, T, N, d_model)

        # Aggregate to hidden: flatten species dim then MLP
        h_flat = h.reshape(B, T, N * self.d_model)  # (B, T, N*D)
        hidden_raw = self.hidden_decoder(h_flat).squeeze(-1)  # (B, T)
        # Constrain to positive (hidden 是物种丰度)
        hidden = F.softplus(hidden_raw) + 0.01

        # Forward simulation: use recovered hidden to predict next visible
        # predicted_log_ratio(t) = baseline(t) + b5 * h(t) + c5 * h(t)^2
        x_current = visible_states[:, :-1]  # (B, T-1, N)
        h_current = hidden[:, :-1]           # (B, T-1)
        baseline_log_ratio = self.r5.view(1, 1, -1) + torch.einsum("btn,mn->btm", x_current, self.A55)
        baseline_log_ratio = baseline_log_ratio + self.bias5.view(1, 1, -1)
        hidden_coupling = h_current.unsqueeze(-1) * self.b5.view(1, 1, -1)
        hidden_coupling_quad = (h_current.unsqueeze(-1) ** 2) * self.c5.view(1, 1, -1)
        reconstructed_log_ratio = baseline_log_ratio + hidden_coupling + hidden_coupling_quad

        # Actual log-ratio for comparison
        safe = torch.clamp(visible_states, min=1e-6)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, min=self.clamp_min, max=self.clamp_max)

        return {
            "hidden": hidden,                                 # (B, T)
            "reconstructed_log_ratio": reconstructed_log_ratio,  # (B, T-1, N)
            "actual_log_ratio": actual_log_ratio,              # (B, T-1, N)
            "baseline_log_ratio": baseline_log_ratio,         # (B, T-1, N)
            "residual": residual,                             # (B, T-1, N)
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        smooth_lambda: float = 0.01,
        sparse_lambda: float = 0.001,
        var_lambda: float = 0.1,
    ) -> Dict[str, torch.Tensor]:
        """Compute training loss.

        Only uses visible reconstruction — NO hidden supervision.
        """
        actual = outputs["actual_log_ratio"]
        reconstructed = outputs["reconstructed_log_ratio"]
        hidden = outputs["hidden"]

        # Main reconstruction loss
        fit_loss = F.mse_loss(reconstructed, actual)

        # Hidden smoothness (2nd difference)
        smooth_loss = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()

        # Sparse A55 off-diagonal
        A_offdiag = self.A55 - torch.diag(torch.diag(self.A55))
        sparse_loss = A_offdiag.abs().mean()

        # Hidden variance normalization (identifiability: prevent hidden collapse)
        # 鼓励 hidden 有 non-trivial variance
        h_var = hidden.var(dim=-1).mean()
        var_loss = F.relu(0.05 - h_var)  # penalize if variance too small

        total = fit_loss + smooth_lambda * smooth_loss + sparse_lambda * sparse_loss + var_lambda * var_loss

        return {
            "total": total,
            "fit": fit_loss,
            "smooth": smooth_loss,
            "sparse": sparse_loss,
            "var": var_loss,
            "h_variance": h_var.detach(),
        }

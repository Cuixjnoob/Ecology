"""Linear-Seeded GNN: Linear sparse 作为 GNN 的结构 anchor, GNN 学 residual correction.

核心设计（回应用户洞察: Linear 应该作为 GNN 的插件/anchor, 不是限制 GNN）:

架构:
  Layer 0: Linear Sparse Baseline
    log_ratio_linear(t, i) = r_i + A[i,:]·x_t + intercept_i
    (L1 正则强制 sparse)

  Input features per (t, n):
    [x, log_x, linear_pred, linear_residual, Takens of residual]
    这些特征让每个节点知道 "linear 预测了什么" 和 "linear 还没解释什么"

  GNN (full capacity):
    Deep spatio-temporal transformer (d=128, 4 blocks, 8 heads)
    作用在 input features 上，学 refined residual

  Hidden decoder:
    读出 hidden 时序

  Final prediction:
    log_ratio_total = linear_pred + α · gnn_output + hidden_coupling
    α 是可学习的 gate (soft identity init)

  Loss:
    MSE(total, actual) + L1(A) + smoothness(hidden) + L2(gnn_output)

为什么这样不会 shortcut learning:
  1. Linear 的 L1 强制 sparse, 不吞 hidden signal
  2. GNN 的 output 被 L2 regularized (keep correction small)
  3. Hidden coupling 是 linear 形式 (bounded contribution)
  4. α gate 初始化为小值, 训练中慢慢打开
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


class LinearBaseline(nn.Module):
    """Linear sparse baseline: r + A·x"""
    def __init__(self, num_visible):
        super().__init__()
        self.r = nn.Parameter(0.05 * torch.ones(num_visible))
        A_init = 0.01 * torch.randn(num_visible, num_visible)
        A_init.fill_diagonal_(-0.2)
        self.A = nn.Parameter(A_init)

    def forward(self, x_t):
        # x_t: (B, T-1, N)
        return self.r.view(1, 1, -1) + x_t @ self.A.T

    def l1_reg(self):
        A_off = self.A - torch.diag(torch.diag(self.A))
        return A_off.abs().mean()


class SpatioTemporalBlock(nn.Module):
    def __init__(self, d_model, num_heads=8, dropout=0.1):
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


class LinearSeededGNN(nn.Module):
    """Linear-Seeded GNN:
    Linear sparse baseline + deep GNN residual corrector + hidden decoder.

    Linear 是 GNN 的结构 anchor (作 input feature), GNN 学 refined residual.
    """
    def __init__(
        self,
        num_visible: int = 5,
        takens_lags: List[int] = (1, 2, 4, 8),
        d_model: int = 128,
        num_blocks: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        clamp_min: float = -1.12,
        clamp_max: float = 0.92,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.takens_lags = list(takens_lags)
        self.d_model = d_model
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Layer 0: Linear sparse baseline (L1 regularized)
        self.linear = LinearBaseline(num_visible)

        # Input features per (t, n):
        # [x, log_x, linear_pred, linear_residual, Takens of residual, Takens of x]
        feat_dim = 4 + 2 * len(self.takens_lags)

        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.species_emb = nn.Parameter(torch.randn(num_visible, d_model) * 0.1)

        # Deep spatio-temporal GNN
        self.blocks = nn.ModuleList([
            SpatioTemporalBlock(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])

        # Output 1: Residual correction per species (for visible reconstruction)
        self.correction_head = nn.Linear(d_model, 1)
        # Learnable gate for GNN correction (init near 0 so linear dominates early)
        self.alpha_raw = nn.Parameter(torch.tensor(-2.0))  # sigmoid(-2) ≈ 0.12

        # Output 2: Hidden decoder (per time step, aggregating all species)
        self.hidden_readout = nn.Sequential(
            nn.Linear(d_model * num_visible, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

        # Hidden coupling (simple linear + quadratic)
        self.b = nn.Parameter(0.1 * torch.randn(num_visible))
        self.c_quad = nn.Parameter(0.01 * torch.randn(num_visible))

    def get_alpha(self):
        return torch.sigmoid(self.alpha_raw)

    def forward(self, visible_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        if visible_states.dim() == 2:
            visible_states = visible_states.unsqueeze(0)
        B, T, N = visible_states.shape

        # Layer 0: Linear prediction + residual
        x_current = visible_states[:, :-1]  # (B, T-1, N)
        linear_pred = self.linear(x_current)  # (B, T-1, N)
        safe = torch.clamp(visible_states, min=1e-6)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, self.clamp_min, self.clamp_max)
        linear_residual = actual_log_ratio - linear_pred  # (B, T-1, N)

        # Pad to T for feature construction
        linear_pred_T = F.pad(linear_pred, (0, 0, 0, 1), value=0.0)  # (B, T, N) - pad last
        linear_residual_T = F.pad(linear_residual, (0, 0, 1, 0), value=0.0)  # (B, T, N) - pad first
        # At t=0, linear_residual is unknown, so use 0

        # Takens of residual
        residual_takens = []
        for lag in self.takens_lags:
            padded = F.pad(linear_residual_T.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            residual_takens.append(padded.unsqueeze(-1))  # (B, T, N, 1)

        # Takens of x
        x_takens = []
        for lag in self.takens_lags:
            padded = F.pad(visible_states.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            x_takens.append(padded.unsqueeze(-1))  # (B, T, N, 1)

        # Feature per (t, n): [x, log_x, linear_pred, residual, residual_takens, x_takens]
        log_v = torch.log(safe)
        features = torch.cat([
            visible_states.unsqueeze(-1),         # (B, T, N, 1)
            log_v.unsqueeze(-1),                   # (B, T, N, 1)
            linear_pred_T.unsqueeze(-1),           # (B, T, N, 1)
            linear_residual_T.unsqueeze(-1),       # (B, T, N, 1)
            torch.cat(residual_takens, dim=-1),    # (B, T, N, L)
            torch.cat(x_takens, dim=-1),           # (B, T, N, L)
        ], dim=-1)  # (B, T, N, feat_dim)

        h = self.input_proj(features) + self.species_emb.view(1, 1, N, -1)
        pe = sinusoidal_pe(T, self.d_model, h.device)
        h = h + pe.view(1, T, 1, self.d_model)

        for block in self.blocks:
            h = block(h)  # (B, T, N, d_model)

        # Output: residual correction per species
        correction = self.correction_head(h).squeeze(-1)  # (B, T, N)
        correction_current = correction[:, :-1]  # (B, T-1, N)
        alpha = self.get_alpha()

        # Output: Hidden time series
        h_flat = h.reshape(B, T, N * self.d_model)
        hidden_raw = self.hidden_readout(h_flat).squeeze(-1)  # (B, T)
        hidden = F.softplus(hidden_raw) + 0.01  # positive

        # Total prediction: linear + α * GNN correction + hidden coupling
        h_current = hidden[:, :-1]
        hidden_linear = h_current.unsqueeze(-1) * self.b.view(1, 1, -1)
        hidden_quad = (h_current.unsqueeze(-1) ** 2) * self.c_quad.view(1, 1, -1)
        predicted_log_ratio = linear_pred + alpha * correction_current + hidden_linear + hidden_quad

        return {
            "hidden": hidden,
            "linear_pred": linear_pred,
            "linear_residual": linear_residual,
            "gnn_correction": correction_current,
            "alpha": alpha.detach(),
            "predicted_log_ratio": predicted_log_ratio,
            "actual_log_ratio": actual_log_ratio,
        }

    def compute_loss(self, outputs, lam_A: float = 0.3, lam_smooth: float = 0.02,
                      lam_correction: float = 0.01, lam_var: float = 0.15):
        actual = outputs["actual_log_ratio"]
        pred = outputs["predicted_log_ratio"]
        hidden = outputs["hidden"]
        correction = outputs["gnn_correction"]

        fit = F.mse_loss(pred, actual)
        # L1 on linear A (sparse ecological structure)
        l1_A = self.linear.l1_reg()
        # Smoothness on hidden
        smooth = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        # Keep GNN correction small (L2)
        correction_mag = correction.square().mean()
        # Variance floor on hidden
        h_var = hidden.var(dim=-1).mean()
        var_loss = F.relu(0.05 - h_var)

        total = fit + lam_A * l1_A + lam_smooth * smooth + lam_correction * correction_mag + lam_var * var_loss
        return {
            "total": total, "fit": fit, "l1_A": l1_A, "smooth": smooth,
            "correction_mag": correction_mag, "var": var_loss,
            "h_variance": h_var.detach(), "alpha": outputs["alpha"],
        }

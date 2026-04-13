"""HNSR: Hybrid Neural-Sparse Recovery.

融合 Linear Sparse Baseline + SINDy-like Library + GNN Discovery + GNN Correction。

架构:
  输入: visible (T, N)

  Stage 1: Linear Sparse Baseline (LSB) — 提供 coarse h
    log_ratio_linear = r_lin + A_lin · x  (L1 正则)
    residual_lsb = actual - log_ratio_linear
    h_coarse = GNN_recover(residual_lsb, visible, Takens)  # 粗恢复

  Stage 2: GNN-enhanced Library
    learned_basis_features = GNN_basis(visible)  # GNN 主动发现新特征
    Combined library: [hand_crafted LV/Holling/Allee terms, learned_basis]
    Sparse regression on combined library → predicts log_ratio_library

  Stage 3: GNN Scale Corrector
    gnn_input = concat(visible, residual_lsb, h_coarse, Takens, learned_basis)
    delta_h = GNN_correction(gnn_input)  # small correction
    h_final = h_coarse + alpha * delta_h

  Stage 4: Final prediction with hidden
    predicted_log_ratio = log_ratio_library + b * h_final + c * h_final^2

  Loss (无 hidden 监督):
    MSE(predicted, actual)
    + L1 on A_lin
    + L1 on library coefficients
    + L2 on delta_h (keep correction small)
    + smoothness on h_final

创新:
  1. 多 stream 融合（linear, library, GNN basis, GNN correction）
  2. GNN 主动发现 basis 增强 library，不只是 select
  3. GNN 只做 correction 利用 LSB 强 prior 防过拟合
  4. 分层 identifiability 约束
"""
from __future__ import annotations

from typing import Dict, List, Tuple

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
# Hand-crafted library (LV, Holling, Allee 等)
# =============================================================================
def build_crafted_library(x: torch.Tensor, half_sat: float = 0.45, allee_thresh: float = 0.15) -> torch.Tensor:
    """Hand-crafted basis functions.
    x: (B, T, N)
    Returns: (B, T, N, L_crafted) per-species basis (NOT per-species-pair, for efficiency)

    Each species gets its own set of basis values:
      φ_0: x_i itself (identity, for A · x type linear)
      φ_1: log(x_i)
      φ_2: x_i² (quadratic self-limit)
      φ_3: x_i / (K + x_i) (Holling II saturating self)
      φ_4: (x_i - A) / (K + x_i) (Allee form)
      φ_5: log(1 + x_i) (log-like)
    """
    safe = torch.clamp(x, min=1e-6)
    log_x = torch.log(safe)
    basis = torch.stack([
        x,
        log_x,
        x ** 2,
        x / (half_sat + x),
        (x - allee_thresh) / (0.5 + x),
        torch.log1p(x),
    ], dim=-1)  # (B, T, N, 6)
    return basis


# =============================================================================
# Sparse Attention GNN block
# =============================================================================
class SparseGATLayer(nn.Module):
    def __init__(self, d_in: int, d_out: int, num_heads: int = 2, top_k: int = 3):
        super().__init__()
        self.num_heads = num_heads
        self.d_head = d_out // num_heads
        self.top_k = top_k
        self.q = nn.Linear(d_in, d_out)
        self.k = nn.Linear(d_in, d_out)
        self.v = nn.Linear(d_in, d_out)
        self.out = nn.Linear(d_out, d_out)

    def forward(self, x):
        # x: (B_total, N, d_in)
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
        return self.out(out)


# =============================================================================
# GNN-based Basis Discovery
# =============================================================================
class GNNBasisDiscoverer(nn.Module):
    """GNN 生成 learned basis features (for extending library)."""
    def __init__(self, num_visible: int, num_learned_basis: int = 6, d_hidden: int = 32,
                 num_layers: int = 2, num_heads: int = 2, top_k: int = 3, dropout: float = 0.1):
        super().__init__()
        self.num_visible = num_visible
        self.num_learned_basis = num_learned_basis

        # Feature encoding: [x, log_x, sqrt_x] → d_hidden
        self.input_proj = nn.Sequential(
            nn.Linear(3, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.species_emb = nn.Parameter(torch.randn(num_visible, d_hidden) * 0.1)
        self.layers = nn.ModuleList([
            SparseGATLayer(d_hidden, d_hidden, num_heads=num_heads, top_k=top_k)
            for _ in range(num_layers)
        ])
        self.norms = nn.ModuleList([nn.LayerNorm(d_hidden) for _ in range(num_layers)])
        self.dropout = nn.Dropout(dropout)
        # Output: per-species learned basis values
        self.basis_head = nn.Linear(d_hidden, num_learned_basis)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N)
        Returns: (B, T, N, num_learned_basis)
        """
        B, T, N = x.shape
        safe = torch.clamp(x, min=1e-6)
        feats = torch.stack([x, torch.log(safe), torch.sqrt(safe)], dim=-1)  # (B, T, N, 3)
        h = self.input_proj(feats) + self.species_emb.view(1, 1, N, -1)
        # Flatten T into batch
        h_flat = h.reshape(B * T, N, -1)
        for layer, norm in zip(self.layers, self.norms):
            h_flat = norm(h_flat + self.dropout(layer(h_flat)))
        basis = self.basis_head(h_flat)  # (B*T, N, num_learned_basis)
        return basis.reshape(B, T, N, -1)


# =============================================================================
# GNN Correction for Hidden
# =============================================================================
class HiddenSpatioTemporalBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.species_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.temporal_attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
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


class HiddenCorrectionGNN(nn.Module):
    """Small GNN for hidden correction (output scalar per time step)."""
    def __init__(self, num_visible: int, feat_dim: int, d_model: int = 64,
                 num_blocks: int = 2, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.num_visible = num_visible
        self.d_model = d_model
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model), nn.LayerNorm(d_model),
        )
        self.species_emb = nn.Parameter(torch.randn(num_visible, d_model) * 0.1)
        self.blocks = nn.ModuleList([
            HiddenSpatioTemporalBlock(d_model, num_heads=num_heads, dropout=dropout)
            for _ in range(num_blocks)
        ])
        self.readout = nn.Sequential(
            nn.Linear(d_model * num_visible, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """features: (B, T, N, feat_dim) → (B, T) scalar correction"""
        B, T, N, _ = features.shape
        h = self.input_proj(features) + self.species_emb.view(1, 1, N, -1)
        pe = sinusoidal_pe(T, self.d_model, h.device)
        h = h + pe.view(1, T, 1, self.d_model)
        for block in self.blocks:
            h = block(h)
        h_flat = h.reshape(B, T, N * self.d_model)
        delta = self.readout(h_flat).squeeze(-1)
        return delta


# =============================================================================
# Full HNSR Model
# =============================================================================
class HNSR(nn.Module):
    """Hybrid Neural-Sparse Recovery.

    融合:
      1. Linear Sparse Baseline (core coarse)
      2. Hand-crafted Library (LV/Holling/Allee)
      3. GNN Basis Discovery (learned basis, extends library)
      4. Library sparse regression
      5. GNN-based Hidden Correction
    """
    def __init__(
        self,
        num_visible: int = 5,
        num_learned_basis: int = 6,
        takens_lags: List[int] = (1, 2, 4, 8),
        # Basis discovery GNN
        basis_gnn_d: int = 32,
        basis_gnn_layers: int = 2,
        basis_gnn_top_k: int = 3,
        # Correction GNN
        correction_d: int = 64,
        correction_blocks: int = 2,
        correction_heads: int = 4,
        # Hidden correction magnitude
        correction_scale: float = 1.0,
        dropout: float = 0.1,
        half_sat: float = 0.45,
        allee_thresh: float = 0.15,
        clamp_min: float = -1.12,
        clamp_max: float = 0.92,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_learned_basis = num_learned_basis
        self.takens_lags = takens_lags
        self.correction_scale = correction_scale
        self.half_sat = half_sat
        self.allee_thresh = allee_thresh
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Linear sparse baseline parameters (always active)
        self.r_linear = nn.Parameter(0.1 * torch.ones(num_visible))
        A_init = 0.02 * torch.randn(num_visible, num_visible)
        A_init.fill_diagonal_(-0.2)
        self.A_linear = nn.Parameter(A_init)

        # Hand-crafted library coefficients (per species)
        # Init to 0 so linear baseline dominates early, then slowly activate
        self.c_crafted = nn.Parameter(torch.zeros(num_visible, 6))

        # GNN basis discoverer
        self.basis_gnn = GNNBasisDiscoverer(
            num_visible=num_visible,
            num_learned_basis=num_learned_basis,
            d_hidden=basis_gnn_d,
            num_layers=basis_gnn_layers,
            top_k=basis_gnn_top_k,
            dropout=dropout,
        )
        # Learned basis coefficients (per species) — init 0 so GNN starts inactive
        self.c_learned = nn.Parameter(torch.zeros(num_visible, num_learned_basis))

        # Hidden correction GNN
        # Input features per (t, n): [x, log_x, residual, learned_basis, Takens of x]
        correction_feat_dim = 3 + num_learned_basis + len(takens_lags)
        self.correction_gnn = HiddenCorrectionGNN(
            num_visible=num_visible,
            feat_dim=correction_feat_dim,
            d_model=correction_d,
            num_blocks=correction_blocks,
            num_heads=correction_heads,
            dropout=dropout,
        )

        # Coarse hidden estimate head (linear combination of residuals)
        # Init with non-trivial weights so hidden path is active from start
        self.coarse_h_head = nn.Linear(num_visible, 1)
        with torch.no_grad():
            self.coarse_h_head.weight.data = 0.2 * torch.randn(1, num_visible)
            self.coarse_h_head.bias.data.zero_()

        # Hidden coupling
        self.b = nn.Parameter(0.1 * torch.randn(num_visible))
        self.c_quad = nn.Parameter(0.01 * torch.randn(num_visible))

    def _linear_baseline(self, x_current: torch.Tensor) -> torch.Tensor:
        """Linear sparse baseline: r + A·x"""
        return self.r_linear.view(1, 1, -1) + x_current @ self.A_linear.T

    def _library_baseline(self, x_current: torch.Tensor) -> torch.Tensor:
        """Hand-crafted library + GNN-learned basis combined."""
        B, T, N = x_current.shape
        crafted = build_crafted_library(x_current, self.half_sat, self.allee_thresh)  # (B, T, N, 6)
        learned = self.basis_gnn(x_current)  # (B, T, N, num_learned_basis)
        # Per-species: crafted × c_crafted + learned × c_learned
        pred_crafted = torch.einsum("btnk,nk->btn", crafted, self.c_crafted)
        pred_learned = torch.einsum("btnk,nk->btn", learned, self.c_learned)
        return pred_crafted + pred_learned, learned

    def forward(self, visible_states: torch.Tensor) -> Dict[str, torch.Tensor]:
        if visible_states.dim() == 2:
            visible_states = visible_states.unsqueeze(0)
        B, T, N = visible_states.shape

        x_current = visible_states[:, :-1]  # (B, T-1, N)

        # --- Stage 1: Linear sparse baseline ---
        linear_log_ratio = self._linear_baseline(x_current)  # (B, T-1, N)

        # --- Stage 2: Extended library (crafted + GNN-learned) ---
        library_log_ratio, learned_basis = self._library_baseline(x_current)  # (B, T-1, N), (B, T-1, N, K)

        # Combined baseline
        baseline_log_ratio = linear_log_ratio + library_log_ratio

        # Compute actual log-ratio and residual
        safe = torch.clamp(visible_states, min=1e-6)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, self.clamp_min, self.clamp_max)
        residual = actual_log_ratio - baseline_log_ratio  # (B, T-1, N)

        # --- Stage 3: Coarse hidden estimate from residual (linear combo) ---
        h_coarse = self.coarse_h_head(residual).squeeze(-1)  # (B, T-1)
        # Pad to T
        h_coarse_T = F.pad(h_coarse, (1, 0), value=0.0)  # (B, T)

        # --- Stage 4: Build features for correction GNN ---
        safe_full = torch.clamp(visible_states, min=1e-6)
        log_v = torch.log(safe_full)
        residual_T = F.pad(residual, (0, 0, 1, 0), value=0.0)  # (B, T, N)
        learned_basis_T = F.pad(learned_basis, (0, 0, 0, 0, 1, 0), value=0.0)  # (B, T, N, K)

        # Takens for visible
        takens_feats = []
        for lag in self.takens_lags:
            padded = F.pad(visible_states.permute(0, 2, 1), (lag, 0), value=0.0)[..., :T].permute(0, 2, 1)
            takens_feats.append(padded.unsqueeze(-1))
        takens_stack = torch.cat(takens_feats, dim=-1)  # (B, T, N, L_takens)

        # Concat features per (t, n): [x, log_x, residual, learned_basis, Takens]
        features = torch.cat([
            visible_states.unsqueeze(-1),
            log_v.unsqueeze(-1),
            residual_T.unsqueeze(-1),
            learned_basis_T,
            takens_stack,
        ], dim=-1)  # (B, T, N, feat_dim)

        # --- Stage 5: GNN correction ---
        delta_h = self.correction_gnn(features)  # (B, T)
        # Combine coarse + correction
        h_raw = h_coarse_T + self.correction_scale * delta_h
        hidden = F.softplus(h_raw) + 0.01

        # --- Stage 6: Forward prediction with hidden ---
        h_current = hidden[:, :-1]  # (B, T-1)
        hidden_term = h_current.unsqueeze(-1) * self.b.view(1, 1, -1)
        hidden_quad = (h_current.unsqueeze(-1) ** 2) * self.c_quad.view(1, 1, -1)
        predicted_log_ratio = baseline_log_ratio + hidden_term + hidden_quad

        return {
            "hidden": hidden,
            "h_coarse": h_coarse_T,
            "delta_h": delta_h,
            "linear_log_ratio": linear_log_ratio,
            "library_log_ratio": library_log_ratio,
            "baseline_log_ratio": baseline_log_ratio,
            "predicted_log_ratio": predicted_log_ratio,
            "actual_log_ratio": actual_log_ratio,
            "residual": residual,
            "learned_basis": learned_basis,
        }

    def compute_loss(
        self,
        outputs: Dict[str, torch.Tensor],
        lam_A_sparse: float = 0.3,
        lam_crafted_sparse: float = 0.05,
        lam_learned_sparse: float = 0.05,
        lam_smooth: float = 0.02,
        lam_correction_mag: float = 0.05,
        lam_var: float = 0.15,
    ) -> Dict[str, torch.Tensor]:
        actual = outputs["actual_log_ratio"]
        pred = outputs["predicted_log_ratio"]
        hidden = outputs["hidden"]
        delta_h = outputs["delta_h"]

        fit = F.mse_loss(pred, actual)
        # L1 on A_linear (sparse baseline)
        A_off = self.A_linear - torch.diag(torch.diag(self.A_linear))
        sparse_A = A_off.abs().mean()
        # L1 on crafted coefficients
        sparse_crafted = self.c_crafted.abs().mean()
        # L1 on learned basis coefficients
        sparse_learned = self.c_learned.abs().mean()
        # Hidden smoothness
        smooth = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        # Correction magnitude (keep small)
        correction_mag = delta_h.square().mean()
        # Variance floor
        h_var = hidden.var(dim=-1).mean()
        var_loss = F.relu(0.05 - h_var)

        total = (fit
                 + lam_A_sparse * sparse_A
                 + lam_crafted_sparse * sparse_crafted
                 + lam_learned_sparse * sparse_learned
                 + lam_smooth * smooth
                 + lam_correction_mag * correction_mag
                 + lam_var * var_loss)

        return {
            "total": total, "fit": fit,
            "sparse_A": sparse_A, "sparse_crafted": sparse_crafted, "sparse_learned": sparse_learned,
            "smooth": smooth, "correction_mag": correction_mag, "var": var_loss,
            "h_variance": h_var.detach(),
        }

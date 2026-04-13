"""UltraSparse GNN Baseline — 容量和 linear sparse 同量级的 GNN baseline。

核心设计原则:
  - GNN baseline 必须受限到 linear sparse 同一量级（几百参数）
  - 但比 linear sparse 更 flexible（能拟合 Holling、Allee 等非线性）
  - L1 正则 + 稀疏 attention 保持 identifiability

参数预算:
  - Linear sparse baseline: ~35 params (5 r + 25 A + 5 bias)
  - UltraSparse GNN: ~几百 params (2 layer × tiny GAT + tiny head)

架构:
  Input: x_i, log(x_i) per species (2 features)
  ↓ Linear to d=16 (per species embedding)
  ↓ GAT layer 1: top_k=2, heads=1  (receive aggregate from 2 neighbors)
  ↓ GAT layer 2: top_k=2, heads=1
  ↓ Linear to 1 (per species log_ratio prediction)

总参数: ~几百

这提供了 Holling/Allee-like 非线性的捕捉能力，同时保持 identifiability。
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F


class TinyGATLayer(nn.Module):
    """Tiny GAT layer with hard top-k sparsity."""
    def __init__(self, d_in: int, d_out: int, top_k: int = 2):
        super().__init__()
        self.d_out = d_out
        self.top_k = top_k
        # Single head, no bias
        self.w = nn.Linear(d_in, d_out, bias=False)
        self.a = nn.Linear(2 * d_out, 1, bias=False)  # attention score

    def forward(self, x):
        # x: (B, N, d_in)
        B, N, _ = x.shape
        h = self.w(x)  # (B, N, d_out)
        # Pairwise features
        h_i = h.unsqueeze(2).expand(-1, -1, N, -1)  # (B, N, N, d_out)
        h_j = h.unsqueeze(1).expand(-1, N, -1, -1)
        pair = torch.cat([h_i, h_j], dim=-1)  # (B, N, N, 2*d_out)
        scores = self.a(pair).squeeze(-1)  # (B, N, N)

        # Top-k mask
        k_eff = min(self.top_k, N)
        topk_vals, topk_idx = scores.topk(k_eff, dim=-1)
        mask = torch.full_like(scores, float("-inf"))
        mask.scatter_(-1, topk_idx, topk_vals)
        attn = F.softmax(mask, dim=-1)
        # Aggregate
        out = torch.einsum("bnm,bmd->bnd", attn, h)
        return out


class UltraSparseGNNBaseline(nn.Module):
    """Tiny GNN baseline: ~几百参数, able to capture Holling-like nonlinearity."""
    def __init__(self, num_visible: int = 5, d_hidden: int = 16, num_layers: int = 2, top_k: int = 2):
        super().__init__()
        self.num_visible = num_visible
        # Input features: [x, log(x)] per species  → d_hidden
        self.input_proj = nn.Linear(2, d_hidden, bias=False)
        # Species embedding (distinguishes species as graph nodes)
        self.species_emb = nn.Parameter(torch.randn(num_visible, d_hidden) * 0.1)
        # GAT layers
        self.gat_layers = nn.ModuleList([
            TinyGATLayer(d_hidden, d_hidden, top_k=top_k) for _ in range(num_layers)
        ])
        # Output: scalar log_ratio per species
        self.output = nn.Linear(d_hidden, 1, bias=True)
        # Intrinsic growth (per species)
        self.r_intrinsic = nn.Parameter(torch.zeros(num_visible))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, T, N) visible states
        Returns: (B, T, N) predicted log_ratio
        """
        B, T, N = x.shape
        safe = torch.clamp(x, min=1e-6)
        feats = torch.stack([x, torch.log(safe)], dim=-1)  # (B, T, N, 2)
        h = self.input_proj(feats) + self.species_emb.view(1, 1, N, -1)  # (B, T, N, d_hidden)

        # Flatten T into batch for per-time graph
        h_flat = h.reshape(B * T, N, -1)  # (B*T, N, d_hidden)
        for layer in self.gat_layers:
            h_flat = h_flat + layer(h_flat)  # residual
        # Output
        out = self.output(h_flat).squeeze(-1)  # (B*T, N)
        out = out.reshape(B, T, N) + self.r_intrinsic.view(1, 1, -1)
        return out

    def l1_reg(self) -> torch.Tensor:
        reg = 0.0
        for m in [self.input_proj, self.output]:
            reg = reg + m.weight.abs().mean()
        for layer in self.gat_layers:
            for mod in [layer.w, layer.a]:
                reg = reg + mod.weight.abs().mean()
        return reg


class UltraSparseHiddenModel(nn.Module):
    """UltraSparse GNN baseline + residual-based hidden recovery + coupling.

    所有 components 都 ultra-sparse / ultra-limited, 保持 identifiability。
    """
    def __init__(self, num_visible: int = 5, d_hidden: int = 16, num_layers: int = 2, top_k: int = 2,
                 clamp_min: float = -1.12, clamp_max: float = 0.92):
        super().__init__()
        self.num_visible = num_visible
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        # Tiny GNN baseline
        self.baseline = UltraSparseGNNBaseline(num_visible, d_hidden, num_layers, top_k)

        # Linear hidden recovery head from residual (simple, ultra-sparse)
        self.h_head = nn.Linear(num_visible, 1, bias=True)
        # Hidden coupling
        self.b = nn.Parameter(0.1 * torch.randn(num_visible))
        self.c = nn.Parameter(0.01 * torch.randn(num_visible))

    def forward(self, visible: torch.Tensor) -> Dict[str, torch.Tensor]:
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape

        x_current = visible[:, :-1]
        baseline_log_ratio = self.baseline(x_current)  # (B, T-1, N)

        safe = torch.clamp(visible, min=1e-6)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, self.clamp_min, self.clamp_max)
        residual = actual_log_ratio - baseline_log_ratio  # (B, T-1, N)

        # Simple linear hidden head from residual
        h_raw = self.h_head(residual).squeeze(-1)  # (B, T-1)
        hidden = F.softplus(h_raw) + 0.01  # positive

        # Cycle
        h_current = hidden
        hidden_term = h_current.unsqueeze(-1) * self.b.view(1, 1, -1)
        hidden_quad = (h_current.unsqueeze(-1) ** 2) * self.c.view(1, 1, -1)
        predicted_log_ratio = baseline_log_ratio + hidden_term + hidden_quad

        return {
            "hidden": hidden,  # (B, T-1)
            "baseline_log_ratio": baseline_log_ratio,
            "actual_log_ratio": actual_log_ratio,
            "predicted_log_ratio": predicted_log_ratio,
            "residual": residual,
        }

    def compute_loss(self, outputs, lam_l1=0.05, lam_smooth=0.02, lam_var=0.15):
        actual = outputs["actual_log_ratio"]
        pred = outputs["predicted_log_ratio"]
        hidden = outputs["hidden"]
        fit = F.mse_loss(pred, actual)
        l1 = self.baseline.l1_reg()
        smooth = ((hidden[:, 2:] - 2 * hidden[:, 1:-1] + hidden[:, :-2]) ** 2).mean()
        h_var = hidden.var(dim=-1).mean()
        var_loss = F.relu(0.05 - h_var)
        total = fit + lam_l1 * l1 + lam_smooth * smooth + lam_var * var_loss
        return {"total": total, "fit": fit, "l1": l1, "smooth": smooth, "var": var_loss, "h_variance": h_var.detach()}

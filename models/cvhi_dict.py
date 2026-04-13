"""CVHI-Dict: CVHI with SINDy-style dictionary-based DynamicsOperator.

Architecture:
  - PosteriorEncoder (GNN + Takens): same as CVHI, 但 output k-dim H
  - DictionaryDynamicsOperator: basis function library + learnable sparse coefficients
  - ELBO loss + L1 on dictionary coefficients → automatic equation discovery

Per-species dictionary Φ_i(x, H):
  1 const, 1 self-linear, 1 self-quadratic, 1 self-Holling-II, 1 self-log,
  (N-1) pairwise linear,
  k hidden-linear, k hidden·self, k hidden-quadratic
  Total D ~= 5 + (N-1) + 3k
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.cvhi import PosteriorEncoder


# =============================================================================
# Multi-channel Posterior Encoder Wrapper
# =============================================================================
class MultiChannelPosteriorEncoder(nn.Module):
    """Wraps k copies of PosteriorEncoder to produce k-dim H.

    Each channel independently outputs (μ_k, log_σ_k), giving H ∈ R^{B, T, k}.
    Different seed initialization encourages channel differentiation.
    """
    def __init__(self, k_hidden: int = 1, **encoder_kwargs):
        super().__init__()
        self.k_hidden = k_hidden
        self.encoders = nn.ModuleList([
            PosteriorEncoder(**encoder_kwargs) for _ in range(k_hidden)
        ])

    def forward(self, visible):
        mus, log_sigmas = [], []
        for enc in self.encoders:
            mu, log_sigma = enc(visible)  # (B, T) each
            mus.append(mu)
            log_sigmas.append(log_sigma)
        mu = torch.stack(mus, dim=-1)         # (B, T, k)
        log_sigma = torch.stack(log_sigmas, dim=-1)  # (B, T, k)
        return mu, log_sigma


# =============================================================================
# Dictionary Dynamics Operator
# =============================================================================
class DictionaryDynamicsOperator(nn.Module):
    """Basis dictionary + learnable sparse coefficient.

    Predict log(x_{t+1}/x_t) for each visible species using:
      log_ratio[t, i] = Σ_d C[i, d] · Φ_d(x_t, H_t; i)

    Basis Φ (per species i):
      [const, x_i, x_i², x_i/(1+x_i), log(x_i+1),
       x_{j1}, x_{j2}, ..., x_{j N-1},
       H_1, H_2, ..., H_k,
       H_1·x_i, H_2·x_i, ..., H_k·x_i,
       H_1², H_2², ..., H_k²]
    """
    def __init__(self, num_visible: int, k_hidden: int = 1,
                  enable_pairwise_product: bool = False,
                  holling_scale: float = 10.0,
                  normalize_basis: bool = True):
        super().__init__()
        self.num_visible = num_visible
        self.k_hidden = k_hidden
        self.holling_scale = holling_scale
        self.enable_pairwise_product = enable_pairwise_product
        self.normalize_basis = normalize_basis

        # Number of basis functions per species
        D = 1 + 4 + (num_visible - 1) + 3 * k_hidden
        if enable_pairwise_product:
            D += (num_visible - 1)  # x_i * x_j
        self.D = D

        # Learnable coefficients C ∈ R^{N, D}
        self.C = nn.Parameter(0.01 * torch.randn(num_visible, D))
        # Learnable per-species intrinsic rate is already in C[:, 0] (constant term)

        # Basis scale (set once via set_normalizer, used to scale basis for fair L1)
        self.register_buffer("basis_scale", torch.ones(num_visible, D))

    def basis_names(self) -> List[List[str]]:
        """Human-readable basis function names for each species."""
        names = [[] for _ in range(self.num_visible)]
        for i in range(self.num_visible):
            names[i].append("const")
            names[i].append(f"x{i}")
            names[i].append(f"x{i}²")
            names[i].append(f"x{i}/(1+x{i}/K)")
            names[i].append(f"log(x{i}+1)")
            for j in range(self.num_visible):
                if j != i:
                    names[i].append(f"x{j}")
            if self.enable_pairwise_product:
                for j in range(self.num_visible):
                    if j != i:
                        names[i].append(f"x{i}·x{j}")
            for k in range(self.k_hidden):
                names[i].append(f"H{k}")
            for k in range(self.k_hidden):
                names[i].append(f"H{k}·x{i}")
            for k in range(self.k_hidden):
                names[i].append(f"H{k}²")
        return names

    def build_basis(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Build basis tensor.

        x: (B, T, N) raw visible abundance
        H: (B, T, k) hidden samples

        Returns: Φ ∈ (B, T, N, D)
        """
        B, T, N = x.shape
        K = self.holling_scale
        x_safe = torch.clamp(x, min=1e-6)

        basis_list = []
        # Per-species i-specific block
        for i in range(N):
            xi = x[..., i:i+1]        # (B, T, 1)
            xi_safe = x_safe[..., i:i+1]
            # Self basis: const, xi, xi², Holling, log
            self_basis = torch.cat([
                torch.ones_like(xi),                        # const
                xi,                                         # linear
                xi ** 2,                                    # quadratic
                xi / (1 + xi_safe / K),                    # Holling II
                torch.log1p(xi_safe),                      # log
            ], dim=-1)  # (B, T, 5)
            # Pairwise linear (all j != i)
            pair_idx = [j for j in range(N) if j != i]
            pair_basis = x[..., pair_idx]  # (B, T, N-1)
            # Pairwise product (optional)
            prod_basis = None
            if self.enable_pairwise_product:
                prod_basis = xi * x[..., pair_idx]  # (B, T, N-1)
            # Hidden basis
            H_lin = H  # (B, T, k)
            H_cross = H * xi  # (B, T, k)
            H_sq = H ** 2  # (B, T, k)

            parts = [self_basis, pair_basis]
            if prod_basis is not None:
                parts.append(prod_basis)
            parts.extend([H_lin, H_cross, H_sq])
            phi_i = torch.cat(parts, dim=-1)  # (B, T, D)
            basis_list.append(phi_i)

        Phi = torch.stack(basis_list, dim=2)  # (B, T, N, D)
        if self.normalize_basis:
            Phi = Phi / (self.basis_scale.view(1, 1, N, self.D) + 1e-6)
        return Phi

    @torch.no_grad()
    def calibrate_normalizer(self, x: torch.Tensor, H: torch.Tensor):
        """Pre-compute empirical std of each basis, use it to normalize."""
        # Temporarily disable normalization to get raw basis
        was_norm = self.normalize_basis
        self.normalize_basis = False
        Phi = self.build_basis(x, H)  # (B, T, N, D)
        std = Phi.std(dim=(0, 1)) + 1e-6  # (N, D)
        self.basis_scale.copy_(std)
        self.normalize_basis = was_norm

    def forward(self, x: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Predict log_ratio for visible species.

        x: (B, T, N), H: (B, T, k)
        Returns: predicted log_ratio (B, T, N) for time t (will use [:-1] for pred)
        """
        Phi = self.build_basis(x, H)  # (B, T, N, D)
        # Contract with C (N, D)
        # pred[b, t, i] = Σ_d Φ[b, t, i, d] * C[i, d]
        pred = (Phi * self.C.view(1, 1, self.num_visible, self.D)).sum(dim=-1)
        return pred


# =============================================================================
# CVHI-Dict
# =============================================================================
class CVHI_Dict(nn.Module):
    """CVHI with SINDy-style dictionary-based dynamics.

    Keeps: PosteriorEncoder (GNN) for q(H|x)
    Replaces: DynamicsOperator with DictionaryDynamicsOperator
    """
    def __init__(
        self,
        num_visible: int,
        k_hidden: int = 1,
        encoder_d: int = 96,
        encoder_blocks: int = 3,
        encoder_heads: int = 4,
        takens_lags: List[int] = (1, 2, 4, 8, 12),
        dropout: float = 0.1,
        prior_std: float = 2.0,
        clamp_min: float = -2.5,
        clamp_max: float = 2.5,
        enable_pairwise_product: bool = False,
        holling_scale: float = 10.0,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.k_hidden = k_hidden
        self.prior_std = prior_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max

        self.encoder = MultiChannelPosteriorEncoder(
            k_hidden=k_hidden,
            num_visible=num_visible,
            takens_lags=takens_lags,
            d_model=encoder_d,
            num_heads=encoder_heads,
            num_blocks=encoder_blocks,
            dropout=dropout,
        )
        self.dynamics = DictionaryDynamicsOperator(
            num_visible=num_visible,
            k_hidden=k_hidden,
            enable_pairwise_product=enable_pairwise_product,
            holling_scale=holling_scale,
        )

    def forward(self, visible: torch.Tensor, n_samples: int = 1,
                 h_anchor: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """visible: (B, T, N) or (T, N). h_anchor: (B, T) or (T,) — scalar anchor for channel 0."""
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape

        mu, log_sigma = self.encoder(visible)  # (B, T, k)

        # Apply anchor on channel 0 only
        if h_anchor is not None:
            if h_anchor.dim() == 1:
                h_anchor = h_anchor.unsqueeze(0)
            mu_channel_0 = mu[..., 0] + h_anchor
            # Broadcast to replace channel 0
            mu_adjusted = mu.clone()
            mu_adjusted[..., 0] = mu_channel_0
            mu = mu_adjusted

        sigma = log_sigma.exp()
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        H_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, B, T, k)
        # Don't clamp — H is a latent factor, can be negative

        # Compute predictions for each sample
        visible_expanded = visible.unsqueeze(0).expand(n_samples, -1, -1, -1)
        visible_flat = visible_expanded.reshape(n_samples * B, T, N)
        H_flat = H_samples.reshape(n_samples * B, T, self.k_hidden)

        pred_log_ratio = self.dynamics(visible_flat, H_flat)  # (S*B, T, N)

        # Compute actual log_ratio from visible
        safe = torch.clamp(visible, min=1e-6)
        actual_log_ratio_visible = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio_visible = torch.clamp(
            actual_log_ratio_visible, self.clamp_min, self.clamp_max
        )

        # Use t=0..T-2 predictions to match
        pred_log_ratio = pred_log_ratio[:, :-1, :]  # (S*B, T-1, N)
        pred_log_ratio = pred_log_ratio.reshape(n_samples, B, T - 1, N)

        return {
            "mu": mu,
            "log_sigma": log_sigma,
            "H_samples": H_samples,
            "predicted_log_ratio_visible": pred_log_ratio,
            "actual_log_ratio_visible": actual_log_ratio_visible,
        }

    def elbo_loss(self, outputs, beta=1.0, lam_sparse_C=0.05, lam_smooth=0.02,
                   free_bits=0.05):
        mu = outputs["mu"]
        log_sigma = outputs["log_sigma"]
        pred = outputs["predicted_log_ratio_visible"]
        actual = outputs["actual_log_ratio_visible"]
        H_samples = outputs["H_samples"]

        recon = F.mse_loss(pred, actual.unsqueeze(0).expand_as(pred))

        prior_var = self.prior_std ** 2
        sigma_sq = torch.exp(2 * log_sigma)
        kl_per_step = 0.5 * (
            torch.log(torch.tensor(prior_var, device=mu.device)) - 2 * log_sigma
            + (sigma_sq + mu ** 2) / prior_var - 1
        )
        kl_clipped = torch.clamp(kl_per_step - free_bits, min=0)
        kl = kl_clipped.mean()

        # Dictionary coefficient L1 sparsity
        l1_C = self.dynamics.C.abs().mean()

        # Smoothness on H
        smooth = ((H_samples[:, :, 2:] - 2 * H_samples[:, :, 1:-1] + H_samples[:, :, :-2]) ** 2).mean()

        total = recon + beta * kl + lam_sparse_C * l1_C + lam_smooth * smooth
        return {
            "total": total,
            "recon": recon,
            "kl": kl,
            "l1_C": l1_C,
            "smooth": smooth,
            "sigma_mean": log_sigma.exp().mean().detach(),
        }

    def discovered_equations(self, tol: float = 0.01) -> List[str]:
        """Read off the discovered equation for each species from C coefficients.

        tol: threshold below which coefficient is considered "zero"
        """
        C = self.dynamics.C.detach().cpu().numpy()
        names = self.dynamics.basis_names()
        equations = []
        for i in range(self.num_visible):
            terms = []
            for d in range(self.dynamics.D):
                c = C[i, d]
                if abs(c) > tol:
                    terms.append(f"{c:+.3f}·{names[i][d]}")
            eq = " ".join(terms) if terms else "(all zero)"
            equations.append(f"log(x{i},t+1/x{i},t) ≈ {eq}")
        return equations

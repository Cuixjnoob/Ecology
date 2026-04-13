"""CVHI-NCD: CVHI with Species-as-Nodes GNN + Soft-Preset Form Messages.

架构 (修订版, 2026-04-13):
  - 节点 = N visible + k hidden 物种 (GNN 本来的语义)
  - 每条边 i ← j 的消息是多种生态形式的软组合:
      * Linear:        x_j
      * LV 双线性:     x_i · x_j
      * Holling II 线:  x_j / (1 + α_j · x_j)
      * Holling II 双: x_i · x_j / (1 + α_j · x_j)
      * Free NN:        MLP(x_i, x_j, s_i, s_j)
  - 每条边每种形式都有独立的 coef (正/负系数) + gate (sigmoid 门控, L1 稀疏)
    模型可以软选择 per-edge 用哪种形式
  - GNN attention 用 top-k 稀疏邻居选择

Posterior Encoder (GNN+Takens) 用来生成 hidden 采样, 不变.
"""
from __future__ import annotations

from typing import Dict, List, Tuple

import torch
from torch import nn
import torch.nn.functional as F

from models.cvhi import PosteriorEncoder


# =============================================================================
# Multi-channel Posterior Encoder
# =============================================================================
class MultiChannelPosteriorEncoder(nn.Module):
    """k 个独立 PosteriorEncoder → H ∈ R^{B,T,k}."""
    def __init__(self, k_hidden: int = 1, **encoder_kwargs):
        super().__init__()
        self.k_hidden = k_hidden
        self.encoders = nn.ModuleList([
            PosteriorEncoder(**encoder_kwargs) for _ in range(k_hidden)
        ])

    def forward(self, visible):
        mus, log_sigmas = [], []
        for enc in self.encoders:
            mu, log_sigma = enc(visible)
            mus.append(mu); log_sigmas.append(log_sigma)
        return torch.stack(mus, dim=-1), torch.stack(log_sigmas, dim=-1)


# =============================================================================
# Per-species Temporal Attention (NOT GNN — standalone sequence attention)
# =============================================================================
class PerSpeciesTemporalAttn(nn.Module):
    """Each species independently does a sequence self-attention on its own time axis.

    非 GNN 模块 — 只让每个物种"看过去的自己", 捕捉滞后/季节.
    Input:  (B, T, N) raw state + (B, T, N, d_feat) optional feature
    Output: (B, T, N, d_out) temporally-enriched feature per (t, species)
    """
    def __init__(self, d_model: int = 32, num_heads: int = 4,
                  takens_lags: Tuple[int, ...] = (1, 2, 4, 8, 12),
                  dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.takens_lags = list(takens_lags)
        feat_dim = 1 + len(self.takens_lags)  # raw + lags of that species only
        self.input_proj = nn.Sequential(
            nn.Linear(feat_dim, d_model), nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.attn = nn.MultiheadAttention(d_model, num_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 2), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model), nn.Dropout(dropout),
        )
        self.norm2 = nn.LayerNorm(d_model)
        # Sinusoidal PE buffer (built on demand)

    def _build_takens_features(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, T, N). Returns: (B, T, N, 1 + |lags|)."""
        B, T, N = state.shape
        feats = [state.unsqueeze(-1)]  # (B, T, N, 1)
        for lag in self.takens_lags:
            padded = torch.cat([state[:, :1].expand(-1, lag, -1), state[:, :-lag]], dim=1) \
                     if lag > 0 else state
            feats.append(padded.unsqueeze(-1))
        return torch.cat(feats, dim=-1)  # (B, T, N, 1 + |lags|)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, T, N) → feature (B, T, N, d_model)."""
        B, T, N = state.shape
        feats = self._build_takens_features(state)  # (B, T, N, feat_dim)
        x = self.input_proj(feats)  # (B, T, N, d_model)

        # Attend per species: reshape to (B*N, T, d_model)
        x_flat = x.permute(0, 2, 1, 3).reshape(B * N, T, self.d_model)
        # Add sinusoidal PE
        pe = torch.zeros(1, T, self.d_model, device=x.device)
        pos = torch.arange(T, device=x.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float()
                         * (-torch.log(torch.tensor(10000.0)) / self.d_model))
        pe[0, :, 0::2] = torch.sin(pos * div)
        pe[0, :, 1::2] = torch.cos(pos * div)
        x_flat = x_flat + pe

        attn_out, _ = self.attn(x_flat, x_flat, x_flat, need_weights=False)
        x_flat = self.norm1(x_flat + attn_out)
        x_flat = self.norm2(x_flat + self.ffn(x_flat))

        out = x_flat.reshape(B, N, T, self.d_model).permute(0, 2, 1, 3)
        return out  # (B, T, N, d_model)


# =============================================================================
# Species-as-Nodes GNN with Soft-Preset Form Messages
# =============================================================================
class SpeciesGNN_SoftForms(nn.Module):
    """GNN on species nodes with soft-preset ecological message forms.

    Form menu (per edge i ← j):
      0. Linear:            x_j
      1. LV bilinear:       x_i · x_j
      2. Holling II linear: x_j / (1 + α_j · x_j)
      3. Holling II bilin:  x_i · x_j / (1 + α_j · x_j)
      4. Free NN:           MLP([x_i, x_j, s_i, s_j])

    Per-edge per-form:
      - coef ∈ R (signed coefficient; negative = competition/predation on self)
      - gate ∈ [0, 1] (sigmoid gate, L1 regularized → per-edge form selection)

    GNN attention (top-k) selects which neighbors to listen to.
    """
    FORM_NAMES = ["Linear", "LV_bilin", "HollingII_lin", "HollingII_bilin", "FreeNN"]
    NUM_FORMS = 5

    def __init__(
        self,
        num_visible: int,
        num_hidden: int = 1,
        d_species: int = 32,
        top_k: int = 4,
        gate_init: float = -2.0,       # sigmoid(-2) ≈ 0.12, 初始都稍微开一点
        use_free_nn: bool = True,
        free_nn_hidden: int = 32,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        N = num_visible + num_hidden
        self.N = N
        self.top_k = min(top_k, N)
        self.d_species = d_species
        self.use_free_nn = use_free_nn

        # Species embeddings (nodes)
        self.species_emb = nn.Parameter(0.1 * torch.randn(N, d_species))

        # Attention projections: input [state(1), species_emb(d), optional temporal_feat(d_temp)]
        # d_temp is set at forward time; we support dynamic dim via extra projection
        self.q_proj = nn.Linear(1 + d_species, d_species)
        self.k_proj = nn.Linear(1 + d_species, d_species)
        # Optional projector for enriched (temporal) features into attention
        self.temp_feat_proj = nn.Linear(d_species, d_species, bias=False)  # identity-ish init

        # Per-form per-edge coefficients (signed) and gates (sigmoid → [0,1])
        self.form_coefs = nn.Parameter(0.05 * torch.randn(self.NUM_FORMS, N, N))
        # Init: most edges low gates, but h→v edges higher (force hidden involvement)
        gates_init = gate_init * torch.ones(self.NUM_FORMS, N, N)
        # h→v edges: receiver i ∈ [0, N_v), sender j ∈ [N_v, N); set these higher
        if num_hidden > 0:
            gates_init[:, :num_visible, num_visible:] = 0.0  # sigmoid(0) = 0.5
        self.form_gates_raw = nn.Parameter(gates_init)

        # Holling saturation α per species (softplus → positive)
        self.holling_alpha_raw = nn.Parameter(torch.zeros(N))

        # Free-form NN message (optional)
        if use_free_nn:
            self.free_mlp = nn.Sequential(
                nn.Linear(2 + 2 * d_species, free_nn_hidden), nn.GELU(),
                nn.Linear(free_nn_hidden, free_nn_hidden), nn.GELU(),
                nn.Linear(free_nn_hidden, 1),
            )

        # Intrinsic growth per species
        self.r = nn.Parameter(torch.zeros(N))

    def get_gates(self) -> torch.Tensor:
        """Return sigmoid-transformed gates ∈ [0,1], shape (num_forms, N, N)."""
        return torch.sigmoid(self.form_gates_raw)

    def get_holling_alpha(self) -> torch.Tensor:
        return F.softplus(self.holling_alpha_raw) + 0.01

    def compute_form_values(self, state: torch.Tensor) -> torch.Tensor:
        """Pre-compute all 5 form messages for each (i, j) pair.

        state: (B, T, N) — all species
        Returns: forms (num_forms, B, T, N_receiver, N_sender)
          forms[f, b, t, i, j] = value of form f for edge j→i at time t
        """
        B, T, N = state.shape
        d = self.d_species
        # Broadcast x_i and x_j
        xi = state.unsqueeze(3)  # (B, T, N, 1) receiver i
        xj = state.unsqueeze(2)  # (B, T, 1, N) sender j
        alpha = self.get_holling_alpha().view(1, 1, 1, N)  # sender's saturation

        # Form 0: Linear = x_j
        f0 = xj.expand(B, T, N, N).contiguous()
        # Form 1: LV bilinear = x_i · x_j
        f1 = xi * xj
        f1 = f1.expand(B, T, N, N).contiguous()
        # Form 2: Holling II linear = x_j / (1 + α_j · x_j)
        f2 = (xj / (1 + alpha * xj)).expand(B, T, N, N).contiguous()
        # Form 3: Holling II bilinear = x_i · x_j / (1 + α_j · x_j)
        f3 = (xi * xj / (1 + alpha * xj)).expand(B, T, N, N).contiguous()
        forms_fixed = [f0, f1, f2, f3]

        if self.use_free_nn:
            # Form 4: MLP([x_i, x_j, s_i, s_j])
            sp = self.species_emb  # (N, d)
            sp_i = sp.view(1, 1, N, 1, d).expand(B, T, N, N, d)
            sp_j = sp.view(1, 1, 1, N, d).expand(B, T, N, N, d)
            xi_e = xi.unsqueeze(-1).expand(B, T, N, N, 1)
            xj_e = xj.unsqueeze(-1).expand(B, T, N, N, 1)
            pair_feats = torch.cat([xi_e, xj_e, sp_i, sp_j], dim=-1)  # (B, T, N, N, 2+2d)
            f4 = self.free_mlp(pair_feats).squeeze(-1)  # (B, T, N, N)
            forms_fixed.append(f4)
        else:
            forms_fixed.append(torch.zeros_like(f0))

        forms = torch.stack(forms_fixed, dim=0)  # (num_forms, B, T, N, N)
        return forms

    def forward(self, state: torch.Tensor,
                 temporal_feat: torch.Tensor = None) -> torch.Tensor:
        """state: (B, T, N) — all species (visible + hidden).
        temporal_feat: optional (B, T, N, d_species) — 从 PerSpeciesTemporalAttn 输出,
          用于 Q/K attention (不影响 message forms).

        Returns: log_ratio (B, T, N), attn
        """
        B, T, N = state.shape
        # Step 1: Form values per edge (用原始 state!)
        forms = self.compute_form_values(state)  # (num_forms, B, T, N, N)
        gates = self.get_gates()
        coefs = self.form_coefs
        weighted = (coefs * gates).view(self.NUM_FORMS, 1, 1, N, N) * forms
        msgs = weighted.sum(dim=0)  # (B, T, N, N)

        # Step 2: Attention — 若有 temporal_feat 则用它, 否则用 (state, species_emb)
        if temporal_feat is not None:
            # 把 temporal_feat 和 species_emb 相加 (temporal_feat 已经包含时间信息)
            feats_attn = temporal_feat + self.species_emb.view(1, 1, N, -1)
            feats = torch.cat([state.unsqueeze(-1), self.temp_feat_proj(feats_attn)], dim=-1)
            # dim = 1 + d_species, 匹配 q_proj 输入
        else:
            feats = torch.cat([state.unsqueeze(-1),
                               self.species_emb.view(1, 1, N, -1).expand(B, T, -1, -1)],
                              dim=-1)
        q = self.q_proj(feats)  # (B, T, N, d)
        k = self.k_proj(feats)  # (B, T, N, d)
        scores = torch.einsum("btnd,btmd->btnm", q, k) / (self.d_species ** 0.5)

        if self.top_k < N:
            topk_vals, topk_idx = scores.topk(self.top_k, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, topk_idx, topk_vals)
            attn = F.softmax(mask, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)
        # (B, T, N_receiver, N_sender)

        # Step 3: Aggregate messages with attention weights
        agg = (attn * msgs).sum(dim=-1)  # (B, T, N)

        log_ratio = self.r.view(1, 1, -1) + agg
        return log_ratio, attn

    def l1_gates(self) -> torch.Tensor:
        """L1 on sigmoid gates → soft form selection sparsity."""
        return self.get_gates().mean()  # penalize total "open gate" amount

    def l1_coefs(self) -> torch.Tensor:
        """L1 on coefficients × gates (effective magnitude)."""
        return (self.form_coefs.abs() * self.get_gates()).mean()

    def hidden_to_visible_mass(self) -> torch.Tensor:
        """Total gate·|coef| mass from hidden → visible edges. For anti-bypass penalty.

        Edges j→i where j is hidden (j >= num_visible), i is visible (i < num_visible).
        Returns a scalar — mass should NOT be near 0.
        """
        gates = self.get_gates()  # (num_forms, N, N)
        coefs = self.form_coefs  # (num_forms, N, N)
        N_v = self.num_visible
        # Slice: i in [0, N_v), j in [N_v, N)
        h2v_mass = (gates[:, :N_v, N_v:] * coefs[:, :N_v, N_v:].abs()).mean()
        return h2v_mass

    def l1_attn(self, attn: torch.Tensor) -> torch.Tensor:
        """L1 on attention (off-diagonal only — self-loops okay)."""
        B, T, N, _ = attn.shape
        mask = 1 - torch.eye(N, device=attn.device).view(1, 1, N, N)
        return (attn * mask).abs().mean()

    @torch.no_grad()
    def decode_edge_forms(self, species_names: List[str] = None,
                           gate_tol: float = 0.2, coef_tol: float = 0.02) -> List[str]:
        """For each edge, report which forms are 'active' (gate > tol, coef > tol)."""
        if species_names is None:
            species_names = [f"x{i}" for i in range(self.N - self.num_hidden)] + \
                           [f"H{k}" for k in range(self.num_hidden)]
        gates = self.get_gates().cpu().numpy()
        coefs = self.form_coefs.cpu().numpy()
        lines = []
        for i in range(self.N):
            for j in range(self.N):
                active_forms = []
                for f, fname in enumerate(self.FORM_NAMES):
                    if gates[f, i, j] > gate_tol and abs(coefs[f, i, j]) > coef_tol:
                        active_forms.append(
                            f"{fname}(g={gates[f,i,j]:.2f},c={coefs[f,i,j]:+.3f})"
                        )
                if active_forms:
                    lines.append(f"{species_names[j]}→{species_names[i]}: "
                                 + ", ".join(active_forms))
        return lines


# =============================================================================
# Species GNN with PURE MLP messages (NO preset formulas)
# =============================================================================
class SpeciesGNN_MLP(nn.Module):
    """Species-as-nodes GNN with learnable MLP edge messages.

    边消息 j → i:
      m_ij = MLP([x_i, x_j, s_i, s_j, 预设公式 as hints])
    预设公式 (Linear, LV bilinear, Holling II × 2) 作为**额外输入特征**
    提供给 MLP. MLP 可以选择性使用/加权/忽略/组合它们, 不强制.
    """
    def __init__(
        self,
        num_visible: int,
        num_hidden: int = 0,
        d_species: int = 24,
        top_k: int = 4,
        d_msg_hidden: int = 32,
        use_free_nn: bool = True,  # API compat
        free_nn_hidden: int = 32,  # API compat
        use_formula_hints: bool = True,  # 是否把预设公式作为 MLP 输入特征
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        N = num_visible + num_hidden
        self.N = N
        self.top_k = min(top_k, N)
        self.d_species = d_species
        self.use_formula_hints = use_formula_hints

        # Learnable species embeddings (distinguish species identity in MLP)
        self.species_emb = nn.Parameter(0.1 * torch.randn(N, d_species))

        # Holling saturation α per species (if using formula hints)
        if use_formula_hints:
            self.holling_alpha_raw = nn.Parameter(torch.zeros(N))

        # Edge message MLP input:
        #   [x_i, x_j, s_i, s_j]              = 2 + 2*d_species
        #   + 4 formula hints (if enabled): [x_j, x_i·x_j, HollingII_lin, HollingII_bilin]
        n_hints = 4 if use_formula_hints else 0
        in_dim = 2 + 2 * d_species + n_hints
        self.msg_mlp = nn.Sequential(
            nn.Linear(in_dim, d_msg_hidden), nn.GELU(),
            nn.Linear(d_msg_hidden, d_msg_hidden), nn.GELU(),
            nn.Linear(d_msg_hidden, 1),
        )

        # Attention projections
        self.q_proj = nn.Linear(1 + d_species, d_species)
        self.k_proj = nn.Linear(1 + d_species, d_species)
        self.temp_feat_proj = nn.Linear(d_species, d_species, bias=False)

        # Intrinsic growth
        self.r = nn.Parameter(torch.zeros(N))

    def compute_messages(self, state: torch.Tensor) -> torch.Tensor:
        """state: (B, T, N). Returns: (B, T, N, N) where [:, :, i, j] = message j→i.

        如果 use_formula_hints: MLP 除了 [x_i, x_j, s_i, s_j] 还接收 4 个预设公式值作为 hints.
        """
        B, T, N = state.shape
        d = self.d_species
        xi = state.unsqueeze(-1).expand(B, T, N, N)
        xj = state.unsqueeze(-2).expand(B, T, N, N)
        sp = self.species_emb
        si = sp.view(1, 1, N, 1, d).expand(B, T, N, N, d)
        sj = sp.view(1, 1, 1, N, d).expand(B, T, N, N, d)
        xi_e = xi.unsqueeze(-1)
        xj_e = xj.unsqueeze(-1)
        feats = [xi_e, xj_e, si, sj]

        if self.use_formula_hints:
            # 4 preset formula values as additional features (hints)
            alpha = (F.softplus(self.holling_alpha_raw) + 0.01).view(1, 1, 1, N)
            f_linear = xj_e                               # (B, T, N, N, 1)  x_j
            f_lv = (xi * xj).unsqueeze(-1)                # x_i * x_j
            f_holl_lin = (xj / (1 + alpha * xj)).unsqueeze(-1)          # x_j / (1+α·x_j)
            f_holl_bi = (xi * xj / (1 + alpha * xj)).unsqueeze(-1)      # x_i·x_j / (1+α·x_j)
            feats.extend([f_linear, f_lv, f_holl_lin, f_holl_bi])

        pair = torch.cat(feats, dim=-1)  # (B, T, N, N, in_dim)
        return self.msg_mlp(pair).squeeze(-1)

    def forward(self, state: torch.Tensor,
                 temporal_feat: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N = state.shape
        msgs = self.compute_messages(state)  # (B, T, N, N)

        # Attention Q/K (same as SoftForms)
        if temporal_feat is not None:
            feats_attn = temporal_feat + self.species_emb.view(1, 1, N, -1)
            feats = torch.cat([state.unsqueeze(-1), self.temp_feat_proj(feats_attn)], dim=-1)
        else:
            feats = torch.cat([state.unsqueeze(-1),
                               self.species_emb.view(1, 1, N, -1).expand(B, T, -1, -1)],
                              dim=-1)
        q = self.q_proj(feats); k = self.k_proj(feats)
        scores = torch.einsum("btnd,btmd->btnm", q, k) / (self.d_species ** 0.5)
        if self.top_k < N:
            tv, ti = scores.topk(self.top_k, dim=-1)
            mask = torch.full_like(scores, float("-inf"))
            mask.scatter_(-1, ti, tv)
            attn = F.softmax(mask, dim=-1)
        else:
            attn = F.softmax(scores, dim=-1)

        agg = (attn * msgs).sum(dim=-1)
        log_ratio = self.r.view(1, 1, -1) + agg
        return log_ratio, attn

    # API compat stubs (no gates/forms in pure-MLP version)
    def l1_gates(self) -> torch.Tensor:
        return torch.tensor(0.0, device=self.r.device)

    def l1_coefs(self) -> torch.Tensor:
        # L2 on MLP weights instead (light regularization)
        l2 = 0.0
        for p in self.msg_mlp.parameters():
            l2 = l2 + p.pow(2).mean()
        return l2 / 4

    def l1_attn(self, attn: torch.Tensor) -> torch.Tensor:
        B, T, N, _ = attn.shape
        mask = 1 - torch.eye(N, device=attn.device).view(1, 1, N, N)
        return (attn * mask).abs().mean()

    def hidden_to_visible_mass(self) -> torch.Tensor:
        # N_h = 0 in CVHI_Residual usage, return 0
        return torch.tensor(0.0, device=self.r.device)


# =============================================================================
# Multi-layer Species GNN (stacked SoftForms, independent params per layer)
# =============================================================================
class MultiLayerSpeciesGNN(nn.Module):
    """Stack L copies of SpeciesGNN_SoftForms; each contributes a log_ratio term.

    log_ratio = Σ_l layer_l(state, temporal_feat)
    Each layer has independent gates/coefs/attention → different forms can activate at
    different depths (layer 1 might learn direct effects, layer 2 indirect).
    """
    def __init__(self, num_layers: int, backbone: str = "softforms", **layer_kwargs):
        super().__init__()
        self.num_layers = num_layers
        self.backbone = backbone
        if backbone == "softforms":
            LayerCls = SpeciesGNN_SoftForms
        elif backbone == "mlp":
            LayerCls = SpeciesGNN_MLP
            # MLP backbone doesn't use these args — strip them
            layer_kwargs = {k: v for k, v in layer_kwargs.items()
                             if k not in ("gate_init",)}
        else:
            raise ValueError(f"Unknown backbone: {backbone}")
        self.layers = nn.ModuleList([
            LayerCls(**layer_kwargs) for _ in range(num_layers)
        ])

    def forward(self, state, temporal_feat=None):
        total_log_ratio = 0
        attns = []
        for layer in self.layers:
            lr, attn = layer(state, temporal_feat=temporal_feat)
            total_log_ratio = total_log_ratio + lr
            attns.append(attn)
        # Scale by 1/L to prevent log_ratio growing with depth
        total_log_ratio = total_log_ratio / self.num_layers
        return total_log_ratio, attns

    def l1_gates(self):
        return torch.stack([l.l1_gates() for l in self.layers]).mean()

    def l1_coefs(self):
        return torch.stack([l.l1_coefs() for l in self.layers]).mean()

    def l1_attn(self, attns):
        # attns is list of per-layer attention tensors
        return torch.stack([l.l1_attn(a) for l, a in zip(self.layers, attns)]).mean()

    def hidden_to_visible_mass(self):
        return torch.stack([l.hidden_to_visible_mass() for l in self.layers]).mean()


# =============================================================================
# CVHI-NCD full model
# =============================================================================
class CVHI_NCD(nn.Module):
    """CVHI with Species-GNN SoftForms dynamics."""
    def __init__(
        self,
        num_visible: int,
        num_hidden: int = 1,
        encoder_d: int = 96,
        encoder_blocks: int = 3,
        encoder_heads: int = 4,
        takens_lags: Tuple[int, ...] = (1, 2, 4, 8, 12),
        dropout: float = 0.1,
        d_species: int = 32,
        top_k: int = 4,
        use_free_nn: bool = True,
        free_nn_hidden: int = 32,
        use_temporal_attn: bool = True,
        num_gnn_layers: int = 2,         # NEW: stacked GNN layers
        anchor_scale: float = 0.3,        # NEW: delta_mu scaled by this (bound drift)
        prior_std: float = 0.5,           # tighter default
        clamp_min: float = -2.5,
        clamp_max: float = 2.5,
    ):
        super().__init__()
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.prior_std = prior_std
        self.clamp_min = clamp_min
        self.clamp_max = clamp_max
        self.anchor_scale = anchor_scale

        self.encoder = MultiChannelPosteriorEncoder(
            k_hidden=num_hidden,
            num_visible=num_visible,
            takens_lags=list(takens_lags),
            d_model=encoder_d,
            num_heads=encoder_heads,
            num_blocks=encoder_blocks,
            dropout=dropout,
        )
        # Per-species temporal attention (non-GNN, seq attention per species, optional)
        self.use_temporal_attn = use_temporal_attn
        if use_temporal_attn:
            self.temporal_attn = PerSpeciesTemporalAttn(
                d_model=d_species, num_heads=4,
                takens_lags=list(takens_lags), dropout=dropout,
            )
        else:
            self.temporal_attn = None
        self.dynamics = MultiLayerSpeciesGNN(
            num_layers=num_gnn_layers,
            num_visible=num_visible,
            num_hidden=num_hidden,
            d_species=d_species,
            top_k=top_k,
            use_free_nn=use_free_nn,
            free_nn_hidden=free_nn_hidden,
        )

    def forward(self, visible: torch.Tensor, n_samples: int = 1,
                 h_anchor: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N_v = visible.shape

        delta_mu, log_sigma = self.encoder(visible)  # (B, T, k) — raw encoder output
        # Scale delta_mu so it can only be a small adjustment around anchor
        delta_mu_scaled = self.anchor_scale * delta_mu

        if h_anchor is not None:
            if h_anchor.dim() == 1:
                h_anchor = h_anchor.unsqueeze(0)
            mu_ch0 = delta_mu_scaled[..., 0] + h_anchor
            mu = torch.cat([mu_ch0.unsqueeze(-1), delta_mu_scaled[..., 1:]], dim=-1)
        else:
            mu = delta_mu_scaled

        sigma = log_sigma.exp()
        eps = torch.randn(n_samples, *mu.shape, device=mu.device)
        H_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps  # (S, B, T, k)
        H_pos = torch.clamp(H_samples, min=0.01)  # force positive for ecology

        # Build full state per sample
        visible_exp = visible.unsqueeze(0).expand(n_samples, -1, -1, -1)
        state = torch.cat([visible_exp, H_pos], dim=-1)  # (S, B, T, N_total)
        state_flat = state.reshape(n_samples * B, *state.shape[-2:])

        # Temporal attention preprocessing (非 GNN, per-species, optional)
        if self.temporal_attn is not None:
            temporal_feat = self.temporal_attn(state_flat)  # (S*B, T, N_total, d_species)
        else:
            temporal_feat = None

        log_ratio, attn = self.dynamics(state_flat, temporal_feat=temporal_feat)
        log_ratio = log_ratio.reshape(n_samples, B, *log_ratio.shape[-2:])

        # Target: actual visible log_ratio
        safe = torch.clamp(visible, min=1e-6)
        actual_log_ratio = torch.log(safe[:, 1:] / safe[:, :-1])
        actual_log_ratio = torch.clamp(actual_log_ratio, self.clamp_min, self.clamp_max)

        pred_log_ratio_visible = log_ratio[:, :, :-1, :N_v]  # (S, B, T-1, N_v)

        return {
            "mu": mu,
            "delta_mu": delta_mu,  # raw encoder output (for KL)
            "log_sigma": log_sigma,
            "H_samples": H_samples,
            "H_pos": H_pos,
            "predicted_log_ratio_visible": pred_log_ratio_visible,
            "actual_log_ratio_visible": actual_log_ratio,
            "attn": attn,
        }

    def elbo_loss(self, outputs, beta=1.0,
                   lam_gates=0.01, lam_coefs=0.005, lam_attn=0.01,
                   lam_smooth=0.02, free_bits=0.05,
                   lam_anti_bypass=1.0, min_h2v_mass=0.05):
        mu = outputs["mu"]
        delta_mu = outputs.get("delta_mu", mu)  # use delta_mu for KL (not anchor-shifted mu)
        log_sigma = outputs["log_sigma"]
        pred = outputs["predicted_log_ratio_visible"]
        actual = outputs["actual_log_ratio_visible"]
        H_samples = outputs["H_samples"]
        attn = outputs["attn"]

        recon = F.mse_loss(pred, actual.unsqueeze(0).expand_as(pred))

        prior_var = self.prior_std ** 2
        sigma_sq = torch.exp(2 * log_sigma)
        # KL on delta_mu (deviation from anchor), NOT on mu (which includes anchor)
        kl_per_step = 0.5 * (
            torch.log(torch.tensor(prior_var, device=mu.device)) - 2 * log_sigma
            + (sigma_sq + delta_mu ** 2) / prior_var - 1
        )
        kl = torch.clamp(kl_per_step - free_bits, min=0).mean()

        l1_gates = self.dynamics.l1_gates()
        l1_coefs = self.dynamics.l1_coefs()
        l1_attn = self.dynamics.l1_attn(attn)  # attn is list of per-layer attention
        smooth = ((H_samples[:, :, 2:] - 2 * H_samples[:, :, 1:-1] + H_samples[:, :, :-2]) ** 2).mean()

        # Anti-bypass: hidden→visible edges must have minimum mass
        h2v_mass = self.dynamics.hidden_to_visible_mass()
        anti_bypass = F.relu(min_h2v_mass - h2v_mass)

        total = (recon + beta * kl
                 + lam_gates * l1_gates + lam_coefs * l1_coefs
                 + lam_attn * l1_attn + lam_smooth * smooth
                 + lam_anti_bypass * anti_bypass)
        return {
            "total": total, "recon": recon, "kl": kl,
            "l1_gates": l1_gates, "l1_coefs": l1_coefs, "l1_attn": l1_attn,
            "anti_bypass": anti_bypass, "h2v_mass": h2v_mass,
            "smooth": smooth, "sigma_mean": log_sigma.exp().mean().detach(),
        }

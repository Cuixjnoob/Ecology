from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from datetime import datetime
from itertools import combinations
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from torchdiffeq import odeint

from models.cvhi_ncd import MultiChannelPosteriorEncoder, MultiLayerSpeciesGNN
from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import hidden_true_substitution
from scripts.load_beninca import load_beninca
from scripts.load_maizuru import load_maizuru


ROOT = Path("重要实验")
RESULTS_ROOT = ROOT / "results"
FIG_ROOT = ROOT / "figures"
LOG_ROOT = ROOT / "logs"
CONFIG_ROOT = ROOT / "configs"

SEEDS_10 = [42, 123, 456, 789, 2024, 31415, 27182, 65537, 11111, 22222]
SEEDS_5 = SEEDS_10[:5]

MAIN_DATASETS = ["lv", "holling", "huisman", "beninca", "maizuru"]
MAIN_METHOD = "eco_gnrd_alt5_hdyn"
BASELINE_METHODS_ALL = ["var_pca", "mlp_pca", "edm_simplex", "mve", "supervised_ridge"]
BASELINE_METHODS_DEEP = ["lstm", "neural_ode", "latent_ode"]
ABLATION_METHODS = [
    "minus_null",
    "minus_shuffle",
    "minus_both_counterfactual",
    "minus_residual_decomp",
    "f_only",
    "minus_formula_hints",
    "rollout_0",
    "rollout_5",
    "minus_takens",
    "minus_alt5",
    "minus_hdyn",
]

PLOT_DATASET_ORDER = ["huisman", "beninca", "maizuru", "lv", "holling"]
REPRESENTATIVE_TASKS = {
    "lv": "hidden_B",
    "holling": "hidden_B",
    "huisman": "sp1",
    "beninca": "Ostracods",
    "maizuru": "Pseudolabrus.sieboldi",
}


def now_stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def ensure_root_layout() -> None:
    for path in [ROOT, RESULTS_ROOT, FIG_ROOT, LOG_ROOT, CONFIG_ROOT]:
        path.mkdir(parents=True, exist_ok=True)


def default_config() -> Dict:
    return {
        "created": now_stamp(),
        "main_method": MAIN_METHOD,
        "main_datasets": MAIN_DATASETS,
        "main_seeds": len(SEEDS_10),
        "baseline_methods_all": BASELINE_METHODS_ALL,
        "baseline_methods_deep": BASELINE_METHODS_DEEP,
        "baseline_seeds_all": len(SEEDS_10),
        "baseline_seeds_deep": len(SEEDS_5),
        "ablation_methods": ABLATION_METHODS,
        "ablation_note": "A0 is the main method result and is not rerun in the ablation suite.",
    }


def write_default_config() -> None:
    ensure_root_layout()
    cfg_path = CONFIG_ROOT / "important_experiment_grid.json"
    if not cfg_path.exists():
        cfg_path.write_text(json.dumps(default_config(), indent=2, ensure_ascii=False), encoding="utf-8")


def safe_float(x: float) -> float:
    if x is None:
        return float("nan")
    try:
        return float(x)
    except Exception:
        return float("nan")


def corrcoef_safe(a: np.ndarray, b: np.ndarray) -> float:
    L = min(len(a), len(b))
    if L <= 2:
        return float("nan")
    aa = np.asarray(a[:L], dtype=np.float64)
    bb = np.asarray(b[:L], dtype=np.float64)
    if np.std(aa) < 1e-10 or np.std(bb) < 1e-10:
        return float("nan")
    return float(np.corrcoef(aa, bb)[0, 1])


def fit_affine_on_train(pred: np.ndarray, true: np.ndarray, train_end: int) -> Tuple[np.ndarray, Dict[str, float]]:
    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    L = min(len(pred), len(true))
    pred = pred[:L]
    true = true[:L]
    te = max(2, min(train_end, L))
    X_tr = np.column_stack([pred[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, true[:te], rcond=None)
    scaled = np.column_stack([pred, np.ones(L)]) @ coef
    metrics = {
        "pearson_all": corrcoef_safe(scaled, true),
        "pearson_val": corrcoef_safe(scaled[te:], true[te:]) if L - te > 2 else float("nan"),
        "scale_a": float(coef[0]),
        "scale_b": float(coef[1]),
        "train_end_aligned": int(te),
        "aligned_length": int(L),
    }
    return scaled.astype(np.float32), metrics


def pad_to_length(values: np.ndarray, total_len: int, offset: int) -> np.ndarray:
    arr = np.full(total_len, np.nan, dtype=np.float32)
    offset = max(0, min(offset, total_len))
    end = min(total_len, offset + len(values))
    arr[offset:end] = values[: end - offset]
    return arr


def dataset_display_name(name: str) -> str:
    mapping = {
        "lv": "LV",
        "holling": "Holling",
        "huisman": "Huisman",
        "beninca": "Beninca",
        "maizuru": "Maizuru",
    }
    return mapping.get(name, name)


@dataclass
class Task:
    dataset: str
    task_name: str
    visible: np.ndarray
    hidden: np.ndarray
    n_recon_channels: int
    input_feature_names: List[str]
    hidden_name: str
    time_axis: np.ndarray


def normalize_columns(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32) + 0.01
    means = np.maximum(x.mean(axis=0, keepdims=True), 1e-3)
    return (x / means).astype(np.float32)


def load_lv_tasks() -> List[Task]:
    d = np.load("runs/analysis_5vs6_species/trajectories.npz")
    visible = normalize_columns(d["states_B_5species"])
    hidden = normalize_columns(d["hidden_B"].reshape(-1, 1)).squeeze(-1)
    T = len(hidden)
    return [Task("lv", "hidden_B", visible, hidden, visible.shape[1],
                 [f"v{i+1}" for i in range(visible.shape[1])], "hidden_B", np.arange(T))]


def load_holling_tasks() -> List[Task]:
    d = np.load("runs/20260413_100414_5vs6_holling/trajectories.npz")
    visible = normalize_columns(d["states_B_5species"])
    hidden = normalize_columns(d["hidden_B"].reshape(-1, 1)).squeeze(-1)
    T = len(hidden)
    return [Task("holling", "hidden_B", visible, hidden, visible.shape[1],
                 [f"v{i+1}" for i in range(visible.shape[1])], "hidden_B", np.arange(T))]


def load_huisman_tasks() -> List[Task]:
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    n_all = d["N_all"].astype(np.float32)
    resources = d["resources"].astype(np.float32)
    t_axis = np.asarray(d["t_axis"])
    tasks = []
    for idx in range(n_all.shape[1]):
        vis_species = np.delete(n_all, idx, axis=1)
        full_vis = normalize_columns(np.concatenate([vis_species, resources], axis=1))
        hidden = normalize_columns(n_all[:, idx].reshape(-1, 1)).squeeze(-1)
        names = [f"sp{i+1}" for i in range(n_all.shape[1]) if i != idx] + [f"res{i+1}" for i in range(resources.shape[1])]
        tasks.append(Task(
            dataset="huisman",
            task_name=f"sp{idx+1}",
            visible=full_vis,
            hidden=hidden,
            n_recon_channels=vis_species.shape[1],
            input_feature_names=names,
            hidden_name=f"sp{idx+1}",
            time_axis=t_axis,
        ))
    return tasks


def load_beninca_tasks() -> List[Task]:
    full, species, days = load_beninca(include_nutrients=True)
    species = [str(s) for s in species]
    order = ["Cyclopoids", "Calanoids", "Rotifers", "Nanophyto",
             "Picophyto", "Filam_diatoms", "Ostracods", "Harpacticoids", "Bacteria"]
    tasks = []
    for name in order:
        h_idx = species.index(name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        feat_names = species[:h_idx] + species[h_idx + 1:]
        tasks.append(Task(
            dataset="beninca",
            task_name=name,
            visible=visible,
            hidden=hidden,
            n_recon_channels=visible.shape[1],
            input_feature_names=feat_names,
            hidden_name=name,
            time_axis=np.asarray(days),
        ))
    return tasks


def load_maizuru_tasks() -> List[Task]:
    full, species, days = load_maizuru(include_temp=True)
    species = [str(s) for s in species]
    species_only = [s for s in species if s not in ["surf_temp", "bot_temp"]]
    tasks = []
    for name in species_only:
        h_idx = species.index(name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        feat_names = species[:h_idx] + species[h_idx + 1:]
        tasks.append(Task(
            dataset="maizuru",
            task_name=name,
            visible=visible,
            hidden=hidden,
            n_recon_channels=len(species_only) - 1,
            input_feature_names=feat_names,
            hidden_name=name,
            time_axis=np.asarray(days),
        ))
    return tasks


def load_tasks(dataset: str) -> List[Task]:
    if dataset == "lv":
        return load_lv_tasks()
    if dataset == "holling":
        return load_holling_tasks()
    if dataset == "huisman":
        return load_huisman_tasks()
    if dataset == "beninca":
        return load_beninca_tasks()
    if dataset == "maizuru":
        return load_maizuru_tasks()
    raise ValueError(f"Unknown dataset: {dataset}")


def task_train_end(task: Task) -> int:
    return int(0.75 * len(task.hidden))


def device_name() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def to_tensor(x: np.ndarray, device: str) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32, device=device)


def alpha_schedule(epoch: int, epochs: int, start: float = 0.5, end: float = 0.95) -> float:
    frac = epoch / max(1, epochs)
    if frac <= start:
        return 1.0
    if frac >= end:
        return 0.0
    return 1.0 - (frac - start) / (end - start)


def cosine_lr(step: int, epochs: int) -> float:
    if step < 50:
        return step / 50
    prog = (step - 50) / max(1, epochs - 50)
    return 0.5 * (1 + np.cos(np.pi * prog))


class LatentDynamicsNet(nn.Module):
    def __init__(self, n_visible: int, d_hidden: int = 32):
        super().__init__()
        self.context = nn.Sequential(
            nn.Linear(n_visible, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
        )
        self.net = nn.Sequential(
            nn.Linear(1 + d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, d_hidden), nn.GELU(),
            nn.Linear(d_hidden, 1),
        )

    def forward(self, h_prev: torch.Tensor, x_prev: torch.Tensor) -> torch.Tensor:
        ctx = self.context(x_prev)
        inp = torch.cat([h_prev.unsqueeze(-1), ctx], dim=-1)
        return h_prev + self.net(inp).squeeze(-1)


class DirectConcatModel(CVHI_Residual):
    def __init__(
        self,
        num_visible: int,
        encoder_d: int,
        encoder_blocks: int,
        encoder_heads: int,
        takens_lags: Tuple[int, ...],
        encoder_dropout: float,
        d_species: int,
        layers: int,
        top_k: int,
        prior_std: float,
        gnn_backbone: str = "mlp",
        use_formula_hints: bool = True,
        point_estimate: bool = False,
    ):
        super().__init__(
            num_visible=num_visible,
            encoder_d=encoder_d,
            encoder_blocks=encoder_blocks,
            encoder_heads=encoder_heads,
            takens_lags=takens_lags,
            encoder_dropout=encoder_dropout,
            d_species_f=d_species,
            f_visible_layers=layers,
            f_visible_top_k=top_k,
            d_species_G=d_species,
            G_field_layers=1,
            G_field_top_k=top_k,
            prior_std=prior_std,
            gnn_backbone=gnn_backbone,
            use_formula_hints=use_formula_hints,
            use_G_field=False,
            num_mixture_components=1,
            G_anchor_first=True,
            G_anchor_sign=+1,
            point_estimate=point_estimate,
        )
        extra_kw = {"use_formula_hints": use_formula_hints} if gnn_backbone == "mlp" else {}
        self.f_visible = MultiLayerSpeciesGNN(
            num_layers=layers,
            backbone=gnn_backbone,
            num_visible=num_visible,
            num_hidden=1,
            d_species=d_species,
            top_k=top_k,
            **extra_kw,
        )
        self.G_field = None

    def _predict_with_h(self, visible: torch.Tensor, h_samples: torch.Tensor) -> torch.Tensor:
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        if h_samples.dim() == 2:
            h_samples = h_samples.unsqueeze(0)
        S, B, T = h_samples.shape
        vis = visible.unsqueeze(0).expand(S, -1, -1, -1)
        aug = torch.cat([vis, h_samples.unsqueeze(-1)], dim=-1)
        flat = aug.reshape(S * B, T, aug.shape[-1])
        pred, _ = self.f_visible(flat, temporal_feat=None)
        pred = pred.reshape(S, B, T, aug.shape[-1])
        return pred[..., :visible.shape[-1]]

    def _rollout(self, visible: torch.Tensor, mu: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N = visible.shape
        num_starts = T - K
        if num_starts <= 0:
            return None, None
        log_x = torch.log(torch.clamp(visible, min=1e-6))
        x_curr = visible[:, :num_starts, :]
        rollout = []
        for k in range(K):
            h_step = mu[:, k:k + num_starts]
            aug = torch.cat([x_curr, h_step.unsqueeze(-1)], dim=-1)
            pred, _ = self.f_visible(aug, temporal_feat=None)
            log_ratio = pred[..., :N]
            log_ratio = torch.clamp(log_ratio, self.clamp_min, self.clamp_max)
            x_next = torch.clamp(x_curr * torch.exp(log_ratio), min=1e-6)
            rollout.append(torch.log(x_next))
            x_curr = x_next
        rollout_logs = torch.stack(rollout, dim=-2)
        target_logs = torch.stack([log_x[:, k + 1:k + 1 + num_starts, :] for k in range(K)], dim=-2)
        return rollout_logs, target_logs

    def forward(self, visible: torch.Tensor, n_samples: int = 1, rollout_K: int = 0, species_ids: torch.Tensor = None):
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape
        mu_k, log_sigma_k = self.encoder(visible, residual=None, species_ids=species_ids)
        mu = mu_k[..., 0]
        log_sigma = log_sigma_k[..., 0]
        if self.point_estimate:
            h_samples = mu.unsqueeze(0).expand(n_samples, B, T)
            log_sigma = torch.zeros_like(log_sigma)
        else:
            sigma = log_sigma.exp()
            eps = torch.randn(n_samples, B, T, device=visible.device)
            h_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        pred_full = self._predict_with_h(visible, h_samples)
        pred_null = self._predict_with_h(visible, torch.zeros(1, B, T, device=visible.device))[0]
        perm = torch.randperm(T, device=visible.device)
        pred_shuf = self._predict_with_h(visible, h_samples[:, :, perm])
        safe = torch.clamp(visible, min=1e-6)
        actual = torch.log(safe[:, 1:] / safe[:, :-1])
        actual = torch.clamp(actual, self.clamp_min, self.clamp_max)
        pred_full = pred_full[:, :, :-1, :]
        pred_null = pred_null[:, :-1, :]
        pred_shuf = pred_shuf[:, :, :-1, :]
        out = {
            "mu": mu,
            "log_sigma": log_sigma,
            "h_samples": h_samples,
            "pred_full": pred_full,
            "pred_null": pred_null,
            "pred_shuf": pred_shuf,
            "actual": actual,
            "G": torch.zeros(B, T, N, device=visible.device),
            "base": torch.zeros(B, T, N, device=visible.device),
            "visible": visible,
        }
        if rollout_K > 0 and T > rollout_K + 1:
            rls, tls = self._rollout(visible, mu, rollout_K)
            out["rollout_log_states"] = rls
            out["target_log_states"] = tls
            out["rollout_K"] = rollout_K
        return out


class FOnlyModel(CVHI_Residual):
    def _rollout_f_only(self, visible: torch.Tensor, K: int) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T, N = visible.shape
        num_starts = T - K
        if num_starts <= 0:
            return None, None
        log_x = torch.log(torch.clamp(visible, min=1e-6))
        x_curr = visible[:, :num_starts, :]
        rollout = []
        for _ in range(K):
            base = self.compute_f_visible(x_curr)
            base = torch.clamp(base, self.clamp_min, self.clamp_max)
            x_next = torch.clamp(x_curr * torch.exp(base), min=1e-6)
            rollout.append(torch.log(x_next))
            x_curr = x_next
        rollout_logs = torch.stack(rollout, dim=-2)
        target_logs = torch.stack([log_x[:, k + 1:k + 1 + num_starts, :] for k in range(K)], dim=-2)
        return rollout_logs, target_logs

    def forward(self, visible: torch.Tensor, n_samples: int = 1, rollout_K: int = 0, species_ids: torch.Tensor = None):
        if visible.dim() == 2:
            visible = visible.unsqueeze(0)
        B, T, N = visible.shape
        mu_k, log_sigma_k = self.encoder(visible, residual=None, species_ids=species_ids)
        mu = mu_k[..., 0]
        log_sigma = log_sigma_k[..., 0]
        if self.point_estimate:
            h_samples = mu.unsqueeze(0).expand(n_samples, B, T)
            log_sigma = torch.zeros_like(log_sigma)
        else:
            sigma = log_sigma.exp()
            eps = torch.randn(n_samples, B, T, device=visible.device)
            h_samples = mu.unsqueeze(0) + sigma.unsqueeze(0) * eps
        base = self.compute_f_visible(visible)
        pred_full = base.unsqueeze(0).expand(n_samples, -1, -1, -1)
        pred_null = base
        pred_shuf = pred_full.clone()
        safe = torch.clamp(visible, min=1e-6)
        actual = torch.log(safe[:, 1:] / safe[:, :-1])
        actual = torch.clamp(actual, self.clamp_min, self.clamp_max)
        out = {
            "mu": mu,
            "log_sigma": log_sigma,
            "h_samples": h_samples,
            "pred_full": pred_full[:, :, :-1, :],
            "pred_null": pred_null[:, :-1, :],
            "pred_shuf": pred_shuf[:, :, :-1, :],
            "actual": actual,
            "G": torch.zeros(B, T, N, device=visible.device),
            "base": base,
            "visible": visible,
        }
        if rollout_K > 0 and T > rollout_K + 1:
            rls, tls = self._rollout_f_only(visible, rollout_K)
            out["rollout_log_states"] = rls
            out["target_log_states"] = tls
            out["rollout_K"] = rollout_K
        return out


def model_param_groups(model: nn.Module) -> Tuple[List[nn.Parameter], List[nn.Parameter]]:
    enc = []
    rest = []
    for name, p in model.named_parameters():
        if "encoder" in name:
            enc.append(p)
        else:
            rest.append(p)
    return rest, enc


def base_preset(dataset: str) -> Dict:
    if dataset == "beninca":
        return {
            "epochs": 500,
            "lr": 0.0006033475528697158,
            "encoder_d": 96,
            "encoder_blocks": 3,
            "encoder_heads": 4,
            "encoder_dropout": 0.1,
            "takens_lags": (1, 2, 4, 8),
            "d_species_f": 20,
            "f_visible_layers": 2,
            "f_visible_top_k": 4,
            "d_species_G": 12,
            "G_field_layers": 1,
            "G_field_top_k": 3,
            "beta_kl": 0.017251789430967935,
            "lam_cf": 9.517725868477207,
            "min_energy": 0.14353013693386804,
            "lam_hdyn": 0.5,
            "lam_rmse_log": 0.1,
            "lam_hf": 0.2,
        }
    if dataset == "maizuru":
        return {
            "epochs": 500,
            "lr": 0.0008,
            "encoder_d": 48,
            "encoder_blocks": 2,
            "encoder_heads": 4,
            "encoder_dropout": 0.15,
            "takens_lags": (1, 2, 4, 8),
            "d_species_f": 16,
            "f_visible_layers": 2,
            "f_visible_top_k": 3,
            "d_species_G": 10,
            "G_field_layers": 1,
            "G_field_top_k": 3,
            "beta_kl": 0.03,
            "lam_cf": 5.0,
            "min_energy": 0.02,
            "lam_hdyn": 0.1,
            "lam_rmse_log": 0.1,
            "lam_hf": 0.0,
        }
    if dataset == "huisman":
        return {
            "epochs": 500,
            "lr": 0.0008,
            "encoder_d": 64,
            "encoder_blocks": 2,
            "encoder_heads": 4,
            "encoder_dropout": 0.1,
            "takens_lags": (1, 2, 4, 8),
            "d_species_f": 20,
            "f_visible_layers": 2,
            "f_visible_top_k": 4,
            "d_species_G": 12,
            "G_field_layers": 1,
            "G_field_top_k": 3,
            "beta_kl": 0.03,
            "lam_cf": 5.0,
            "min_energy": 0.02,
            "lam_hdyn": 0.2,
            "lam_rmse_log": 0.1,
            "lam_hf": 0.0,
        }
    return {
        "epochs": 300,
        "lr": 0.0008,
        "encoder_d": 64,
        "encoder_blocks": 2,
        "encoder_heads": 4,
        "encoder_dropout": 0.1,
        "takens_lags": (1, 2, 4, 8),
        "d_species_f": 20,
        "f_visible_layers": 2,
        "f_visible_top_k": 4,
        "d_species_G": 12,
        "G_field_layers": 1,
        "G_field_top_k": 3,
        "beta_kl": 0.03,
        "lam_cf": 5.0,
        "min_energy": 0.02,
        "lam_hdyn": 0.2,
        "lam_rmse_log": 0.1,
        "lam_hf": 0.0,
    }


def apply_main_variant(base: Dict, method: str) -> Dict:
    cfg = dict(base)
    cfg["method_name"] = method
    cfg["use_formula_hints"] = True
    cfg["alternating"] = True
    cfg["alt_phase_a"] = 5
    cfg["alt_phase_b"] = 1
    cfg["use_hdyn"] = True
    cfg["rollout_max_k"] = 3
    cfg["lam_necessary"] = cfg["lam_cf"]
    cfg["lam_shuffle"] = cfg["lam_cf"] * 0.6
    cfg["margin_null"] = 0.002 if cfg.get("epochs", 300) >= 300 else 0.003
    cfg["margin_shuf"] = 0.001 if cfg.get("epochs", 300) >= 300 else 0.002
    cfg["model_kind"] = "residual"

    if method == MAIN_METHOD:
        return cfg
    if method == "minus_null":
        cfg["lam_necessary"] = 0.0
        return cfg
    if method == "minus_shuffle":
        cfg["lam_shuffle"] = 0.0
        return cfg
    if method == "minus_both_counterfactual":
        cfg["lam_necessary"] = 0.0
        cfg["lam_shuffle"] = 0.0
        return cfg
    if method == "minus_residual_decomp":
        cfg["model_kind"] = "concat"
        return cfg
    if method == "f_only":
        cfg["model_kind"] = "f_only"
        cfg["lam_necessary"] = 0.0
        cfg["lam_shuffle"] = 0.0
        cfg["use_hdyn"] = False
        return cfg
    if method == "minus_formula_hints":
        cfg["use_formula_hints"] = False
        return cfg
    if method == "rollout_0":
        cfg["rollout_max_k"] = 0
        return cfg
    if method == "rollout_5":
        cfg["rollout_max_k"] = 5
        return cfg
    if method == "minus_takens":
        cfg["takens_lags"] = ()
        return cfg
    if method == "minus_alt5":
        cfg["alternating"] = False
        return cfg
    if method == "minus_hdyn":
        cfg["use_hdyn"] = False
        return cfg
    raise ValueError(f"Unknown main/ablation method: {method}")


def build_main_model(task: Task, cfg: Dict, device: str) -> nn.Module:
    common = dict(
        num_visible=task.visible.shape[1],
        encoder_d=cfg["encoder_d"],
        encoder_blocks=cfg["encoder_blocks"],
        encoder_heads=cfg["encoder_heads"],
        takens_lags=cfg["takens_lags"],
        encoder_dropout=cfg["encoder_dropout"],
        prior_std=1.0,
        gnn_backbone="mlp",
        point_estimate=False,
    )
    if cfg["model_kind"] == "residual":
        model = CVHI_Residual(
            **common,
            d_species_f=cfg["d_species_f"],
            f_visible_layers=cfg["f_visible_layers"],
            f_visible_top_k=cfg["f_visible_top_k"],
            d_species_G=cfg["d_species_G"],
            G_field_layers=cfg["G_field_layers"],
            G_field_top_k=cfg["G_field_top_k"],
            use_formula_hints=cfg["use_formula_hints"],
            use_G_field=True,
            num_mixture_components=1,
            G_anchor_first=True,
            G_anchor_sign=+1,
        )
    elif cfg["model_kind"] == "concat":
        model = DirectConcatModel(
            **common,
            d_species=cfg["d_species_f"],
            layers=cfg["f_visible_layers"],
            top_k=cfg["f_visible_top_k"],
            use_formula_hints=cfg["use_formula_hints"],
        )
    elif cfg["model_kind"] == "f_only":
        model = FOnlyModel(
            **common,
            d_species_f=cfg["d_species_f"],
            f_visible_layers=cfg["f_visible_layers"],
            f_visible_top_k=cfg["f_visible_top_k"],
            d_species_G=cfg["d_species_G"],
            G_field_layers=cfg["G_field_layers"],
            G_field_top_k=cfg["G_field_top_k"],
            use_formula_hints=cfg["use_formula_hints"],
            use_G_field=False,
            num_mixture_components=1,
            G_anchor_first=True,
            G_anchor_sign=+1,
        )
    else:
        raise ValueError(f"Unknown model kind: {cfg['model_kind']}")
    return model.to(device)


def rollout_weights_for_k(k: int) -> Tuple[float, ...]:
    if k <= 0:
        return ()
    if k == 1:
        return (1.0,)
    if k == 3:
        return (1.0, 0.5, 0.25)
    vals = np.linspace(1.0, 0.2, k)
    return tuple(float(v) for v in vals)


def extract_mu(out: Dict[str, torch.Tensor]) -> torch.Tensor:
    return out["mu"]


def train_main_seed(task: Task, dataset_cfg: Dict, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    T, N = task.visible.shape
    te = task_train_end(task)
    x_full = to_tensor(task.visible, device).unsqueeze(0)
    x_train_clean = x_full[:, :te]
    x_val_clean = x_full[:, te:]
    model = build_main_model(task, dataset_cfg, device)
    hdyn = LatentDynamicsNet(N, 32).to(device) if dataset_cfg["use_hdyn"] else None

    base_params, enc_params = model_param_groups(model)
    all_params = list(model.parameters()) + ([] if hdyn is None else list(hdyn.parameters()))
    opt = torch.optim.AdamW(all_params, lr=dataset_cfg["lr"], weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: cosine_lr(s, dataset_cfg["epochs"]))

    warmup = int(0.2 * dataset_cfg["epochs"])
    ramp = max(1, int(0.2 * dataset_cfg["epochs"]))
    best_val = float("inf")
    best_state = None
    best_hdyn_state = None
    best_epoch = -1

    for epoch in range(dataset_cfg["epochs"]):
        if hasattr(model, "G_anchor_alpha"):
            model.G_anchor_alpha = alpha_schedule(epoch, dataset_cfg["epochs"])

        if epoch < warmup:
            h_weight = 0.0
            rollout_k = 0
        else:
            post = epoch - warmup
            h_weight = min(1.0, post / ramp)
            max_roll = dataset_cfg["rollout_max_k"]
            if max_roll <= 0:
                rollout_k = 0
            else:
                roll_frac = min(1.0, post / max(1, dataset_cfg["epochs"] - warmup) * 2.0)
                rollout_k = max(1 if h_weight > 0 else 0, int(round(roll_frac * max_roll)))
                rollout_k = min(rollout_k, max_roll)

        if dataset_cfg["alternating"] and epoch >= warmup:
            cycle_len = dataset_cfg["alt_phase_a"] + dataset_cfg["alt_phase_b"]
            pos = (epoch - warmup) % cycle_len
            if pos < dataset_cfg["alt_phase_a"]:
                for p in enc_params:
                    p.requires_grad_(False)
                for p in base_params:
                    p.requires_grad_(True)
                if hdyn is not None:
                    for p in hdyn.parameters():
                        p.requires_grad_(False)
            else:
                for p in enc_params:
                    p.requires_grad_(True)
                for p in base_params:
                    p.requires_grad_(False)
                if hdyn is not None:
                    for p in hdyn.parameters():
                        p.requires_grad_(True)
        else:
            for p in model.parameters():
                p.requires_grad_(True)
            if hdyn is not None:
                for p in hdyn.parameters():
                    p.requires_grad_(True)

        if epoch > warmup:
            mask = (torch.rand_like(x_train_clean[:, :, :1]) > 0.05).float()
            train_mean = x_train_clean.mean(dim=1, keepdim=True)
            x_train = x_train_clean * mask + (1 - mask) * train_mean
        else:
            x_train = x_train_clean

        model.train()
        if hdyn is not None:
            hdyn.train()
        opt.zero_grad()
        out = model(x_train, n_samples=2, rollout_K=rollout_k)
        losses = model.loss(
            out,
            beta_kl=dataset_cfg["beta_kl"],
            free_bits=0.02,
            margin_null=dataset_cfg["margin_null"],
            margin_shuf=dataset_cfg["margin_shuf"],
            lam_necessary=dataset_cfg["lam_necessary"],
            lam_shuffle=dataset_cfg["lam_shuffle"],
            lam_energy=2.0,
            min_energy=dataset_cfg["min_energy"],
            lam_smooth=0.02,
            lam_sparse=0.02,
            h_weight=h_weight,
            lam_rollout=0.5 * h_weight if rollout_k > 0 else 0.0,
            rollout_weights=rollout_weights_for_k(rollout_k),
            lam_hf=dataset_cfg["lam_hf"],
            lowpass_sigma=6.0,
            lam_rmse_log=dataset_cfg["lam_rmse_log"],
            n_recon_channels=task.n_recon_channels,
        )
        total = losses["total"]
        hdyn_loss = torch.tensor(0.0, device=device)
        if hdyn is not None and h_weight > 0:
            hm = out["h_samples"].mean(dim=0)
            hp = hdyn(hm[:, :-1], x_train[:, :-1])
            tgt = hm[:, 1:].detach() if epoch < warmup + 100 else hm[:, 1:]
            hdyn_loss = F.mse_loss(hp, tgt)
            total = total + dataset_cfg["lam_hdyn"] * h_weight * hdyn_loss
        total.backward()
        torch.nn.utils.clip_grad_norm_(all_params, 1.0)
        opt.step()
        sched.step()

        with torch.no_grad():
            if x_val_clean.shape[1] > 2:
                model.eval()
                out_val = model(x_val_clean, n_samples=2, rollout_K=min(rollout_k, max(0, x_val_clean.shape[1] - 2)))
                val_losses = model.loss(
                    out_val,
                    beta_kl=dataset_cfg["beta_kl"],
                    free_bits=0.02,
                    margin_null=dataset_cfg["margin_null"],
                    margin_shuf=dataset_cfg["margin_shuf"],
                    lam_necessary=dataset_cfg["lam_necessary"],
                    lam_shuffle=dataset_cfg["lam_shuffle"],
                    lam_energy=2.0,
                    min_energy=dataset_cfg["min_energy"],
                    lam_smooth=0.02,
                    lam_sparse=0.02,
                    h_weight=1.0,
                    lam_rollout=0.5 if rollout_k > 0 else 0.0,
                    rollout_weights=rollout_weights_for_k(min(rollout_k, max(0, x_val_clean.shape[1] - 2))),
                    lam_hf=dataset_cfg["lam_hf"],
                    lowpass_sigma=6.0,
                    lam_rmse_log=0.0,
                    n_recon_channels=task.n_recon_channels,
                )
                val_recon = float(val_losses["recon_full"])
            else:
                val_recon = float(losses["recon_full"])
        if epoch > warmup + 15 and val_recon < best_val:
            best_val = val_recon
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_hdyn_state = None if hdyn is None else {k: v.detach().cpu().clone() for k, v in hdyn.state_dict().items()}
            best_epoch = epoch + 1

    if best_state is not None:
        model.load_state_dict(best_state)
    if hdyn is not None and best_hdyn_state is not None:
        hdyn.load_state_dict(best_hdyn_state)

    model.eval()
    with torch.no_grad():
        out_eval = model(x_full, n_samples=30, rollout_K=min(3, max(0, T - 2)))
        h_mean = out_eval["h_samples"].mean(dim=0)[0].cpu().numpy()
    h_scaled, metrics = fit_affine_on_train(h_mean, task.hidden, te)
    d_ratio = float("nan")
    if isinstance(model, CVHI_Residual) and dataset_cfg["model_kind"] == "residual":
        diag = hidden_true_substitution(model, task.visible, task.hidden, device=device)
        d_ratio = safe_float(diag["recon_true_scaled"] / max(diag["recon_encoder"], 1e-8))

    result = {
        "seed": seed,
        "task_name": task.task_name,
        "dataset": task.dataset,
        "method": dataset_cfg["method_name"],
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "val_recon": best_val,
        "d_ratio": d_ratio,
        "scale_a": metrics["scale_a"],
        "scale_b": metrics["scale_b"],
        "train_end": te,
        "aligned_length": metrics["aligned_length"],
        "best_epoch": best_epoch,
        "hdyn_loss_last": safe_float(hdyn_loss.item() if torch.is_tensor(hdyn_loss) else hdyn_loss),
        "trajectory_offset": 0,
    }
    result["h_scaled"] = h_scaled
    result["h_raw"] = h_mean.astype(np.float32)
    del model, hdyn
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def fit_pca_on_train_transform_full(features: np.ndarray, train_end: int) -> np.ndarray:
    feats = np.asarray(features, dtype=np.float64)
    te = max(2, min(train_end, len(feats)))
    pca = PCA(n_components=1)
    pca.fit(feats[:te])
    return pca.transform(feats).squeeze(-1).astype(np.float32)


def var_pca_seed(task: Task, seed: int) -> Dict:
    _ = seed
    x = np.asarray(task.visible, dtype=np.float32)
    T, N = x.shape
    te = task_train_end(task)
    p = 4
    safe = np.maximum(x, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1])
    rows = []
    targets = []
    for t in range(p, len(log_ratio)):
        rows.append(np.concatenate([x[t - lag] for lag in range(p)]))
        targets.append(log_ratio[t])
    X = np.asarray(rows)
    Y = np.asarray(targets)
    train_rows = max(2, te - p)
    X_aug = np.column_stack([X[:train_rows], np.ones(train_rows)])
    coef, _, _, _ = np.linalg.lstsq(X_aug, Y[:train_rows], rcond=None)
    Y_pred = np.column_stack([X, np.ones(len(X))]) @ coef
    residual = Y - Y_pred
    h_est = fit_pca_on_train_transform_full(residual, train_rows)
    hidden_aligned = task.hidden[p + 1:p + 1 + len(h_est)]
    h_scaled, metrics = fit_affine_on_train(h_est, hidden_aligned, train_rows)
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": p + 1,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), p + 1),
        "h_raw": pad_to_length(h_est, len(task.hidden), p + 1),
        "deterministic": True,
    }


def mlp_pca_seed(task: Task, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.asarray(task.visible, dtype=np.float32)
    T, N = x.shape
    te = task_train_end(task)
    safe = np.maximum(x, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1]).astype(np.float32)
    x_in = to_tensor(x[:-1], device)
    y_out = to_tensor(log_ratio, device)
    train_len = max(8, te - 1)
    model = nn.Sequential(
        nn.Linear(N, 64), nn.ReLU(),
        nn.Linear(64, 64), nn.ReLU(),
        nn.Linear(64, N),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    for _ in range(200):
        pred = model(x_in[:train_len])
        loss = F.mse_loss(pred, y_out[:train_len])
        opt.zero_grad()
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        pred_all = model(x_in).cpu().numpy()
    residual = log_ratio - pred_all
    h_est = fit_pca_on_train_transform_full(residual, train_len)
    hidden_aligned = task.hidden[1:1 + len(h_est)]
    h_scaled, metrics = fit_affine_on_train(h_est, hidden_aligned, train_len)
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": 1,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), 1),
        "h_raw": pad_to_length(h_est, len(task.hidden), 1),
    }


def build_embedding(series: np.ndarray, E: int, tau: int = 1) -> Optional[np.ndarray]:
    T = len(series)
    n = T - (E - 1) * tau
    if n <= 0:
        return None
    return np.column_stack([series[(E - 1) * tau - l * tau:T - l * tau] for l in range(E)])


def simplex_predict(lib_data: np.ndarray, pred_data: np.ndarray, E: int, nn_k: Optional[int] = None) -> np.ndarray:
    if nn_k is None:
        nn_k = E + 1
    T_pred = len(pred_data)
    predictions = np.full(T_pred, np.nan, dtype=np.float32)
    for i in range(T_pred):
        dists = np.sqrt(((lib_data[:-1] - pred_data[i]) ** 2).sum(axis=1))
        idx = np.argsort(dists)[:nn_k]
        d_nn = dists[idx]
        d_min = d_nn[0]
        if d_min < 1e-10:
            weights = np.zeros(nn_k, dtype=np.float32)
            weights[0] = 1.0
        else:
            weights = np.exp(-d_nn / d_min)
        weights = weights / (weights.sum() + 1e-10)
        predictions[i] = np.sum(weights * lib_data[idx + 1, 0])
    return predictions


def choose_simplex_E(task: Task, E_range: Sequence[int] = (2, 3, 4, 5)) -> int:
    x = np.asarray(task.visible, dtype=np.float32)
    te = task_train_end(task)
    best_E = E_range[0]
    best_skill = -float("inf")
    for E in E_range:
        skills = []
        for j in range(x.shape[1]):
            emb = build_embedding(x[:, j], E, tau=1)
            if emb is None:
                continue
            offset = E - 1
            train_len = te - offset
            if train_len < E + 3:
                continue
            lib = emb[:train_len]
            pred = emb[:train_len]
            preds = simplex_predict(lib, pred, E)
            actual = x[offset:offset + len(preds), j]
            skill = corrcoef_safe(np.nan_to_num(preds, nan=0.0), actual)
            if not math.isnan(skill):
                skills.append(skill)
        if skills:
            avg = float(np.mean(skills))
            if avg > best_skill:
                best_skill = avg
                best_E = E
    return best_E


def edm_simplex_seed(task: Task, seed: int) -> Dict:
    _ = seed
    x = np.asarray(task.visible, dtype=np.float32)
    te = task_train_end(task)
    E = choose_simplex_E(task)
    residuals = []
    min_len = None
    offset = E - 1
    for j in range(x.shape[1]):
        emb = build_embedding(x[:, j], E, tau=1)
        if emb is None:
            continue
        train_len = te - offset
        if train_len < E + 3:
            continue
        lib = emb[:train_len]
        pred = emb
        preds = simplex_predict(lib, pred, E)
        actual = x[offset:offset + len(preds), j]
        resid = np.nan_to_num(actual - preds, nan=0.0)
        residuals.append(resid)
        min_len = len(resid) if min_len is None else min(min_len, len(resid))
    if not residuals or min_len is None:
        return {"seed": seed, "pearson_all": float("nan"), "pearson_val": float("nan"),
                "trajectory_offset": offset, "h_scaled": np.full(len(task.hidden), np.nan), "h_raw": np.full(len(task.hidden), np.nan),
                "deterministic": True}
    resid_mat = np.column_stack([r[:min_len] for r in residuals])
    train_len = max(2, te - offset)
    h_est = fit_pca_on_train_transform_full(resid_mat, train_len)
    hidden_aligned = task.hidden[offset:offset + len(h_est)]
    h_scaled, metrics = fit_affine_on_train(h_est, hidden_aligned, train_len)
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": offset,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), offset),
        "h_raw": pad_to_length(h_est, len(task.hidden), offset),
        "edm_E": E,
        "deterministic": True,
    }


def simplex_predict_mve(lib_emb: np.ndarray, lib_target: np.ndarray, pred_emb: np.ndarray, E: int, nn_k: Optional[int] = None) -> np.ndarray:
    if nn_k is None:
        nn_k = E + 1
    preds = np.full(len(pred_emb), np.nan, dtype=np.float32)
    for i in range(len(pred_emb)):
        dists = np.sqrt(((lib_emb[:-1] - pred_emb[i]) ** 2).sum(axis=1))
        idx = np.argsort(dists)[:nn_k]
        d_nn = dists[idx]
        d_min = d_nn[0]
        if d_min < 1e-10:
            weights = np.zeros(nn_k, dtype=np.float32)
            weights[0] = 1.0
        else:
            weights = np.exp(-d_nn / d_min)
        weights = weights / (weights.sum() + 1e-10)
        if idx.max() + 1 < len(lib_target):
            preds[i] = np.sum(weights * lib_target[idx + 1])
    return preds


def multiview_seed(task: Task, seed: int) -> Dict:
    _ = seed
    x = np.asarray(task.visible, dtype=np.float32)
    T, N = x.shape
    te = task_train_end(task)
    E = min(3, N)
    top_k = max(1, int(np.sqrt(N)))
    combos = list(combinations(range(N), E))
    if len(combos) > 200:
        rng = np.random.RandomState(42)
        idx = rng.choice(len(combos), 200, replace=False)
        combos = [combos[i] for i in idx]
    residuals = []
    for target_j in range(N):
        view_preds = []
        view_skills = []
        for combo in combos:
            emb = np.column_stack([x[:, c] for c in combo])
            emb_cur = emb[1:]
            target_next = x[1:, target_j]
            lib_emb = emb_cur[:te - 1]
            lib_target = target_next[:te - 1]
            if len(lib_emb) < E + 3:
                continue
            preds = simplex_predict_mve(lib_emb, lib_target, emb_cur, E)
            skill = corrcoef_safe(np.nan_to_num(preds[:te - 1], nan=0.0), target_next[:te - 1])
            if math.isnan(skill):
                continue
            view_preds.append(preds)
            view_skills.append(skill)
        if not view_preds:
            residuals.append(np.zeros(T - 1, dtype=np.float32))
            continue
        order = np.argsort(view_skills)[::-1][:top_k]
        w = np.maximum(np.asarray([view_skills[i] for i in order]), 0.0)
        if np.sum(w) < 1e-8:
            w = np.ones_like(w)
        w = w / np.sum(w)
        avg_pred = np.zeros(T - 1, dtype=np.float32)
        for ww, idx in zip(w, order):
            avg_pred += ww * np.nan_to_num(view_preds[idx], nan=0.0)
        residuals.append(x[1:, target_j] - avg_pred)
    resid_mat = np.column_stack(residuals)
    train_len = max(2, te - 1)
    h_est = fit_pca_on_train_transform_full(resid_mat, train_len)
    hidden_aligned = task.hidden[1:1 + len(h_est)]
    h_scaled, metrics = fit_affine_on_train(h_est, hidden_aligned, train_len)
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": 1,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), 1),
        "h_raw": pad_to_length(h_est, len(task.hidden), 1),
        "deterministic": True,
    }


def supervised_ridge_seed(task: Task, seed: int) -> Dict:
    _ = seed
    x = np.asarray(task.visible, dtype=np.float32)
    y = np.asarray(task.hidden, dtype=np.float32)
    te = task_train_end(task)
    p = 4
    rows = []
    targets = []
    for t in range(p, len(x)):
        rows.append(np.concatenate([x[t - lag] for lag in range(p)]))
        targets.append(y[t])
    X = np.asarray(rows)
    Y = np.asarray(targets)
    train_rows = max(8, te - p)
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X[:train_rows])
    model = Ridge(alpha=1.0)
    model.fit(Xtr, Y[:train_rows])
    pred = model.predict(scaler.transform(X))
    hidden_aligned = y[p:p + len(pred)]
    h_scaled, metrics = fit_affine_on_train(pred, hidden_aligned, train_rows)
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": p,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), p),
        "h_raw": pad_to_length(pred.astype(np.float32), len(task.hidden), p),
        "deterministic": True,
    }


def lstm_seed(task: Task, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.asarray(task.visible, dtype=np.float32)
    T, N = x.shape
    te = task_train_end(task)
    safe = np.maximum(x, 1e-6)
    log_ratio = np.log(safe[1:] / safe[:-1]).astype(np.float32)
    train_len = max(8, te - 1)
    x_seq = to_tensor(x[:-1], device).unsqueeze(0)
    y_seq = to_tensor(log_ratio, device).unsqueeze(0)
    lstm = nn.LSTM(N, 32, batch_first=True).to(device)
    fc = nn.Linear(32, N).to(device)
    params = list(lstm.parameters()) + list(fc.parameters())
    opt = torch.optim.Adam(params, lr=0.001)
    for _ in range(300):
        out, _ = lstm(x_seq[:, :train_len])
        pred = fc(out)
        loss = F.mse_loss(pred, y_seq[:, :train_len])
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        hidden_states, _ = lstm(x_seq)
    hs = hidden_states[0].cpu().numpy()
    h_est = fit_pca_on_train_transform_full(hs, train_len)
    hidden_aligned = task.hidden[:-1][:len(h_est)]
    h_scaled, metrics = fit_affine_on_train(h_est, hidden_aligned, train_len)
    del lstm, fc
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": 0,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), 0),
        "h_raw": pad_to_length(h_est, len(task.hidden), 0),
    }


class ODEFunc(nn.Module):
    def __init__(self, dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, dim),
        )

    def forward(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


def neural_ode_seed(task: Task, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.asarray(task.visible, dtype=np.float32)
    T, N = x.shape
    te = task_train_end(task)
    train_len = te
    x_all = to_tensor(x, device)
    func = ODEFunc(N, 64).to(device)
    opt = torch.optim.Adam(func.parameters(), lr=0.003)
    window = min(20, max(5, train_len - 2))
    for _ in range(200):
        total = 0.0
        n_starts = min(16, max(1, train_len - window))
        starts = np.random.choice(max(1, train_len - window), n_starts, replace=False)
        for s in starts:
            t_w = torch.linspace(0, 1, window, device=device)
            pred = odeint(func, x_all[s], t_w, method="rk4")
            target = x_all[s:s + window]
            total = total + ((pred - target) ** 2).mean()
        loss = total / max(1, n_starts)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(func.parameters(), 1.0)
        opt.step()
    with torch.no_grad():
        residuals = []
        for t in range(T - 1):
            t_w = torch.tensor([0.0, 1.0], device=device)
            pred = odeint(func, x_all[t], t_w, method="rk4")[1]
            residuals.append((x_all[t + 1] - pred).cpu().numpy())
    residual = np.asarray(residuals)
    h_est = fit_pca_on_train_transform_full(residual, max(2, te - 1))
    hidden_aligned = task.hidden[1:1 + len(h_est)]
    h_scaled, metrics = fit_affine_on_train(h_est, hidden_aligned, max(2, te - 1))
    del func
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": 1,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), 1),
        "h_raw": pad_to_length(h_est, len(task.hidden), 1),
    }


class GRUEncoder(nn.Module):
    def __init__(self, n_vis: int, latent_dim: int, hidden: int = 32):
        super().__init__()
        self.gru = nn.GRU(n_vis, hidden, batch_first=True)
        self.fc = nn.Linear(hidden, latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.fc(h.squeeze(0))


def latent_ode_seed(task: Task, seed: int, device: str) -> Dict:
    torch.manual_seed(seed)
    np.random.seed(seed)
    x = np.asarray(task.visible, dtype=np.float32)
    T, N = x.shape
    te = task_train_end(task)
    train_len = te
    x_all = to_tensor(x, device)
    latent_dim = 4
    aug_dim = N + latent_dim
    func = ODEFunc(aug_dim, 64).to(device)
    encoder = GRUEncoder(N, latent_dim, 32).to(device)
    decoder = nn.Linear(aug_dim, N).to(device)
    params = list(func.parameters()) + list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=0.003)
    window = min(20, max(5, train_len - 2))
    for _ in range(200):
        total = 0.0
        n_starts = min(8, max(1, train_len - window))
        starts = np.random.choice(max(1, train_len - window), n_starts, replace=False)
        for s in starts:
            context_len = min(s + 1, 50)
            context = x_all[max(0, s - context_len + 1):s + 1].unsqueeze(0)
            z0_lat = encoder(context).squeeze(0)
            z0 = torch.cat([x_all[s], z0_lat], dim=0)
            t_w = torch.linspace(0, 1, window, device=device)
            z_traj = odeint(func, z0, t_w, method="rk4")
            pred = decoder(z_traj)
            target = x_all[s:s + window]
            total = total + ((pred - target) ** 2).mean()
        loss = total / max(1, n_starts)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    with torch.no_grad():
        latent_full = np.zeros((T, latent_dim), dtype=np.float32)
        counts = np.zeros(T, dtype=np.float32)
        stride = max(1, window // 2)
        for s in range(0, max(1, T - window + 1), stride):
            context_len = min(s + 1, 50)
            context = x_all[max(0, s - context_len + 1):s + 1].unsqueeze(0)
            z0_lat = encoder(context).squeeze(0)
            z0 = torch.cat([x_all[s], z0_lat], dim=0)
            t_w = torch.linspace(0, 1, min(window, T - s), device=device)
            z_traj = odeint(func, z0, t_w, method="rk4")
            lat = z_traj[:, N:].cpu().numpy()
            w = min(len(lat), T - s)
            latent_full[s:s + w] += lat[:w]
            counts[s:s + w] += 1
        counts = np.maximum(counts, 1.0)
        latent_full /= counts[:, None]
    h_est = fit_pca_on_train_transform_full(latent_full, te)
    h_scaled, metrics = fit_affine_on_train(h_est, task.hidden[:len(h_est)], te)
    del func, encoder, decoder
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "seed": seed,
        "pearson_all": metrics["pearson_all"],
        "pearson_val": metrics["pearson_val"],
        "trajectory_offset": 0,
        "h_scaled": pad_to_length(h_scaled, len(task.hidden), 0),
        "h_raw": pad_to_length(h_est, len(task.hidden), 0),
    }


def method_runner(method: str, task: Task, seed: int, device: str) -> Dict:
    if method == "var_pca":
        return var_pca_seed(task, seed)
    if method == "mlp_pca":
        return mlp_pca_seed(task, seed, device)
    if method == "edm_simplex":
        return edm_simplex_seed(task, seed)
    if method == "mve":
        return multiview_seed(task, seed)
    if method == "supervised_ridge":
        return supervised_ridge_seed(task, seed)
    if method == "lstm":
        return lstm_seed(task, seed, device)
    if method == "neural_ode":
        return neural_ode_seed(task, seed, device)
    if method == "latent_ode":
        return latent_ode_seed(task, seed, device)
    raise ValueError(f"Unknown baseline method: {method}")


def seed_dir(root: Path, seed: int) -> Path:
    return root / f"seed_{seed:05d}"


def write_seed_result(root: Path, payload: Dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    out = dict(payload)
    out.pop("h_scaled", None)
    out.pop("h_raw", None)
    (root / "metrics.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [
        f"# Seed {payload['seed']}",
        "",
        f"- pearson_all: {payload.get('pearson_all', float('nan')):+.4f}",
        f"- pearson_val: {payload.get('pearson_val', float('nan')):+.4f}",
    ]
    if "val_recon" in payload:
        lines.append(f"- val_recon: {payload['val_recon']:.6f}")
    if "d_ratio" in payload:
        lines.append(f"- d_ratio: {payload['d_ratio']:.4f}")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def task_result_root(group: str, method: str, dataset: str, task_name: str) -> Path:
    return RESULTS_ROOT / group / method / dataset / task_name


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    best_key = "pearson_val"
    usable = [r for r in seed_results if not math.isnan(safe_float(r.get(best_key, float("nan"))))]
    if not usable:
        usable = seed_results
        best_key = "pearson_all"
    best = max(usable, key=lambda r: safe_float(r.get(best_key, -1e9)))
    agg = {
        "n_seeds": len(seed_results),
        "pearson_all_mean": float(np.nanmean([safe_float(r.get("pearson_all")) for r in seed_results])),
        "pearson_all_std": float(np.nanstd([safe_float(r.get("pearson_all")) for r in seed_results])),
        "pearson_val_mean": float(np.nanmean([safe_float(r.get("pearson_val")) for r in seed_results])),
        "pearson_val_std": float(np.nanstd([safe_float(r.get("pearson_val")) for r in seed_results])),
        "best_seed": int(best["seed"]),
        "best_key": best_key,
        "best_pearson_all": safe_float(best.get("pearson_all")),
        "best_pearson_val": safe_float(best.get("pearson_val")),
        "trajectory_offset": int(best.get("trajectory_offset", 0)),
        "best_h_scaled": np.asarray(best["h_scaled"], dtype=np.float32),
        "best_h_raw": np.asarray(best["h_raw"], dtype=np.float32),
    }
    if "d_ratio" in seed_results[0]:
        dvals = [safe_float(r.get("d_ratio")) for r in seed_results]
        if any(not math.isnan(v) for v in dvals):
            agg["d_ratio_mean"] = float(np.nanmean(dvals))
            agg["d_ratio_std"] = float(np.nanstd(dvals))
    if "val_recon" in seed_results[0]:
        agg["val_recon_mean"] = float(np.nanmean([safe_float(r.get("val_recon")) for r in seed_results]))
        agg["val_recon_std"] = float(np.nanstd([safe_float(r.get("val_recon")) for r in seed_results]))
    return agg


def save_task_aggregate(root: Path, task: Task, method: str, seed_results: List[Dict], agg: Dict) -> None:
    root.mkdir(parents=True, exist_ok=True)
    save_json = {k: v for k, v in agg.items() if k not in ("best_h_scaled", "best_h_raw")}
    save_json["dataset"] = task.dataset
    save_json["task_name"] = task.task_name
    save_json["hidden_name"] = task.hidden_name
    save_json["method"] = method
    save_json["time_length"] = int(len(task.hidden))
    (root / "aggregate.json").write_text(json.dumps(save_json, indent=2, ensure_ascii=False), encoding="utf-8")
    np.savez(
        root / "best_trajectory.npz",
        time_axis=np.asarray(task.time_axis),
        hidden_true=np.asarray(task.hidden, dtype=np.float32),
        hidden_pred_scaled=np.asarray(agg["best_h_scaled"], dtype=np.float32),
        hidden_pred_raw=np.asarray(agg["best_h_raw"], dtype=np.float32),
    )
    lines = [
        f"# {dataset_display_name(task.dataset)} / {task.task_name} / {method}",
        "",
        f"- seeds: {len(seed_results)}",
        f"- pearson_all_mean: {agg['pearson_all_mean']:+.4f} +- {agg['pearson_all_std']:.4f}",
        f"- pearson_val_mean: {agg['pearson_val_mean']:+.4f} +- {agg['pearson_val_std']:.4f}",
        f"- best_seed: {agg['best_seed']}",
    ]
    if "d_ratio_mean" in agg:
        lines.append(f"- d_ratio_mean: {agg['d_ratio_mean']:.4f} +- {agg['d_ratio_std']:.4f}")
    if "val_recon_mean" in agg:
        lines.append(f"- val_recon_mean: {agg['val_recon_mean']:.6f}")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_dataset_summary(root: Path, dataset: str, method: str, task_rows: List[Dict]) -> None:
    lines = [
        f"# {dataset_display_name(dataset)} / {method}",
        "",
        "| Task | P(all) mean | P(val) mean | Best seed |",
        "|---|---|---|---|",
    ]
    for row in task_rows:
        lines.append(
            f"| {row['task_name']} | {row['pearson_all_mean']:+.4f} | {row['pearson_val_mean']:+.4f} | {row['best_seed']} |"
        )
    lines.append("")
    lines.append(f"**Overall P(all)**: {np.nanmean([r['pearson_all_mean'] for r in task_rows]):+.4f}")
    lines.append(f"**Overall P(val)**: {np.nanmean([r['pearson_val_mean'] for r in task_rows]):+.4f}")
    (root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    out = {
        "dataset": dataset,
        "method": method,
        "task_rows": task_rows,
        "overall_pearson_all_mean": float(np.nanmean([r["pearson_all_mean"] for r in task_rows])),
        "overall_pearson_val_mean": float(np.nanmean([r["pearson_val_mean"] for r in task_rows])),
    }
    (root / "aggregate.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")


def run_main_or_ablation_dataset(dataset: str, method: str, seeds: Sequence[int], skip_existing: bool, smoke: bool = False) -> None:
    device = device_name()
    tasks = load_tasks(dataset)
    if smoke:
        tasks = tasks[:1]
        seeds = seeds[:1]
    base = base_preset(dataset)
    cfg = apply_main_variant(base, method)
    dataset_root = RESULTS_ROOT / ("main" if method == MAIN_METHOD else "ablations") / method / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    task_rows = []
    for task in tasks:
        root = task_result_root("main" if method == MAIN_METHOD else "ablations", method, dataset, task.task_name)
        agg_path = root / "aggregate.json"
        if skip_existing and agg_path.exists():
            task_rows.append(json.loads(agg_path.read_text(encoding="utf-8")))
            continue
        seed_results = []
        for seed in seeds:
            sd = seed_dir(root, seed)
            metrics_path = sd / "metrics.json"
            if skip_existing and metrics_path.exists():
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                traj = np.load(root / "best_trajectory.npz")
                payload["h_scaled"] = traj["hidden_pred_scaled"]
                payload["h_raw"] = traj["hidden_pred_raw"]
            else:
                payload = train_main_seed(task, cfg, seed, device)
                write_seed_result(sd, payload)
            seed_results.append(payload)
        agg = aggregate_seed_results(seed_results)
        save_task_aggregate(root, task, method, seed_results, agg)
        row = {k: v for k, v in agg.items() if k not in ("best_h_scaled", "best_h_raw")}
        row["task_name"] = task.task_name
        row["dataset"] = task.dataset
        row["method"] = method
        task_rows.append(row)
        print(f"[{dataset}/{method}/{task.task_name}] P(all)={agg['pearson_all_mean']:+.4f}  P(val)={agg['pearson_val_mean']:+.4f}")
    save_dataset_summary(dataset_root, dataset, method, task_rows)


def run_baseline_dataset(dataset: str, method: str, seeds: Sequence[int], skip_existing: bool, smoke: bool = False) -> None:
    device = device_name()
    tasks = load_tasks(dataset)
    if smoke:
        tasks = tasks[:1]
        seeds = seeds[:1]
    dataset_root = RESULTS_ROOT / "baselines" / method / dataset
    dataset_root.mkdir(parents=True, exist_ok=True)
    task_rows = []
    for task in tasks:
        root = task_result_root("baselines", method, dataset, task.task_name)
        agg_path = root / "aggregate.json"
        if skip_existing and agg_path.exists():
            task_rows.append(json.loads(agg_path.read_text(encoding="utf-8")))
            continue
        deterministic = method in {"var_pca", "edm_simplex", "mve", "supervised_ridge"}
        seed_results = []
        cached_det = None
        for seed in seeds:
            sd = seed_dir(root, seed)
            metrics_path = sd / "metrics.json"
            if skip_existing and metrics_path.exists():
                payload = json.loads(metrics_path.read_text(encoding="utf-8"))
                traj = np.load(root / "best_trajectory.npz")
                payload["h_scaled"] = traj["hidden_pred_scaled"]
                payload["h_raw"] = traj["hidden_pred_raw"]
            else:
                if deterministic and cached_det is not None:
                    payload = dict(cached_det)
                    payload["seed"] = seed
                else:
                    payload = method_runner(method, task, seed, device)
                    if deterministic:
                        cached_det = dict(payload)
                write_seed_result(sd, payload)
            seed_results.append(payload)
        agg = aggregate_seed_results(seed_results)
        save_task_aggregate(root, task, method, seed_results, agg)
        row = {k: v for k, v in agg.items() if k not in ("best_h_scaled", "best_h_raw")}
        row["task_name"] = task.task_name
        row["dataset"] = task.dataset
        row["method"] = method
        task_rows.append(row)
        print(f"[{dataset}/{method}/{task.task_name}] P(all)={agg['pearson_all_mean']:+.4f}  P(val)={agg['pearson_val_mean']:+.4f}")
    save_dataset_summary(dataset_root, dataset, method, task_rows)


def collect_dataset_overall(group: str, method: str, dataset: str) -> Optional[Dict]:
    agg_path = RESULTS_ROOT / group / method / dataset / "aggregate.json"
    if not agg_path.exists():
        return None
    return json.loads(agg_path.read_text(encoding="utf-8"))


def expected_seed_count(group: str, method: str) -> int:
    if group == "main":
        return len(SEEDS_10)
    if group == "ablations":
        return len(SEEDS_5)
    if group == "baselines":
        return len(SEEDS_10) if method in BASELINE_METHODS_ALL else len(SEEDS_5)
    raise ValueError(f"Unknown group: {group}")


def dataset_task_aggregate_paths(group: str, method: str, dataset: str) -> List[Path]:
    root = RESULTS_ROOT / group / method / dataset
    if not root.exists():
        return []
    out = []
    for child in root.iterdir():
        if child.is_dir():
            agg = child / "aggregate.json"
            if agg.exists():
                out.append(agg)
    return out


def dataset_completion_status(group: str, method: str, dataset: str) -> Dict:
    task_aggs = dataset_task_aggregate_paths(group, method, dataset)
    expected_tasks = len(load_tasks(dataset))
    expected_seeds = expected_seed_count(group, method)
    complete_tasks = 0
    rows = []
    for agg_path in task_aggs:
        payload = json.loads(agg_path.read_text(encoding="utf-8"))
        rows.append(payload)
        if int(payload.get("n_seeds", 0)) >= expected_seeds:
            complete_tasks += 1
    dataset_agg = collect_dataset_overall(group, method, dataset)
    is_complete = (complete_tasks == expected_tasks and expected_tasks > 0)
    return {
        "expected_tasks": expected_tasks,
        "found_task_aggregates": len(task_aggs),
        "complete_tasks": complete_tasks,
        "expected_seeds": expected_seeds,
        "complete": is_complete,
        "dataset_aggregate": dataset_agg,
        "task_rows": rows,
    }


def complete_dataset_overall(group: str, method: str, dataset: str) -> Optional[float]:
    status = dataset_completion_status(group, method, dataset)
    agg = status["dataset_aggregate"]
    if not status["complete"] or agg is None:
        return None
    return agg["overall_pearson_all_mean"]


def load_best_task_curve(group: str, method: str, dataset: str, task_name: str) -> Optional[Dict]:
    root = RESULTS_ROOT / group / method / dataset / task_name
    agg_path = root / "aggregate.json"
    traj_path = root / "best_trajectory.npz"
    if not agg_path.exists() or not traj_path.exists():
        return None
    agg = json.loads(agg_path.read_text(encoding="utf-8"))
    traj = np.load(traj_path)
    return {
        "aggregate": agg,
        "time_axis": traj["time_axis"],
        "hidden_true": traj["hidden_true"],
        "hidden_pred_scaled": traj["hidden_pred_scaled"],
    }


def plot_main_results() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    methods = [MAIN_METHOD, "var_pca", "mlp_pca", "edm_simplex", "mve", "supervised_ridge", "lstm", "neural_ode", "latent_ode"]
    labels = {
        MAIN_METHOD: "Eco-GNRD",
        "var_pca": "VAR+PCA",
        "mlp_pca": "MLP+PCA",
        "edm_simplex": "EDM",
        "mve": "MVE",
        "supervised_ridge": "Supervised",
        "lstm": "LSTM",
        "neural_ode": "Neural ODE",
        "latent_ode": "Latent ODE",
    }
    fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
    x = np.arange(len(PLOT_DATASET_ORDER))
    width = 0.08
    offsets = np.linspace(-width * (len(methods) - 1) / 2, width * (len(methods) - 1) / 2, len(methods))
    incomplete_notes = []
    for off, method in zip(offsets, methods):
        vals = []
        for ds in PLOT_DATASET_ORDER:
            group = "main" if method == MAIN_METHOD else "baselines"
            overall = complete_dataset_overall(group, method, ds)
            if overall is None:
                incomplete_notes.append(f"{labels[method]}:{dataset_display_name(ds)}")
            vals.append(np.nan if overall is None else overall)
        ax.bar(x + off, vals, width=width, label=labels[method])
    ax.set_xticks(x, [dataset_display_name(ds) for ds in PLOT_DATASET_ORDER])
    ax.set_ylabel("Pearson")
    ax.set_title("Main Results")
    ax.legend(ncol=3, fontsize=9)
    ax.grid(axis="y", alpha=0.25)
    if incomplete_notes:
        ax.text(0.99, 0.98, "incomplete suites hidden", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="crimson")
    fig.savefig(FIG_ROOT / "main_results.png", dpi=180)
    plt.close(fig)


def plot_representative_curves(with_baselines: bool) -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = ["lv", "holling", "huisman", "beninca", "maizuru"]
    fig, axes = plt.subplots(len(datasets), 1, figsize=(14, 3.4 * len(datasets)), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        task_name = REPRESENTATIVE_TASKS[ds]
        main_curve = load_best_task_curve("main", MAIN_METHOD, ds, task_name)
        if main_curve is None:
            ax.set_axis_off()
            continue
        t = np.arange(len(main_curve["hidden_true"]))
        ax.plot(t, main_curve["hidden_true"], color="black", lw=1.5, label="True")
        ax.plot(t, main_curve["hidden_pred_scaled"], color="#d62728", lw=1.2, label="Eco-GNRD")
        if with_baselines:
            for method, color in [("var_pca", "#1f77b4"), ("mve", "#2ca02c"), ("lstm", "#9467bd"), ("supervised_ridge", "#ff7f0e")]:
                curve = load_best_task_curve("baselines", method, ds, task_name)
                if curve is not None:
                    ax.plot(t, curve["hidden_pred_scaled"], lw=1.0, alpha=0.9, color=color, label=method)
        ax.set_title(f"{dataset_display_name(ds)} / {task_name}")
        ax.grid(alpha=0.25)
        ax.legend(ncol=5 if with_baselines else 2, fontsize=8)
    out = "representative_h_with_baselines.png" if with_baselines else "representative_h_main_only.png"
    fig.savefig(FIG_ROOT / out, dpi=180)
    plt.close(fig)


def plot_ablation_bars() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    datasets = ["huisman", "beninca", "maizuru"]
    methods = [MAIN_METHOD] + ABLATION_METHODS
    labels = ["A0"] + [
        "-null", "-shuffle", "-both_cf", "-residual", "f_only", "-formula",
        "rollout0", "rollout5", "-takens", "-alt5", "-hdyn",
    ]
    fig, axes = plt.subplots(1, len(datasets), figsize=(5.5 * len(datasets), 5), constrained_layout=True)
    if len(datasets) == 1:
        axes = [axes]
    for ax, ds in zip(axes, datasets):
        vals = []
        incomplete = False
        for method in methods:
            group = "main" if method == MAIN_METHOD else "ablations"
            overall = complete_dataset_overall(group, method, ds)
            if overall is None:
                incomplete = True
            vals.append(np.nan if overall is None else overall)
        ax.bar(np.arange(len(methods)), vals, color=["#d62728"] + ["#7f7f7f"] * (len(methods) - 1))
        ax.set_xticks(np.arange(len(methods)), labels, rotation=60, ha="right")
        ax.set_title(dataset_display_name(ds) + (" (incomplete)" if incomplete else ""))
        ax.set_ylabel("Pearson")
        ax.grid(axis="y", alpha=0.25)
        if incomplete:
            ax.text(0.98, 0.96, "waiting for full ablations", transform=ax.transAxes,
                    ha="right", va="top", fontsize=9, color="crimson")
    fig.savefig(FIG_ROOT / "ablation_bar_chart.png", dpi=180)
    plt.close(fig)


def plot_coupling_scatter() -> None:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6), constrained_layout=True)
    colors = {"lv": "#1f77b4", "holling": "#ff7f0e", "huisman": "#2ca02c", "beninca": "#d62728", "maizuru": "#9467bd"}
    for ds in MAIN_DATASETS:
        dataset_root = RESULTS_ROOT / "main" / MAIN_METHOD / ds
        agg_path = dataset_root / "aggregate.json"
        if not agg_path.exists():
            continue
        agg = json.loads(agg_path.read_text(encoding="utf-8"))
        for row in agg["task_rows"]:
            if "d_ratio_mean" not in row:
                continue
            ax.scatter(row["d_ratio_mean"], row["pearson_all_mean"], s=40, color=colors[ds], label=dataset_display_name(ds))
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), fontsize=9)
    ax.set_xlabel("Coupling strength (d_ratio)")
    ax.set_ylabel("Pearson")
    ax.set_title("Coupling-Recoverability Scatter")
    ax.grid(alpha=0.25)
    fig.savefig(FIG_ROOT / "coupling_recoverability_scatter.png", dpi=180)
    plt.close(fig)


def run_plots() -> None:
    plot_main_results()
    plot_representative_curves(with_baselines=False)
    plot_representative_curves(with_baselines=True)
    plot_ablation_bars()
    plot_coupling_scatter()


def run_main_suite(skip_existing: bool, smoke: bool = False) -> None:
    for ds in MAIN_DATASETS:
        run_main_or_ablation_dataset(ds, MAIN_METHOD, SEEDS_10, skip_existing, smoke=smoke)


def run_baseline_suite(skip_existing: bool, smoke: bool = False) -> None:
    for ds in MAIN_DATASETS:
        for method in BASELINE_METHODS_ALL:
            run_baseline_dataset(ds, method, SEEDS_10, skip_existing, smoke=smoke)
    for ds in ["maizuru", "huisman", "beninca"]:
        for method in BASELINE_METHODS_DEEP:
            run_baseline_dataset(ds, method, SEEDS_5, skip_existing, smoke=smoke)


def run_ablation_suite(skip_existing: bool, smoke: bool = False) -> None:
    for ds in ["huisman", "maizuru", "beninca"]:
        for method in ABLATION_METHODS:
            run_main_or_ablation_dataset(ds, method, SEEDS_5 if smoke else SEEDS_5, skip_existing, smoke=smoke)


def write_root_readme() -> None:
    ensure_root_layout()
    lines = [
        "# 重要实验",
        "",
        "This directory stores the unified rerun suite for the paper-critical experiments.",
        "",
        "## Structure",
        "",
        "- `configs/`: experiment grid and presets",
        "- `results/main/`: latest Eco-GNRD main results",
        "- `results/baselines/`: fair baseline reruns",
        "- `results/ablations/`: ablation reruns (A0 is reused from main results)",
        "- `figures/`: paper-facing figures",
        "- `logs/`: console logs for long-running jobs",
        "",
        "Every task folder contains `aggregate.json`, `summary.md`, and `best_trajectory.npz`.",
        "Each seed folder contains `metrics.json` and `summary.md`.",
    ]
    (ROOT / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["main", "baselines", "ablations", "plots", "all"], default="all")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    ensure_root_layout()
    write_default_config()
    write_root_readme()

    if args.mode in {"main", "all"}:
        run_main_suite(skip_existing=args.skip_existing, smoke=args.smoke)
    if args.mode in {"baselines", "all"}:
        run_baseline_suite(skip_existing=args.skip_existing, smoke=args.smoke)
    if args.mode in {"ablations", "all"}:
        run_ablation_suite(skip_existing=args.skip_existing, smoke=args.smoke)
    if args.mode in {"plots", "all"} and not args.smoke:
        run_plots()


if __name__ == "__main__":
    main()

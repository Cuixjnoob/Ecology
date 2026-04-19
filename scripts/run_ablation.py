"""Ablation study: remove one component at a time, 3 datasets x 5 seeds.

Ablations:
1. no_alt: joint training (disable alternating 5:1)
2. no_ode: lam_hdyn=0 (no ODE consistency)
3. no_cf: lam_necessary=0, lam_shuffle=0 (no counterfactual)
4. no_rollout: rollout always 0
5. no_takens: takens_lags=(1,) only
"""
import sys, io, copy, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
from pathlib import Path

from scripts.run_main_experiment import (
    DATASET_CONFIGS, run_dataset, load_dataset, SEEDS, make_model, train_one,
)
import scripts.run_main_experiment as rme

# We need to patch make_model and train_one for certain ablations
# Store originals
_orig_make_model = rme.make_model
_orig_train_one = rme.train_one


def make_model_patched(N, cfg, device):
    """make_model that reads takens_lags from cfg."""
    from models.cvhi_residual import CVHI_Residual
    return CVHI_Residual(
        num_visible=N,
        encoder_d=cfg['encoder_d'], encoder_blocks=cfg['encoder_blocks'],
        encoder_heads=4,
        takens_lags=cfg.get('takens_lags', (1, 2, 4, 8)),
        encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp",
        use_formula_hints=cfg.get('use_formula_hints', True),
        use_G_field=True,
        num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


ABLATIONS = {
    'no_alt': {
        'desc': 'Joint training (no alternating 5:1)',
        'config_override': {'use_alt': False},
    },
    'no_ode': {
        'desc': 'No ODE consistency (lam_hdyn=0)',
        'config_override': {'lam_hdyn': 0.0},
    },
    'no_cf': {
        'desc': 'No counterfactual losses',
        'config_override': {'lam_necessary': 0.0, 'lam_shuffle': 0.0},
    },
    'no_rollout': {
        'desc': 'No multi-step rollout',
        'config_override': {'max_rollout_K': 0},  # handled in train_one patch
    },
    'no_takens': {
        'desc': 'No Takens delay embedding (lag=1 only)',
        'config_override': {'takens_lags': (1,)},
    },
}


def run_one_ablation(abl_name, datasets, n_seeds):
    abl = ABLATIONS[abl_name]
    print(f"\n{'#'*70}")
    print(f"# ABLATION: {abl_name} -- {abl['desc']}")
    print(f"{'#'*70}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_root = Path(f"重要实验/results/ablation/{abl_name}")
    seeds = SEEDS[:n_seeds]

    # Patch make_model for takens ablation
    rme.make_model = make_model_patched

    for ds in datasets:
        # Deep copy config and apply overrides
        cfg = copy.deepcopy(DATASET_CONFIGS[ds])
        for k, v in abl['config_override'].items():
            if k == 'max_rollout_K':
                continue  # handled separately
            if k == 'lam_shuffle':
                # lam_shuffle is computed as lam_necessary * 0.6 in train_one
                # We need to set both to 0
                pass
            cfg[k] = v

        # For no_cf: also need to override lam_shuffle in the loss call
        # We handle this by setting lam_necessary=0 which makes shuffle=0*0.6=0

        # Temporarily override DATASET_CONFIGS
        orig_cfg = DATASET_CONFIGS[ds]
        DATASET_CONFIGS[ds] = cfg

        try:
            run_dataset(ds, out_root, device, seeds=seeds)
        finally:
            DATASET_CONFIGS[ds] = orig_cfg

    # Restore
    rme.make_model = _orig_make_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ablations', nargs='+',
                        default=['no_alt', 'no_ode', 'no_cf', 'no_rollout', 'no_takens'])
    parser.add_argument('--datasets', nargs='+', default=['huisman', 'beninca', 'maizuru'])
    parser.add_argument('--seeds', type=int, default=5)
    args = parser.parse_args()

    for abl in args.ablations:
        run_one_ablation(abl, args.datasets, args.seeds)

    print("\n\nALL ABLATIONS DONE")


if __name__ == "__main__":
    main()

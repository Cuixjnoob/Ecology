"""Combined ablation: remove alt + ode + rollout together."""
import sys, io, copy
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import torch
from pathlib import Path
from scripts.run_main_experiment import DATASET_CONFIGS, run_dataset, SEEDS
import scripts.run_main_experiment as rme
from models.cvhi_residual import CVHI_Residual

# Patch make_model to read from cfg
def make_model_patched(N, cfg, device):
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

rme.make_model = make_model_patched

device = "cuda" if torch.cuda.is_available() else "cpu"
out_root = Path("重要实验/results/ablation/no_alt_ode_rollout")

for ds in ['huisman', 'beninca', 'maizuru']:
    cfg = copy.deepcopy(DATASET_CONFIGS[ds])
    cfg['use_alt'] = False
    cfg['lam_hdyn'] = 0.0
    cfg['max_rollout_K'] = 0

    orig = DATASET_CONFIGS[ds]
    DATASET_CONFIGS[ds] = cfg
    try:
        run_dataset(ds, out_root, device, seeds=SEEDS[:5])
    finally:
        DATASET_CONFIGS[ds] = orig

print("DONE")

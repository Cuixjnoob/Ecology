"""Ablation: Huisman without formula hints (use_formula_hints=False)."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from pathlib import Path
import torch
from scripts.run_main_experiment import DATASET_CONFIGS, run_dataset, SEEDS

# Override huisman config: no formula hints
DATASET_CONFIGS['huisman']['use_formula_hints'] = False

device = "cuda" if torch.cuda.is_available() else "cpu"
out_root = Path("重要实验/results/ablation/nohint")
out_root.mkdir(parents=True, exist_ok=True)

print("Starting Huisman no-hint ablation...")
run_dataset('huisman', out_root, device, seeds=SEEDS[:10])
print("DONE")

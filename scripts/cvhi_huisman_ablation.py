"""Ablation on Huisman: test each component's contribution.

Configs:
  A. baseline: joint training, no h_dyn, no alt (= Stage 1b)
  B. + alt 5:1 only
  C. + h_dyn only (original implementation, no alt)
  D. + alt 5:1 + h_dyn (NbedDyn)
  E. + alt 5:1 + h_dyn + lam_hf=0 (turn off high-freq penalty)

All use 6->1 rotation on Huisman, 3 seeds.
Resources as input-only (n_recon_channels=5 for species-only recon).
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate

SEEDS = [42, 123, 456]
EPOCHS = 300
# Test on 3 representative species: sp2 (easy), sp4 (mid), sp6 (hard)
TEST_SP = [1, 3, 5]
TEST_NAMES = ["sp2", "sp4", "sp6"]


class HDynNet(nn.Module):
    def __init__(self, n_vis, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1+n_vis, d), nn.SiLU(),
                                  nn.Linear(d, d), nn.SiLU(), nn.Linear(d, 1))
    def forward(self, h, x):
        if h.dim() == 2: h = h.unsqueeze(-1)
        return self.net(torch.cat([h, x], dim=-1)).squeeze(-1)


def load_huisman():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    full = np.concatenate([d["N_all"], d["resources"]], axis=1)
    full = (full + 0.01) / (full.mean(axis=0, keepdims=True) + 1e-3)
    return full.astype(np.float32)


def make_model(N, device):
    return CVHI_Residual(
        num_visible=N, encoder_d=64, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1,2,4,8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp", use_formula_hints=True,
        use_G_field=True, num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def get_groups(model):
    enc_ids = set()
    for n, p in model.named_parameters():
        if 'encoder' in n or 'readout' in n: enc_ids.add(id(p))
    fvis = [p for p in model.parameters() if id(p) not in enc_ids]
    enc = [p for p in model.parameters() if id(p) in enc_ids]
    return fvis, enc

def freeze(ps):
    for p in ps: p.requires_grad_(False)
def unfreeze(ps):
    for p in ps: p.requires_grad_(True)

def alpha_sched(ep, total, s=0.5, e=0.95):
    f = ep / max(1, total)
    if f <= s: return 1.0
    if f >= e: return 0.0
    return 1.0 - (f - s) / (e - s)


def train(visible, hidden, seed, device, use_alt, use_hdyn, lam_hf, n_recon_ch):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    hdyn = HDynNet(N, 32).to(device) if use_hdyn else None
    fvis, enc = get_groups(model)
    params = list(model.parameters()) + (list(hdyn.parameters()) if hdyn else [])
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=1e-4)
    wu = int(0.2*EPOCHS); ramp = max(1, int(0.2*EPOCHS))
    def lr_fn(s):
        if s < 50: return s/50
        p = (s-50)/max(1,EPOCHS-50)
        return 0.5*(1+np.cos(np.pi*p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    te = int(0.75*T); bv = float('inf'); bs = None
    PA, PB = 5, 1
    DETACH_UNTIL = 100

    for ep in range(EPOCHS):
        if hasattr(model,'G_anchor_alpha'): model.G_anchor_alpha = alpha_sched(ep,EPOCHS)
        if ep < wu: hw, Kr = 0.0, 0
        else:
            post = ep-wu; hw = min(1.0, post/ramp)
            kr = min(1.0, post/max(1,EPOCHS-wu)*2)
            Kr = max(1 if hw>0 else 0, int(round(kr*3)))
        if use_alt and ep >= wu:
            cyc = (ep-wu)%(PA+PB)
            if cyc < PA:
                freeze(enc)
                if hdyn: [p.requires_grad_(False) for p in hdyn.parameters()]
                unfreeze(fvis)
            else:
                freeze(fvis); unfreeze(enc)
                if hdyn: [p.requires_grad_(True) for p in hdyn.parameters()]
        else:
            unfreeze(fvis); unfreeze(enc)
            if hdyn: [p.requires_grad_(True) for p in hdyn.parameters()]
        if ep > wu:
            m = (torch.rand(1,T,1,device=device)>0.05).float()
            xt = x*m+(1-m)*x.mean(dim=1,keepdim=True)
        else: xt = x
        model.train()
        if hdyn: hdyn.train()
        opt.zero_grad()
        out = model(xt, n_samples=2, rollout_K=Kr)
        tr = model.slice_out(out, 0, te)
        tr['visible'] = out['visible'][:,:te]; tr['G'] = out['G'][:,:te]
        losses = model.loss(tr, beta_kl=0.03, free_bits=0.02,
            margin_null=0.003, margin_shuf=0.002,
            lam_necessary=5.0, lam_shuffle=3.0,
            lam_energy=2.0, min_energy=0.02,
            lam_smooth=0.02, lam_sparse=0.02,
            h_weight=hw, lam_rollout=0.5*hw,
            rollout_weights=(1.0,0.5,0.25),
            lam_hf=lam_hf, lowpass_sigma=6.0,
            lam_rmse_log=0.1, n_recon_channels=n_recon_ch)
        total = losses['total']
        if use_hdyn and hw > 0:
            hm = tr['h_samples'].mean(dim=0)
            Th = hm.shape[-1]
            hp = hdyn(hm[:,:-1], xt[:,:Th-1])
            tgt = hm[:,1:].detach() if ep < DETACH_UNTIL+wu else hm[:,1:]
            total = total + 0.3 * hw * F.mse_loss(hp, tgt)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out, te, T)
            vo['visible'] = out['visible'][:,te:T]; vo['G'] = out['G'][:,te:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02, lam_rollout=0.5,
                rollout_weights=(1.0,0.5,0.25), lam_hf=lam_hf, lowpass_sigma=6.0,
                n_recon_channels=n_recon_ch)
            vr = vl['recon_full'].item()
        if ep > wu+15 and vr < bv:
            bv = vr; bs = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        oe = model(x, n_samples=30, rollout_K=3)
        hm = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    p, hs = evaluate(hm, hidden)
    del model, hdyn
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return p, hs.flatten()


CONFIGS = [
    # name,      use_alt, use_hdyn, lam_hf
    ("baseline",  False,   False,    0.0),
    ("alt_only",  True,    False,    0.0),
    ("hdyn_only", False,   True,     0.0),
    ("alt+hdyn",  True,    True,     0.0),
    ("alt+hdyn+hf", True,  True,     0.2),
]


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_ablation")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load_huisman()
    n_sp = 6  # 6 species, 5 resources
    # n_recon_ch: when hiding species i, visible has 5 species + 5 resources = 10 ch
    # recon on first 5 (species only)
    n_recon_ch = 5

    print(f"Huisman Ablation: {len(CONFIGS)} configs x {len(TEST_SP)} species x {len(SEEDS)} seeds")
    print(f"Resources as input-only (n_recon_channels={n_recon_ch})\n")

    results = {c[0]: {n: [] for n in TEST_NAMES} for c in CONFIGS}
    total = len(CONFIGS) * len(TEST_SP) * len(SEEDS)
    ri = 0

    for sp_idx, sp_name in zip(TEST_SP, TEST_NAMES):
        visible = np.delete(full, sp_idx, axis=1)
        hidden = full[:, sp_idx]
        print(f"--- hidden={sp_name} ---")
        for cfg_name, use_alt, use_hdyn, lam_hf in CONFIGS:
            for seed in SEEDS:
                ri += 1
                t0 = datetime.now()
                p, _ = train(visible, hidden, seed, device, use_alt, use_hdyn, lam_hf, n_recon_ch)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  [{ri}/{total}] {cfg_name:<15} seed={seed}  P={p:+.3f}  ({dt:.1f}s)")
                results[cfg_name][sp_name].append(p)

    print(f"\n{'='*90}")
    print("HUISMAN ABLATION RESULTS")
    print('='*90)
    hdr = f"{'Config':<18}" + "".join(f"{n:>10}" for n in TEST_NAMES) + f"{'Overall':>10}"
    print(hdr); print('-'*90)
    for cfg_name, _, _, _ in CONFIGS:
        row = f"{cfg_name:<18}"
        all_p = []
        for n in TEST_NAMES:
            m = np.mean(results[cfg_name][n])
            row += f"{m:>+10.3f}"
            all_p.extend(results[cfg_name][n])
        row += f"{np.mean(all_p):>+10.3f}"
        print(row)
    print('-'*90)
    print(f"Ref: previous hdyn_only = 0.444, previous baseline = 0.380")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Huisman Ablation\n\n")
        f.write(f"| Config |" + "|".join(f" {n} " for n in TEST_NAMES) + "| Overall |\n")
        f.write("|---|" + "---|" * len(TEST_NAMES) + "---|\n")
        for cfg_name, _, _, _ in CONFIGS:
            row = f"| {cfg_name} "
            all_p = []
            for n in TEST_NAMES:
                m = np.mean(results[cfg_name][n])
                row += f"| {m:+.3f} "
                all_p.extend(results[cfg_name][n])
            row += f"| {np.mean(all_p):+.3f} |"
            f.write(row + "\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

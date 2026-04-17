"""Tuned Huisman: hdyn_only (best from ablation) + 500 epochs + sweep lam_hdyn.

Target: push overall Pearson toward 0.6.
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
EPOCHS = 500
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


def alpha_sched(ep, total, s=0.5, e=0.95):
    f = ep/max(1,total)
    if f <= s: return 1.0
    if f >= e: return 0.0
    return 1.0-(f-s)/(e-s)


def train(visible, hidden, seed, device, lam_hdyn, n_recon_ch, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    hdyn = HDynNet(N, 32).to(device)
    params = list(model.parameters()) + list(hdyn.parameters())
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=1e-4)
    wu = int(0.2*epochs); ramp = max(1, int(0.2*epochs))
    DETACH_UNTIL = 100
    def lr_fn(s):
        if s < 50: return s/50
        p = (s-50)/max(1,epochs-50)
        return 0.5*(1+np.cos(np.pi*p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    te = int(0.75*T); bv = float('inf'); bs = None

    for ep in range(epochs):
        if hasattr(model,'G_anchor_alpha'): model.G_anchor_alpha = alpha_sched(ep,epochs)
        if ep < wu: hw, Kr = 0.0, 0
        else:
            post = ep-wu; hw = min(1.0, post/ramp)
            kr = min(1.0, post/max(1,epochs-wu)*2)
            Kr = max(1 if hw>0 else 0, int(round(kr*3)))
        if ep > wu:
            m = (torch.rand(1,T,1,device=device)>0.05).float()
            xt = x*m+(1-m)*x.mean(dim=1,keepdim=True)
        else: xt = x
        model.train(); hdyn.train(); opt.zero_grad()
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
            lam_hf=0.0, lowpass_sigma=6.0,
            lam_rmse_log=0.1, n_recon_channels=n_recon_ch)
        total = losses['total']
        if hw > 0:
            hm = tr['h_samples'].mean(dim=0)
            Th = hm.shape[-1]
            hp = hdyn(hm[:,:-1], xt[:,:Th-1])
            tgt = hm[:,1:].detach() if ep < DETACH_UNTIL+wu else hm[:,1:]
            total = total + lam_hdyn * hw * F.mse_loss(hp, tgt)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out, te, T)
            vo['visible'] = out['visible'][:,te:T]; vo['G'] = out['G'][:,te:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02, lam_rollout=0.5,
                rollout_weights=(1.0,0.5,0.25), lam_hf=0.0, lowpass_sigma=6.0,
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
    return p


LAMBDAS = [0.1, 0.3, 0.5, 1.0]


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_tuned")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load_huisman()
    n_recon_ch = 5

    print(f"Huisman tuned: hdyn_only, 500ep, sweep lam_hdyn={LAMBDAS}")
    print(f"3 species x 3 seeds x {len(LAMBDAS)} lambdas = {3*3*len(LAMBDAS)} runs\n")

    results = {lam: {n: [] for n in TEST_NAMES} for lam in LAMBDAS}
    total = len(LAMBDAS) * len(TEST_SP) * len(SEEDS)
    ri = 0

    for sp_idx, sp_name in zip(TEST_SP, TEST_NAMES):
        visible = np.delete(full, sp_idx, axis=1)
        hidden = full[:, sp_idx]
        print(f"--- hidden={sp_name} ---")
        for lam in LAMBDAS:
            for seed in SEEDS:
                ri += 1
                t0 = datetime.now()
                p = train(visible, hidden, seed, device, lam, n_recon_ch)
                dt = (datetime.now() - t0).total_seconds()
                print(f"  [{ri}/{total}] lam={lam}  seed={seed}  P={p:+.3f}  ({dt:.1f}s)")
                results[lam][sp_name].append(p)

    print(f"\n{'='*90}")
    print("HUISMAN TUNED RESULTS (500ep, hdyn_only)")
    print('='*90)
    hdr = f"{'lam_hdyn':<12}" + "".join(f"{n:>10}" for n in TEST_NAMES) + f"{'Overall':>10}"
    print(hdr); print('-'*90)
    for lam in LAMBDAS:
        row = f"{lam:<12}"
        all_p = []
        for n in TEST_NAMES:
            m = np.mean(results[lam][n])
            row += f"{m:>+10.3f}"
            all_p.extend(results[lam][n])
        row += f"{np.mean(all_p):>+10.3f}"
        print(row)
    print('-'*90)
    print(f"Ref: ablation hdyn_only(300ep) = 0.525")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Huisman Tuned: hdyn_only 500ep\n\n")
        f.write(f"| lam_hdyn |" + "|".join(f" {n} " for n in TEST_NAMES) + "| Overall |\n")
        f.write("|---|" + "---|" * len(TEST_NAMES) + "---|\n")
        for lam in LAMBDAS:
            row = f"| {lam} "
            all_p = []
            for n in TEST_NAMES:
                m = np.mean(results[lam][n])
                row += f"| {m:+.3f} "
                all_p.extend(results[lam][n])
            row += f"| {np.mean(all_p):+.3f} |"
            f.write(row + "\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

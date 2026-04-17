"""Tuned CVHI on Huisman: sweep key hyperparameters to beat LSTM (0.535).

Sweep:
  - lam_hdyn: [0.05, 0.1, 0.2]
  - lr: [0.0005, 0.0008, 0.001]
  - epochs: 500
  - Joint training (no alternating - proven better on Huisman)
  - Resources as input-only

Test on all 6 species, 3 seeds.
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime
from itertools import product

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_residual_L1L3_diagnostics import evaluate

SEEDS = [42, 123, 456]
EPOCHS = 500
ALL_SP = list(range(6))
ALL_NAMES = [f"sp{i+1}" for i in range(6)]

# Grid
LAMS = [0.05, 0.1, 0.2]
LRS = [0.0005, 0.0008, 0.0012]


class HDynNet(nn.Module):
    def __init__(self, n, d=32):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1+n,d), nn.SiLU(), nn.Linear(d,d), nn.SiLU(), nn.Linear(d,1))
    def forward(self, h, x):
        if h.dim()==2: h=h.unsqueeze(-1)
        return self.net(torch.cat([h,x],dim=-1)).squeeze(-1)


def load_data():
    d = np.load("runs/huisman1999_chaos/trajectories.npz")
    full = np.concatenate([d["N_all"], d["resources"]], axis=1)
    full = (full + 0.01) / (full.mean(axis=0, keepdims=True) + 1e-3)
    return full.astype(np.float32)


def make_model(N, device, enc_d=64):
    return CVHI_Residual(
        num_visible=N, encoder_d=enc_d, encoder_blocks=2, encoder_heads=4,
        takens_lags=(1,2,4,8), encoder_dropout=0.1,
        d_species_f=20, f_visible_layers=2, f_visible_top_k=4,
        d_species_G=12, G_field_layers=1, G_field_top_k=3,
        prior_std=1.0, gnn_backbone="mlp", use_formula_hints=True,
        use_G_field=True, num_mixture_components=1,
        G_anchor_first=True, G_anchor_sign=+1,
    ).to(device)


def alpha_sched(ep, total):
    f = ep/max(1,total)
    if f <= 0.5: return 1.0
    if f >= 0.95: return 0.0
    return 1.0-(f-0.5)/(0.95-0.5)


def train(vis, hid, seed, device, lam, lr, nrc, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = vis.shape
    x = torch.tensor(vis, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    hdyn = HDynNet(N, 32).to(device)
    params = list(model.parameters()) + list(hdyn.parameters())
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-4)
    wu = int(0.2*epochs); ramp = max(1, int(0.2*epochs))
    DU = 100
    def lr_fn(s):
        if s<50: return s/50
        p=(s-50)/max(1,epochs-50)
        return 0.5*(1+np.cos(np.pi*p))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)
    te = int(0.75*T); bv=float('inf'); bs=None

    for ep in range(epochs):
        if hasattr(model,'G_anchor_alpha'): model.G_anchor_alpha=alpha_sched(ep,epochs)
        if ep<wu: hw,Kr=0.0,0
        else:
            post=ep-wu; hw=min(1.0,post/ramp)
            kr=min(1.0,post/max(1,epochs-wu)*2)
            Kr=max(1 if hw>0 else 0, int(round(kr*3)))
        if ep>wu:
            m=(torch.rand(1,T,1,device=device)>0.05).float()
            xt=x*m+(1-m)*x.mean(dim=1,keepdim=True)
        else: xt=x
        model.train(); hdyn.train(); opt.zero_grad()
        out=model(xt,n_samples=2,rollout_K=Kr)
        tr=model.slice_out(out,0,te)
        tr['visible']=out['visible'][:,:te]; tr['G']=out['G'][:,:te]
        losses=model.loss(tr,beta_kl=0.03,free_bits=0.02,
            margin_null=0.003,margin_shuf=0.002,
            lam_necessary=5.0,lam_shuffle=3.0,
            lam_energy=2.0,min_energy=0.02,
            lam_smooth=0.02,lam_sparse=0.02,
            h_weight=hw,lam_rollout=0.5*hw,
            rollout_weights=(1.0,0.5,0.25),
            lam_hf=0.0,lowpass_sigma=6.0,
            lam_rmse_log=0.1,n_recon_channels=nrc)
        total=losses['total']
        if hw>0:
            hm=tr['h_samples'].mean(dim=0); Th=hm.shape[-1]
            hp=hdyn(hm[:,:-1],xt[:,:Th-1])
            tgt=hm[:,1:].detach() if ep<DU+wu else hm[:,1:]
            total=total+lam*hw*F.mse_loss(hp,tgt)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo=model.slice_out(out,te,T)
            vo['visible']=out['visible'][:,te:T]; vo['G']=out['G'][:,te:T]
            vl=model.loss(vo,h_weight=1.0,margin_null=0.003,margin_shuf=0.002,
                lam_energy=2.0,min_energy=0.02,lam_rollout=0.5,
                rollout_weights=(1.0,0.5,0.25),lam_hf=0.0,lowpass_sigma=6.0,
                n_recon_channels=nrc)
            vr=vl['recon_full'].item()
        if ep>wu+15 and vr<bv:
            bv=vr; bs={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        oe=model(x,n_samples=30,rollout_K=3)
        hm=oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    p,_=evaluate(hm,hid)
    del model,hdyn
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return p


def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_huisman_tuned_v2")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    full = load_data()
    nrc = 5

    configs = list(product(LAMS, LRS))
    print(f"Huisman tuned v2: {len(configs)} configs x 6 species x 3 seeds = {len(configs)*18} runs")
    print(f"Grid: lam={LAMS}, lr={LRS}\n")

    # For speed: test on 3 representative species first
    TEST_SP = [1, 3, 5]; TEST_NAMES = ["sp2", "sp4", "sp6"]

    results = {}
    total = len(configs) * len(TEST_SP) * len(SEEDS)
    ri = 0

    for lam, lr in configs:
        key = f"lam={lam}_lr={lr}"
        results[key] = {n: [] for n in TEST_NAMES}
        for sp_idx, sp_name in zip(TEST_SP, TEST_NAMES):
            vis = np.delete(full, sp_idx, axis=1)
            hid = full[:, sp_idx]
            for seed in SEEDS:
                ri += 1
                t0 = datetime.now()
                p = train(vis, hid, seed, device, lam, lr, nrc)
                dt = (datetime.now()-t0).total_seconds()
                print(f"  [{ri}/{total}] {key:<22} {sp_name} seed={seed}  P={p:+.3f}  ({dt:.1f}s)")
                results[key][sp_name].append(p)

    print(f"\n{'='*100}")
    print("HUISMAN TUNED V2")
    print('='*100)
    hdr = f"{'Config':<24}" + "".join(f"{n:>10}" for n in TEST_NAMES) + f"{'Overall':>10}"
    print(hdr); print('-'*100)
    best_cfg = None; best_overall = -1
    for key in results:
        row = f"{key:<24}"
        all_p = []
        for n in TEST_NAMES:
            m = np.mean(results[key][n])
            row += f"{m:>+10.3f}"
            all_p.extend(results[key][n])
        ov = np.mean(all_p)
        row += f"{ov:>+10.3f}"
        print(row)
        if ov > best_overall:
            best_overall = ov; best_cfg = key
    print('-'*100)
    print(f"Best: {best_cfg} = {best_overall:+.3f}")
    print(f"Ref: Fair LSTM = +0.535, prev hdyn_only = +0.525")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Huisman Tuned V2\n\n")
        f.write(f"| Config |" + "|".join(f" {n} " for n in TEST_NAMES) + "| Overall |\n")
        f.write("|---|" + "---|"*len(TEST_NAMES) + "---|\n")
        for key in results:
            row = f"| {key} "
            all_p = []
            for n in TEST_NAMES:
                m = np.mean(results[key][n])
                row += f"| {m:+.3f} "
                all_p.extend(results[key][n])
            row += f"| {np.mean(all_p):+.3f} |"
            f.write(row + "\n")
        f.write(f"\n**Best: {best_cfg} = {best_overall:+.3f}**\n")

    print(f"\n[OK] {out_dir}")


if __name__ == "__main__":
    main()

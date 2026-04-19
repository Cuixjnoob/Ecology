"""Maizuru without temperature input — test if recovery is from species interactions or Moran effect."""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import numpy as np
import torch
import torch.nn.functional as F
from pathlib import Path
from datetime import datetime

from models.cvhi_residual import CVHI_Residual
from scripts.cvhi_maizuru import make_model, alpha_sched, HDynNet, LAM_HDYN, EPOCHS
from scripts.load_maizuru import load_maizuru

SEEDS = [42, 123, 456]

def train_one(visible, hidden, seed, device, n_recon_ch, epochs=EPOCHS):
    torch.manual_seed(seed)
    T, N = visible.shape
    x = torch.tensor(visible, dtype=torch.float32, device=device).unsqueeze(0)
    model = make_model(N, device)
    hdyn = HDynNet(N, 32).to(device)
    params = list(model.parameters()) + list(hdyn.parameters())
    opt = torch.optim.AdamW(params, lr=0.0008, weight_decay=1e-4)
    wu = int(0.2*epochs); ramp = max(1, int(0.2*epochs))
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
            post = ep-wu; hw = min(1.0,post/ramp)
            kr = min(1.0,post/max(1,epochs-wu)*2)
            Kr = max(1 if hw>0 else 0, int(round(kr*3)))
        if ep > wu:
            m = (torch.rand(1,T,1,device=device)>0.05).float()
            xt = x*m+(1-m)*x.mean(dim=1,keepdim=True)
        else: xt = x
        model.train(); hdyn.train(); opt.zero_grad()
        out = model(xt, n_samples=2, rollout_K=Kr)
        tr = model.slice_out(out,0,te); tr['visible']=out['visible'][:,:te]; tr['G']=out['G'][:,:te]
        losses = model.loss(tr, beta_kl=0.03, free_bits=0.02, margin_null=0.003, margin_shuf=0.002,
            lam_necessary=5.0, lam_shuffle=3.0, lam_energy=2.0, min_energy=0.02,
            lam_smooth=0.02, lam_sparse=0.02, h_weight=hw, lam_rollout=0.5*hw,
            rollout_weights=(1.0,0.5,0.25), lam_hf=0.0, lowpass_sigma=6.0,
            lam_rmse_log=0.1, n_recon_channels=n_recon_ch)
        total = losses['total']
        if hw > 0:
            hm = tr['h_samples'].mean(dim=0); Th = hm.shape[-1]
            hp = hdyn(hm[:,:-1],xt[:,:Th-1])
            tgt = hm[:,1:].detach() if ep < 100+wu else hm[:,1:]
            total = total + LAM_HDYN*hw*F.mse_loss(hp,tgt)
        total.backward()
        torch.nn.utils.clip_grad_norm_(params,1.0); opt.step(); sched.step()
        with torch.no_grad():
            vo = model.slice_out(out,te,T); vo['visible']=out['visible'][:,te:T]; vo['G']=out['G'][:,te:T]
            vl = model.loss(vo, h_weight=1.0, margin_null=0.003, margin_shuf=0.002,
                lam_energy=2.0, min_energy=0.02, lam_rollout=0.5,
                rollout_weights=(1.0,0.5,0.25), lam_hf=0.0, lowpass_sigma=6.0, n_recon_channels=n_recon_ch)
            vr = vl['recon_full'].item()
        if ep > wu+15 and vr < bv:
            bv = vr; bs = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if bs: model.load_state_dict(bs)
    model.eval()
    with torch.no_grad():
        oe = model(x, n_samples=30, rollout_K=3)
        hm = oe['h_samples'].mean(dim=0)[0].cpu().numpy()
    L = min(len(hm), len(hidden))
    X_tr = np.column_stack([hm[:te], np.ones(te)])
    coef, _, _, _ = np.linalg.lstsq(X_tr, hidden[:te], rcond=None)
    X_all = np.column_stack([hm[:L], np.ones(L)])
    h_sc = X_all @ coef
    r_val = float(np.corrcoef(h_sc[te:L], hidden[te:L])[0,1]) if L > te+2 else 0
    del model, hdyn
    if torch.cuda.is_available(): torch.cuda.empty_cache()
    return r_val

def main():
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(f"runs/{ts}_maizuru_notemp")
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load WITHOUT temperature
    full, species, days = load_maizuru(include_temp=False)
    species = [str(s) for s in species]
    n_species = len(species)

    print(f"Maizuru NO TEMP: {n_species} species, no temperature input")
    print(f"Testing: is recovery from species interactions or Moran effect?\n")

    results = {sp: [] for sp in species}
    total = len(species) * len(SEEDS)
    ri = 0
    for h_name in species:
        h_idx = species.index(h_name)
        visible = np.delete(full, h_idx, axis=1).astype(np.float32)
        hidden = full[:, h_idx].astype(np.float32)
        n_rc = n_species - 1
        print(f"--- hidden={h_name} ---")
        for seed in SEEDS:
            ri += 1
            t0 = datetime.now()
            rv = train_one(visible, hidden, seed, device, n_rc)
            dt = (datetime.now() - t0).total_seconds()
            print(f"  [{ri}/{total}] seed={seed}  val={rv:+.3f}  ({dt:.1f}s)")
            results[h_name].append(rv)

    print(f"\n{'='*60}")
    print(f"{'Species':<35} {'with_temp':>10} {'no_temp':>10}")
    print('-'*60)
    # Reference: with-temp results
    ref = {'Aurelia.sp':0.045,'Engraulis.japonicus':0.071,'Plotosus.japonicus':0.273,
           'Sebastes.inermis':0.236,'Trachurus.japonicus':0.512,'Girella.punctata':0.271,
           'Pseudolabrus.sieboldi':0.529,'Parajulis.poecilepterus':0.407,
           'Halichoeres.tenuispinis':0.308,'Chaenogobius.gulosus':0.174,
           'Pterogobius.zonoleucus':0.328,'Tridentiger.trigonocephalus':-0.034,
           'Siganus.fuscescens':0.214,'Sphyraena.pinguis':0.099,'Rudarius.ercodes':0.181}
    all_wt = []; all_nt = []
    for sp in species:
        wt = ref.get(sp, 0)
        nt = np.mean(results[sp])
        all_wt.append(wt); all_nt.append(nt)
        print(f"{sp:<35} {wt:>+10.3f} {nt:>+10.3f}")
    print('-'*60)
    print(f"{'Overall':<35} {np.mean(all_wt):>+10.3f} {np.mean(all_nt):>+10.3f}")

    with open(out_dir / "summary.md", "w", encoding="utf-8") as f:
        f.write("# Maizuru: with temp vs no temp\n\n")
        f.write("| Species | with_temp | no_temp |\n|---|---|---|\n")
        for sp in species:
            wt = ref.get(sp, 0)
            nt = np.mean(results[sp])
            f.write(f"| {sp} | {wt:+.3f} | {nt:+.3f} |\n")
        f.write(f"\n**Overall**: with_temp={np.mean(all_wt):+.3f}, no_temp={np.mean(all_nt):+.3f}\n")

    print(f"\n[OK] {out_dir}")

if __name__ == "__main__":
    main()

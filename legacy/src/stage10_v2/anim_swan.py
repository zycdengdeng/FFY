# -*- coding: utf-8 -*-
"""落震对照：号手天鹅（真鸟实测几何）vs 生成器在同一体重/工况下的设计。
两侧共用同一组关节刚度 κ/τ —— 真鸟没有刚度实测数据，这是唯一诚实的做法。
唯一的变量是几何（L1、段比）和触地姿态。"""
import os, sys, json
import numpy as np, torch
sys.path.insert(0, os.path.abspath("src/stage10_v2")); sys.path.insert(0, os.path.abspath("src/stage7_generative"))
import physics_v2 as P
from bioprior import BioPrior
from factory_v2 import zeta_of_kc
from e17_emergent_b import load as load_cvae
from train_cvae import norm
from e18b_corridor_multi import CONDS, GCAP_G, SMAX

L1_MM, M_KG, R2, R3, THA_B, THK_B = 108.5, 11.07113, 1.80, 1.06, 144.0, 133.0
COND = "concrete1.2"; OUT = "outputs/anim_swan"
BASE = {**P.SCEN_BIRD_X, "hip_damp_unified": True, "foot_mode": "bearing"}
cd = CONDS[COND]; os.makedirs(OUT, exist_ok=True)

model, meta_m = load_cvae("outputs/v23_e5_bio/cvae_r12.pt")
pr = meta_m["prior"]
prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
c_lo, c_hi = np.array(meta_m["c_lo"]), np.array(meta_m["c_hi"])
c = np.array([np.log10(M_KG), cd["v0"], np.log10(cd["kc"]), GCAP_G * 9.81, SMAX])
torch.manual_seed(11)
with torch.no_grad():
    u = model.sample(torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32), 32).numpy()
XG = list(prior.expand(np.clip(u, 0, 1), M_KG).mean(0))
KAP = list(XG[3:7])
XB = [L1_MM, R2, R3] + KAP + [THA_B, THK_B]

CASES = [("bio",  "real swan (measured)", XB, "#8E2A34"),
         ("phys", "generator",            XG, "#1b6ca8")]
out, meta = {}, {}
for k, lab, x, col in CASES:
    r = P.eval_v2(tuple(x), M_KG, cd["v0"], kc=cd["kc"], zeta_c=zeta_of_kc(cd["kc"]),
                  npass=2, base=BASE, keep_history=True)
    assert not r.get("fail"), (k, r)
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    out[k] = r.pop("hist")
    meta[k] = dict(label=lab, color=col, L1_mm=float(x[0]), thA=float(x[7]), thK=float(x[8]),
                   peak_g=r["peak_a"] / 9.81, leg_stroke_mm=r["leg_stroke_mm"],
                   leg_mass_g=r["leg_mass_kg"] * 1e3, ok=bool(ok), why=list(why),
                   x=[float(v) for v in x])
    print(f"  {lab:<22} L1 {x[0]:6.1f} mm  thA {x[7]:5.1f}  peak {meta[k]['peak_g']:5.2f} g"
          f"  stroke {meta[k]['leg_stroke_mm']:5.1f} mm  mass {meta[k]['leg_mass_g']:5.0f} g"
          f"  {'ok' if ok else 'x ' + ','.join(why)}")
np.savez_compressed(f"{OUT}/hist_swan.npz", meta=json.dumps(meta),
    **{f"{k}__{n}": np.asarray(v) for k, h in out.items() for n, v in h.items()})
json.dump(dict(m_kg=M_KG, v0=cd["v0"], kc=cd["kc"], species="Cygnus buccinator",
               kappa=KAP, gcap_g=GCAP_G, smax_mm=1e3 * SMAX, meta=meta),
          open(f"{OUT}/swan_compare.json", "w"), ensure_ascii=False, indent=1)
print("[saved]", OUT)

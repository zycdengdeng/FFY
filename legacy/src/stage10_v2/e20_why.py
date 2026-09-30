# -*- coding: utf-8 -*-
"""E20 补充诊断:模型输出的那个设计,在每个体重上究竟被哪一条判据卡住。

解码器已近乎确定性(变异系数 ~0.2%),所以每格只需 1 个样本即可代表,
成本是主实验的 1/216。补出主实验没存的:失败原因、三段管径、细长比余量。
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np, torch

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P
from bioprior import BioPrior
from factory_v2 import zeta_of_kc
from e17_emergent_b import load as load_cvae
from train_cvae import norm
from e18b_corridor_multi import CONDS, GCAP_G, SMAX
from e20_gen_corridor import M_GRID, ANCHORS

WHY = {"gcap": "过载超限", "smax": "行程超限", "slenderness": "细长比超限",
       "massbudget": "质量超预算", "deep_sink": "模型失效(侵入超界)",
       "collapse": "腿压塌", "solver": "求解器失败", "nonfinite": "数值发散"}


def _one(a):
    x7, m, v0, kc = a
    r = P.eval_v2(tuple(x7), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2)
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    if r is None or r.get("fail"):
        return dict(ok=False, why=list(why))
    return dict(ok=bool(ok), why=list(why), peak_g=r["peak_a"] / 9.81,
                eta=r.get("eta", np.nan), stroke=r["leg_stroke_mm"],
                sink=r["sink_mm"], D=[round(v, 1) for v in r["D_mm"]],
                Dmax=[round(v, 1) for v in r["D_max_mm"]],
                legmass=r["leg_mass_kg"] * 1e3, frac=r["mass_frac"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v2_e5_bio/cvae_r40.pt")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_e20")
    a = ap.parse_args()
    model, meta = load_cvae(a.ckpt); pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"])
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    ms = list(M_GRID) + sorted(ANCHORS)
    jobs, tags = [], []
    for cn, cd in CONDS.items():
        for m in ms:
            c = np.array([np.log10(m), cd["v0"], np.log10(cd["kc"]),
                          GCAP_G * 9.81, SMAX])
            torch.manual_seed(1234)
            with torch.no_grad():
                u = model.sample(torch.tensor(norm(c, c_lo, c_hi),
                                              dtype=torch.float32), 8).numpy()
            x = prior.expand(np.clip(u, 0, 1), float(m)).mean(0)
            jobs.append((tuple(x), float(m), cd["v0"], cd["kc"]))
            tags.append((cn, float(m), [float(v) for v in x]))
    print(f"[why] {len(jobs)} 次评价(每格 1 个代表设计)", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_one, jobs, chunksize=1))
    out = [dict(cond=cn, label=CONDS[cn]["label"], m_kg=m, x7=x,
                is_anchor=(m in ANCHORS), **r) for (cn, m, x), r in zip(tags, res)]
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, "e20_why.json")
    json.dump(out, open(fp, "w"), indent=1, ensure_ascii=False)
    for cn, cd in CONDS.items():
        print(f"\n=== {cd['label']} ===")
        for o in [q for q in out if q["cond"] == cn]:
            v = "✓可行" if o["ok"] else "×" + "、".join(WHY.get(w, w) for w in o["why"])
            ex_ = "" if not o.get("D") else (
                f"  管径 {o['D'][2]:.1f}/上限 {o['Dmax'][2]:.1f}"
                f"  行程 {o['stroke']:.1f}  峰值 {o['peak_g']:.2f}g")
            print(f"  {o['m_kg']:>6.1f}kg{'*' if o['is_anchor'] else ' '} {v:<28}{ex_}")
    print(f"\n[why] → {fp}")

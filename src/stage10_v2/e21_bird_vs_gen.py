# -*- coding: utf-8 -*-
"""E21 · 在水鸟体重区间内:真鸟几何 vs 生成几何,四项力学指标对照。

**只比几何。** 真实鸟类没有关节刚度的实测数据 —— 这是绕不开的限制。
所以两侧用**完全相同**的 κ踝/κ膝/κ髋/τ(取 cVAE 在该体重给出的那一组),
唯一变量是三段骨长 (L1, r2, r3)。任何"AI 比水鸟好"的说法都不成立,
本表回答的是:**着陆物理更偏好哪种几何**。

真鸟几何来源:
  L1 = AVONET 跗跖长(标本级中位)      r2/r3 = Watanabe 2017 该科中位
生成几何来源:
  cVAE(bio 臂 r40)在同一体重、同一工况下输出的设计

用法:  python src/stage10_v2/e21_bird_vs_gen.py --workers 32 --out outputs/v2_e21
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P                              # noqa: E402
from bioprior import BioPrior                       # noqa: E402
from factory_v2 import zeta_of_kc                   # noqa: E402
from e17_emergent_b import load as load_cvae        # noqa: E402
from train_cvae import norm                         # noqa: E402
from e18b_corridor_multi import CONDS, GCAP_G, SMAX  # noqa: E402

# 各科的三段骨比例中位数。雁鸭科来自 Watanabe 2017 实测;
# 其余科无同源骨骼表,统一用雁鸭中位作近似 —— 这是本表的第二个限制,须披露。
FAM_RATIO = {"Anatidae": (1.80, 1.06), "_default": (1.80, 1.06)}
# 挑选的物种:覆盖水鸟体重全程,且每科都有代表(n_spec ≥ 4 才纳入)
PICK = ["Tachybaptus ruficollis", "Aythya fuligula", "Anas platyrhynchos",
        "Phalacrocorax carbo", "Gavia immer", "Pelecanus onocrotalus",
        "Cygnus olor"]
MET = ["peak_g", "eta", "cfe", "leg_stroke_mm", "sink_mm", "leg_mass_g", "mass_frac"]


def _probe(a):
    x7, m, v0, kc = a
    r = P.eval_v2(tuple(x7), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2)
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    if r is None or r.get("fail"):
        return dict(ok=False, why=[(r or {}).get("fail", "?")],
                    **{k: float("nan") for k in MET})
    return dict(ok=bool(ok), why=list(why),
                peak_g=r["peak_a"] / 9.81, eta=r.get("eta", np.nan),
                cfe=r.get("cfe", np.nan), leg_stroke_mm=r["leg_stroke_mm"],
                sink_mm=r["sink_mm"], leg_mass_g=r["leg_mass_kg"] * 1e3,
                mass_frac=r["mass_frac"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avonet", default="data/bio/avonet_waterbirds.csv")
    ap.add_argument("--ckpt", default="outputs/v2_e5_bio/cvae_r40.pt")
    ap.add_argument("--conds", default="concrete1.2,turf1.2,wetsand1.2")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_e21")
    a = ap.parse_args()

    rows = [l.strip().split(",") for l in open(a.avonet, encoding="utf-8")][1:]
    db = {r[0]: (r[1], float(r[2]), float(r[3])) for r in rows if len(r) >= 4}
    birds = [(s, *db[s]) for s in PICK if s in db]
    assert birds, "AVONET 里找不到 PICK 中的物种,检查 --avonet 路径"

    model, meta = load_cvae(a.ckpt)
    pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"])
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])

    jobs, tags = [], []
    for cname in a.conds.split(","):
        cd = CONDS[cname]
        for sp, fam, tarsus, mass_g in birds:
            m = mass_g / 1000.0
            c = np.array([np.log10(m), cd["v0"], np.log10(cd["kc"]),
                          GCAP_G * 9.81, SMAX])
            torch.manual_seed(11)
            with torch.no_grad():
                u = model.sample(torch.tensor(norm(c, c_lo, c_hi),
                                              dtype=torch.float32), 32).numpy()
            xg = prior.expand(np.clip(u, 0, 1), m).mean(0)   # 解码器已近乎确定性
            kap = list(xg[3:7])                              # 两侧共用这一组刚度
            r2, r3 = FAM_RATIO.get(fam, FAM_RATIO["_default"])
            xb = [tarsus, r2, r3] + kap                      # 真鸟几何
            for who, x in (("bird", xb), ("gen", list(xg))):
                jobs.append((tuple(x), m, cd["v0"], cd["kc"]))
                tags.append((cname, sp, fam, m, who, [float(v) for v in x]))

    print(f"[e21] {len(birds)} 种 × {len(a.conds.split(','))} 工况 × 2 = {len(jobs)} 次评价")
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_probe, jobs, chunksize=1))

    out = []
    for (cname, sp, fam, m, who, x), r in zip(tags, res):
        out.append(dict(cond=cname, label=CONDS[cname]["label"], species=sp,
                        family=fam, m_kg=m, who=who, x7=x,
                        L_mm=[x[0], x[0] * x[1], x[0] * x[2]], **r))
    os.makedirs(a.out, exist_ok=True)
    fp = os.path.join(a.out, "e21_bird_vs_gen.json")
    json.dump(out, open(fp, "w"), indent=1, ensure_ascii=False)

    for cname in a.conds.split(","):
        print(f"\n=== {CONDS[cname]['label']} ===")
        print(f"{'物种':<24}{'体重kg':>7}{'来源':>6}{'L1mm':>7}{'峰值g':>7}"
              f"{'η':>6}{'行程mm':>8}{'腿重g':>7}{'判定':>16}")
        for sp, fam, tar, mg in birds:
            for who, nm in (("bird", "真鸟"), ("gen", "生成")):
                d = next(o for o in out if o["cond"] == cname and
                         o["species"] == sp and o["who"] == who)
                v = "可行" if d["ok"] else "×" + ",".join(d["why"])
                print(f"{sp if who=='bird' else '':<24}{d['m_kg']:>7.2f}{nm:>5}"
                      f"{d['L_mm'][0]:>7.1f}{d['peak_g']:>7.2f}{d['eta']:>6.2f}"
                      f"{d['leg_stroke_mm']:>8.1f}{d['leg_mass_g']:>7.0f}{v:>16}")
    print(f"\n[e21] → {fp}")


if __name__ == "__main__":
    main()

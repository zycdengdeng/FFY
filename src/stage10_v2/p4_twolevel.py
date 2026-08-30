# -*- coding: utf-8 -*-
"""P4 · 两级优化:固定几何下,按工况调节刚度/阻尼到底值多少。

一架飞机 = 一副造死的几何(L1, r2, r3, θA, θK);它要落到很多种工况上。
四条对照线,同一副几何、同一批工况、同一判官:

  ① 纯被动      刚度也造死(参考工况下 cVAE 给的那组),全工况一套 —— 现在的板簧
  ② cVAE 前馈   每个工况查一次模型,取它的 (κ踝,κ膝,τ) 配到固定几何上 —— 实际控制器
  ③ 理想主动    每个工况在 (κ踝,κ膝,τ) 盒内真搜最优 —— 主动收益的上界
  ④ 带误差主动  用邻格工况的 ③ 最优设定,落在真实工况上 —— 感知估错一档的代价

κ髋 固定在几何设计自带的值:v2.1 实测它不随工况变(1.16×),不作为控制通道。
验收要求固定 g_cap=10g / s_max=24mm(变要求属另一维,不混进来)。
①–④ 的差值分别回答:主动值不值得做(③−①)、模型离上界多远(③−②)、
感知要做到多准(④−③)。

用法(A100,单行):
  OMP_NUM_THREADS=1 python src/stage10_v2/p4_twolevel.py --workers 128 --out outputs/v2_p4
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P
from bioprior import BioPrior, KAP_RANGE_V21, ZETA_RANGE_V21
from factory_v2 import zeta_of_kc
from e17_emergent_b import load as load_cvae
from train_cvae import norm

GCAP, SMAX = 10 * 9.81, 0.024
KA_RG, KK_RG = KAP_RANGE_V21[0], KAP_RANGE_V21[1]      # κ踝 [0.75,8] κ膝 [1.5,8]
TAU_RG = ZETA_RANGE_V21                                # τ [0.005,0.1]
BASE = {**P.SCEN_BIRD_X, "hip_damp_unified": True}
BIG = 1e6                                              # 不可行罚(排序用)


def _ev(a):
    """x9 → (score, g, ok, why)。score = 峰值g + 不可行罚,内层搜索按它排序。"""
    x9, m, v0, kc = a
    r = P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=BASE)
    if r is None or r.get("fail"):
        return (BIG * 2, None, False, [(r or {}).get("fail", "?")])
    ok, why = P.feasible_v2(r, GCAP, SMAX)
    g = r["peak_a"] / 9.81
    return (g + (0 if ok else BIG), g, bool(ok), list(why))


def assemble(geom, kap_h, ka, kk, tau):
    """固定几何 + 三个控制量 → 完整 9 维设计。geom=(L1,r2,r3,θA,θK)。"""
    L1, r2, r3, tA, tK = geom
    return [L1, r2, r3, float(ka), float(kk), float(kap_h), float(tau), tA, tK]


def loguni(rng, rg, n):
    lo, hi = np.log10(rg[0]), np.log10(rg[1])
    return 10 ** (lo + (hi - lo) * rng.random(n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v21_e5_bio/cvae_r39.pt",
                    help="r39:r40 有单轮抖动,取稳的")
    ap.add_argument("--masses", default="2,12")
    ap.add_argument("--n1", type=int, default=32, help="内层第一阶段 LHS 点数")
    ap.add_argument("--n2", type=int, default=24, help="内层第二阶段局部加密点数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/v2_p4")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    model, meta = load_cvae(a.ckpt)
    pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    import torch

    def cvae_x(m, v0, kc):
        c = np.array([np.log10(m), v0, np.log10(kc), GCAP, SMAX])
        torch.manual_seed(7)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, c_lo, c_hi),
                                          dtype=torch.float32), 64).numpy()
        return prior.expand(np.clip(u, 0, 1), m).mean(0)

    KCs = [5e4, 1e5, 2.5e5, 1e6]
    V0s = [0.6, 1.0, 1.4, 1.8]
    conds = [(kc, v0) for kc in KCs for v0 in V0s]
    rng = np.random.default_rng(a.seed)
    RES = {}

    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for m in [float(v) for v in a.masses.split(",")]:
            xref = cvae_x(m, 1.2, 1e5)                     # 参考工况:草地 1.2
            geom = (xref[0], xref[1], xref[2], xref[7], xref[8])
            kap_h = float(xref[5])
            pas = (float(xref[3]), float(xref[4]), float(xref[6]))
            print(f"\n[m={m:g}kg] 几何 L1={geom[0]:.1f} θ={geom[3]:.0f}/{geom[4]:.0f} "
                  f"κ髋={kap_h:.1f}(固定)  被动刚度 κ踝={pas[0]:.2f} κ膝={pas[1]:.2f} "
                  f"τ={pas[2]:.4f}", flush=True)

            # ---- 阶段 1:所有工况 × (被动 1 + cVAE 前馈 1 + LHS n1) 一批摔
            jobs, tag = [], []
            for ci, (kc, v0) in enumerate(conds):
                xc = cvae_x(m, v0, kc)
                for lab, (ka, kk, tau) in (("pas", pas),
                                           ("ff", (xc[3], xc[4], xc[6]))):
                    jobs.append((assemble(geom, kap_h, ka, kk, tau), m, v0, kc))
                    tag.append((ci, lab, (float(ka), float(kk), float(tau))))
                for ka, kk, tau in zip(loguni(rng, KA_RG, a.n1),
                                       loguni(rng, KK_RG, a.n1),
                                       loguni(rng, TAU_RG, a.n1)):
                    jobs.append((assemble(geom, kap_h, ka, kk, tau), m, v0, kc))
                    tag.append((ci, "s1", (float(ka), float(kk), float(tau))))
            r1 = list(ex.map(_ev, jobs, chunksize=1))
            print(f"  阶段1 {len(jobs)} 次完成", flush=True)

            # ---- 阶段 2:每工况取 s1∪ff 最好 3 点,对数邻域加密
            jobs2, tag2 = [], []
            best1 = {}
            for ci in range(len(conds)):
                cand = [(r1[i][0], tag[i][2]) for i in range(len(tag))
                        if tag[i][0] == ci and tag[i][1] in ("s1", "ff")]
                cand.sort(key=lambda t: t[0])
                best1[ci] = cand[0]
                kc, v0 = conds[ci]
                for _, (ka, kk, tau) in cand[:3]:
                    for _ in range(a.n2 // 3):
                        f = lambda v, rg: float(np.clip(
                            v * 10 ** rng.normal(0, 0.12), rg[0], rg[1]))
                        p3 = (f(ka, KA_RG), f(kk, KK_RG), f(tau, TAU_RG))
                        jobs2.append((assemble(geom, kap_h, *p3), m, v0, kc))
                        tag2.append((ci, p3))
            r2 = list(ex.map(_ev, jobs2, chunksize=1))
            print(f"  阶段2 {len(jobs2)} 次完成", flush=True)

            # ---- 汇总每工况四条线;理想主动 = 全部候选最优
            per = []
            for ci, (kc, v0) in enumerate(conds):
                allc = ([(r1[i][0], r1[i], tag[i][2]) for i in range(len(tag))
                         if tag[i][0] == ci] +
                        [(r2[i][0], r2[i], tag2[i][1]) for i in range(len(tag2))
                         if tag2[i][0] == ci])
                sc, rb, pb = min(allc, key=lambda t: t[0])
                get = lambda lab: next(r1[i] for i in range(len(tag))
                                       if tag[i][0] == ci and tag[i][1] == lab)
                rp, rf = get("pas"), get("ff")
                per.append(dict(kc=kc, v0=v0,
                                passive=dict(g=rp[1], ok=rp[2], why=rp[3]),
                                ff=dict(g=rf[1], ok=rf[2], why=rf[3]),
                                active=dict(g=rb[1], ok=rb[2], why=rb[3],
                                            ka=pb[0], kk=pb[1], tau=pb[2])))

            # ---- 阶段 3:带误差主动 —— 用邻格最优设定落真实工况
            jobs3, tag3 = [], []
            for ci, (kc, v0) in enumerate(conds):
                ki, vi = KCs.index(kc), V0s.index(v0)
                for dk, dv in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    kj, vj = ki + dk, vi + dv
                    if not (0 <= kj < len(KCs) and 0 <= vj < len(V0s)):
                        continue
                    cj = conds.index((KCs[kj], V0s[vj]))
                    pa = per[cj]["active"]
                    jobs3.append((assemble(geom, kap_h, pa["ka"], pa["kk"],
                                           pa["tau"]), m, v0, kc))
                    tag3.append(ci)
            r3 = list(ex.map(_ev, jobs3, chunksize=1))
            print(f"  阶段3 {len(jobs3)} 次完成", flush=True)
            for ci in range(len(conds)):
                es = [r3[i] for i in range(len(tag3)) if tag3[i] == ci]
                gs = [e[1] for e in es if e[1] is not None]
                per[ci]["err"] = dict(
                    g_mean=(float(np.mean(gs)) if gs else None),
                    g_worst=(float(np.max(gs)) if gs else None),
                    ok_frac=float(np.mean([e[2] for e in es])) if es else None)
            RES[m] = dict(geom=list(geom), kap_hip=kap_h, passive_stiff=list(pas),
                          conds=per)

            # ---- 现场表
            print(f"\n  {'kc':>8}{'v0':>5} │{'①被动':>8}{'②前馈':>8}{'③理想':>8}"
                  f"{'④误差worst':>11} │ 可行 ①②③④")
            for c in per:
                f = lambda d: ("  —  " if d["g"] is None else f"{d['g']:5.2f}")
                fw = "  —  " if c["err"]["g_worst"] is None else f"{c['err']['g_worst']:5.2f}"
                oks = "".join("✓" if d["ok"] else "✗" for d in
                              (c["passive"], c["ff"], c["active"]))
                oke = "✓" if (c["err"]["ok_frac"] or 0) == 1 else \
                      ("△" if (c["err"]["ok_frac"] or 0) > 0 else "✗")
                print(f"  {c['kc']:>8.0e}{c['v0']:>5.1f} │{f(c['passive']):>8}"
                      f"{f(c['ff']):>8}{f(c['active']):>8}{fw:>11} │  {oks}{oke}")

    json.dump(RES, open(os.path.join(a.out, "p4_twolevel.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"\n[p4] → {a.out}/p4_twolevel.json")


if __name__ == "__main__":
    main()

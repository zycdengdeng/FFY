# -*- coding: utf-8 -*-
"""P4c-abl · 连杆耦合的三臂消融(高预算,多种子,带失败格救援)。

P4c 初版只有"独立 vs 耦合"两臂,而耦合同时改了两件事:
  (i) 去掉膝关节那套减震器   (ii) 用连杆锁掉膝的自由度
两者混在一起,无法归因。本实验补上中间臂,做成严格消融:

  A   独立 + 膝簧    搜 (κ踝, κ膝, τ),κ髋 固定,无连杆      [现状]
  A0  独立 − 膝簧    搜 (κ踝, κ髋, τ),κ膝 = 0,无连杆       [只去掉膝减震器]
  B   耦合 − 膝簧    搜 (κ踝, κ髋, τ),κ膝 = 0,有连杆       [再锁掉自由度]

  A → A0  = 去掉一套减震器的代价
  A0 → B  = **锁自由度的净代价(唯一变量就是那根连杆)**  ← 本实验的主结论

三处相对初版的加固:
 1. 搜索预算 n1=64 / n2=48(初版 32/24);
 2. **失败格救援**:任何判为不可行的格,再跑一遍稠密网格确认;
    初版发现打分函数 score=g+BIG 会让失败格报出"最软的不可行解",没有诊断价值,
    且 12kg 有一格是搜索漏解而非架构做不到 —— 救援把"没搜到"和"做不到"分开;
 3. 多种子(默认 0,1),报种子间散布。

用法(A100,单行):
  OMP_NUM_THREADS=1 python src/stage10_v2/p4c_ablation.py --workers 128 --out outputs/v2_p4c_abl
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
KA_RG, KK_RG, KH_RG = KAP_RANGE_V21
TAU_RG = ZETA_RANGE_V21
ROD = dict(off_hip=0.050, off_knee=0.050)
BIG = 1e6
KCs = [5e4, 1e5, 2.5e5, 1e6]; V0s = [0.6, 1.0, 1.4, 1.8]
CONDS = [(kc, v0) for kc in KCs for v0 in V0s]
ARMS = {"A":  dict(label="独立+膝簧", rod=False, free="knee"),
        "A0": dict(label="独立−膝簧", rod=False, free="hip"),
        "B":  dict(label="耦合−膝簧", rod=True,  free="hip")}


def _ev(a):
    x9, m, v0, kc, rod = a
    base = {**P.SCEN_BIRD_X, "hip_damp_unified": True}
    if rod:
        base["couple_rod"] = ROD
    r = P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=base)
    if r is None or r.get("fail"):
        return (BIG * 2, None, None, False, [(r or {}).get("fail", "?")])
    ok, why = P.feasible_v2(r, GCAP, SMAX)
    g = r["peak_a"] / 9.81
    return (g + (0 if ok else BIG), g, r["leg_stroke_mm"], bool(ok), list(why))


def assemble(geom, arm, p, kap_hip):
    """p = (p0, p1, p2) = (κ踝, κ膝或κ髋, τ)。"""
    L1, r2, r3, tA, tK = geom
    if ARMS[arm]["free"] == "knee":
        ka, kk, kh = p[0], p[1], kap_hip
    else:
        ka, kk, kh = p[0], 1e-6, p[1]
    return [L1, r2, r3, float(ka), float(kk), float(kh), float(p[2]), tA, tK]


def rg_of(arm):
    return (KA_RG, KK_RG if ARMS[arm]["free"] == "knee" else KH_RG, TAU_RG)


def loguni(rng, rg, n):
    lo, hi = np.log10(rg[0]), np.log10(rg[1])
    return 10 ** (lo + (hi - lo) * rng.random(n))


def search(ex, geom, kap_hip, arm, m, rng, n1, n2):
    RG = rg_of(arm); rod = ARMS[arm]["rod"]
    jobs, tag = [], []
    for ci, (kc, v0) in enumerate(CONDS):
        for p in zip(loguni(rng, RG[0], n1), loguni(rng, RG[1], n1), loguni(rng, RG[2], n1)):
            pp = tuple(float(v) for v in p)
            jobs.append((assemble(geom, arm, pp, kap_hip), m, v0, kc, rod)); tag.append((ci, pp))
    r1 = list(ex.map(_ev, jobs, chunksize=2))
    jobs2, tag2 = [], []
    for ci, (kc, v0) in enumerate(CONDS):
        cand = sorted([(r1[i][0], tag[i][1]) for i in range(len(tag)) if tag[i][0] == ci],
                      key=lambda t: t[0])[:3]
        for _, pp in cand:
            for _ in range(max(1, n2 // 3)):
                f = lambda v, r: float(np.clip(v * 10 ** rng.normal(0, 0.12), r[0], r[1]))
                q = (f(pp[0], RG[0]), f(pp[1], RG[1]), f(pp[2], RG[2]))
                jobs2.append((assemble(geom, arm, q, kap_hip), m, v0, kc, rod)); tag2.append((ci, q))
    r2 = list(ex.map(_ev, jobs2, chunksize=2))
    out = []
    for ci in range(len(CONDS)):
        allc = ([(r1[i], tag[i][1]) for i in range(len(tag)) if tag[i][0] == ci] +
                [(r2[i], tag2[i][1]) for i in range(len(tag2)) if tag2[i][0] == ci])
        r, pp = min(allc, key=lambda t: t[0][0])
        out.append(dict(kc=CONDS[ci][0], v0=CONDS[ci][1], g=r[1], stroke=r[2],
                        ok=r[3], why=r[4], params=list(pp), rescued=False))
    return out


def rescue(ex, geom, kap_hip, arm, m, cells, nr):
    """对失败格跑稠密网格,把"没搜到"和"架构做不到"分开。"""
    RG = rg_of(arm); rod = ARMS[arm]["rod"]
    na, nb, nt = nr, nr, max(4, nr // 2)
    jobs, tag = [], []
    for ci in cells:
        kc, v0 = CONDS[ci]
        for a_ in np.geomspace(*RG[0], na):
            for b_ in np.geomspace(*RG[1], nb):
                for t_ in np.geomspace(*RG[2], nt):
                    pp = (float(a_), float(b_), float(t_))
                    jobs.append((assemble(geom, arm, pp, kap_hip), m, v0, kc, rod))
                    tag.append((ci, pp))
    if not jobs:
        return {}
    res = list(ex.map(_ev, jobs, chunksize=4))
    found = {}
    for ci in cells:
        ok = [(res[i], tag[i][1]) for i in range(len(tag)) if tag[i][0] == ci and res[i][3]]
        if ok:
            r, pp = min(ok, key=lambda t: t[0][1])
            found[ci] = dict(g=r[1], stroke=r[2], params=list(pp))
    return found


def robust(ex, geom, kap_hip, arm, m, cands, good):
    rod = ARMS[arm]["rod"]
    jobs = [(assemble(geom, arm, pp, kap_hip), m, v0, kc, rod)
            for pp in cands for (kc, v0) in CONDS]
    res = list(ex.map(_ev, jobs, chunksize=4))
    rows = []
    for i, pp in enumerate(cands):
        rr = res[i * len(CONDS):(i + 1) * len(CONDS)]
        okg = [rr[j][1] for j in good if rr[j][3] and rr[j][1]]
        rows.append(dict(params=list(pp), cover=sum(1 for j in good if rr[j][3]), n=len(good),
                         g_med=(float(np.median(okg)) if okg else None),
                         g_worst=(float(np.max(okg)) if okg else None)))
    rows.sort(key=lambda r: (-r["cover"], r["g_worst"] or 99))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v21_e5_bio/cvae_r39.pt")
    ap.add_argument("--masses", default="2,12")
    ap.add_argument("--seeds", default="0,1")
    ap.add_argument("--n1", type=int, default=64)
    ap.add_argument("--n2", type=int, default=48)
    ap.add_argument("--nrescue", type=int, default=10, help="救援网格每维点数(τ 取一半)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_p4c_abl")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import torch
    model, meta = load_cvae(a.ckpt); pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])

    def cvae_x(m, v0, kc):
        c = np.array([np.log10(m), v0, np.log10(kc), GCAP, SMAX]); torch.manual_seed(7)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32), 64).numpy()
        return prior.expand(np.clip(u, 0, 1), m).mean(0)

    seeds = [int(v) for v in a.seeds.split(",")]
    OUT = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for m in [float(v) for v in a.masses.split(",")]:
            xr = cvae_x(m, 1.2, 1e5)
            geom = (xr[0], xr[1], xr[2], xr[7], xr[8]); kap_hip = float(xr[5])
            good = [i for i, c in enumerate(CONDS) if not (m == 12.0 and c == (1e6, 1.8))]
            print(f"\n{'='*74}\nm = {m:g} kg   L1={geom[0]:.1f}  θ={geom[3]:.0f}/{geom[4]:.0f}  "
                  f"κ髋(A臂固定)={kap_hip:.1f}   有效工况 {len(good)}", flush=True)
            per_seed = {}
            for sd in seeds:
                rng = np.random.default_rng(sd); res = {}
                for arm in ARMS:
                    r = search(ex, geom, kap_hip, arm, m, rng, a.n1, a.n2)
                    bad = [i for i in good if not r[i]["ok"]]
                    fnd = rescue(ex, geom, kap_hip, arm, m, bad, a.nrescue)
                    for ci, v in fnd.items():
                        r[ci].update(g=v["g"], stroke=v["stroke"], params=v["params"],
                                     ok=True, why=[], rescued=True)
                    nres = len(fnd)
                    print(f"  seed{sd} [{arm:<2} {ARMS[arm]['label']}] 覆盖 "
                          f"{sum(1 for i in good if r[i]['ok'])}/{len(good)}"
                          + (f"  (救援救回 {nres} 格)" if nres else ""), flush=True)
                    res[arm] = r
                per_seed[sd] = res

            # ---------- 汇总 ----------
            print(f"\n  {'':>16}{'覆盖':>10}{'共同可行处峰值中位':>20}")
            summ = {}
            for arm in ARMS:
                cov = [sum(1 for i in good if per_seed[sd][arm][i]["ok"]) for sd in seeds]
                summ[arm] = dict(cov=cov)
                print(f"  {arm:<3}{ARMS[arm]['label']:<13}{'/'.join(map(str,cov)):>8}/{len(good)}", flush=True)
            # 三臂都可行的格,用于公平比峰值
            for sd in seeds:
                common = [i for i in good if all(per_seed[sd][arm][i]["ok"] for arm in ARMS)]
                if not common: continue
                gv = {arm: np.median([per_seed[sd][arm][i]["g"] for i in common]) for arm in ARMS}
                print(f"\n  seed{sd}  三臂共同可行 {len(common)} 格,峰值中位:")
                print(f"     A {gv['A']:.2f}g → A0 {gv['A0']:.2f}g "
                      f"({(gv['A0']-gv['A'])/gv['A']*100:+.1f}%,去掉膝减震器的代价)")
                print(f"     A0 {gv['A0']:.2f}g → B {gv['B']:.2f}g "
                      f"({(gv['B']-gv['A0'])/gv['A0']*100:+.1f}%,**锁自由度的净代价**)")
                d = [(per_seed[sd]['B'][i]['g']-per_seed[sd]['A0'][i]['g'])/per_seed[sd]['A0'][i]['g']*100
                     for i in common]
                print(f"     A0→B 逐格 Δ 中位 {np.median(d):+.1f}%  "
                      f"(B 更好 {sum(1 for x in d if x<0)}/{len(d)} 格)")
            # 鲁棒被动(用 seed0 的逐格最优当候选池)
            print(f"\n  鲁棒被动(一组固定设定打全部工况):")
            rob = {}
            for arm in ARMS:
                cands = [per_seed[seeds[0]][arm][i]["params"] for i in good]
                rob[arm] = robust(ex, geom, kap_hip, arm, m, cands, good)
                r0 = rob[arm][0]
                gm = "—" if r0["g_med"] is None else f"{r0['g_med']:.2f}"
                gw = "—" if r0["g_worst"] is None else f"{r0['g_worst']:.2f}"
                print(f"     {arm:<3}{ARMS[arm]['label']:<13} 覆盖 {r0['cover']:>2}/{r0['n']}"
                      f"   g中位 {gm}   g最坏 {gw}", flush=True)
            OUT[m] = dict(geom=list(geom), kap_hip=kap_hip, good_ci=good, seeds=seeds,
                          per_seed={str(k): v for k, v in per_seed.items()}, robust=rob)
    json.dump(OUT, open(os.path.join(a.out, "p4c_ablation.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n[p4c-abl] → {a.out}/p4c_ablation.json")


if __name__ == "__main__":
    main()

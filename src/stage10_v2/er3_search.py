# -*- coding: utf-8 -*-
"""E-R3 · 生成器无关的近最优标度检验(预注册协议见 stilt/审稿回应_行动计划_v1.md v2)。

双层搜索空间:
  --space bio|geo|elastic|none : 该臂先验盒内搜索(评"先验如何影响结果")
  --space common               : 共同工程空间(评"物理+目标偏好什么尺度关系")
      维度 1–9 与四臂完全相同(本就是绝对范围);
      L1 ∈ [20, 350] mm **常数区间,不随质量平移** —— 待检验的指数无法经边界走私。
      区间在对数尺度上以 ≥30% 裕量严格包含四臂盒在 4–36 kg 的并集([~30,300]);
      贴边检查:近最优集内 log10(L1) 落在边界 5% 带内的比例 >10% 即判空间无效并报警。

方法(≥2,互为对照):
  random  纯随机搜索          de  scipy differential_evolution(种子化)

近最优集:可行 且 目标 ≤ (1+ε)·该质量点最优,ε∈{3%,5%,10%};报 L1 带(P5/中位/P95)。
目标:a_res(缺则 peak_a),固定正则工况 v0=1.2, kc=1e5, zc=0.15, gcap=10g, smax=24mm。
用法: python src/stage10_v2/er3_search.py --space common --mass 12 --method de \
        --budget 3000 --seed 0 --workers 32 --out outputs/er3
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import physics_v2 as P                                 # noqa: E402
from bioprior import (BioPrior, R2_RANGE, R3_RANGE, KAP_RANGE_V21, ZETA_RANGE_V21,
                      THA_RANGE, THK_RANGE, Q1_RANGE_V25)   # noqa: E402

GCAP, SMAX = 10 * 9.81, 0.024
COND = dict(v0=1.2, kc=1e5, zeta_c=0.15)
L1_COMMON = (20.0, 350.0)          # mm,常数区间(共同空间的关键性质)
EDGE_BAND = 0.05                   # 贴边判定:log10 归一坐标的边界带宽

RANGES9 = [R2_RANGE, R3_RANGE] + list(KAP_RANGE_V21) + [ZETA_RANGE_V21,
           THA_RANGE, THK_RANGE, Q1_RANGE_V25]

def make_space(space, m):
    """返回 (lo10, hi10):10 维物理设计的边界。common 的 L1 是常数区间;
    臂空间的 L1 界取该臂盒在质量 m 处的 ±u_max·σ(即先验盒本身)。"""
    if space == "common":
        l1lo, l1hi = L1_COMMON
    else:
        pr = BioPrior(space, v21=True, v25=True)
        c = pr.a + pr.b * np.log10(m * 1000.0)
        l1lo, l1hi = 10**(c - pr.u_max*pr.sigma), 10**(c + pr.u_max*pr.sigma)
    lo = [l1lo] + [r[0] for r in RANGES9]
    hi = [l1hi] + [r[1] for r in RANGES9]
    return np.array(lo, float), np.array(hi, float)

def to_x(z, lo, hi):
    """z∈[0,1]^10 → 物理设计;L1 维在对数尺度插值(其余线性)。"""
    x = lo + z * (hi - lo)
    x[0] = 10 ** (np.log10(lo[0]) + z[0] * (np.log10(hi[0]) - np.log10(lo[0])))
    return x

def evaluate(args):
    x, m = args
    try:
        r = P.eval_v2(list(x), m, COND["v0"], COND["kc"], zeta_c=COND["zeta_c"])
        if r.get("fail"):
            return None, None, ["sim_fail"]
        ok, viol = P.feasible_v2(r, GCAP, SMAX)
        obj = float(r.get("a_res", r.get("peak_a")))
        return obj, bool(ok), list(viol)
    except Exception as e:                              # noqa: BLE001
        return None, None, [f"exc:{type(e).__name__}"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--space", required=True,
                    choices=["common", "bio", "bio407", "geo", "elastic", "none"])
    ap.add_argument("--mass", type=float, required=True)
    ap.add_argument("--method", required=True, choices=["random", "de"])
    ap.add_argument("--budget", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--out", default="outputs/er3")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    lo, hi = make_space(a.space, a.mass)
    rng = np.random.default_rng(a.seed)
    from multiprocessing import Pool
    pool = Pool(a.workers)
    log = []                                   # (x10, obj, feasible)

    def batch(Z):
        X = [to_x(z, lo, hi) for z in Z]
        res = pool.map(evaluate, [(x, a.mass) for x in X])
        for x, (obj, ok, viol) in zip(X, res):
            log.append((list(map(float, x)), obj, ok))
        return np.array([o if (o is not None and ok) else 1e9
                         for (o, ok, _) in res])

    t0 = time.time()
    if a.method == "random":
        n = 0
        while n < a.budget:
            k = min(a.workers * 8, a.budget - n)
            batch(rng.random((k, 10))); n += k
    else:
        from scipy.optimize import differential_evolution
        popsize = 24
        maxiter = max(1, a.budget // (popsize * 10) - 1)
        differential_evolution(
            lambda zz: float(batch(np.atleast_2d(zz))[0]) if np.ndim(zz) == 1 else None,
            bounds=[(0, 1)]*10, seed=a.seed, popsize=popsize, maxiter=maxiter,
            vectorized=False, workers=1, polish=False, updating="deferred",
            init=rng.random((popsize*10, 10)))

    feas = [(x, o) for x, o, ok in log if ok and o is not None]
    out = dict(space=a.space, mass=a.mass, method=a.method, seed=a.seed,
               budget=a.budget, n_eval=len(log), n_feas=len(feas),
               minutes=round((time.time()-t0)/60, 1),
               cond=COND, gcap=GCAP, smax=SMAX,
               l1_bounds=[float(lo[0]), float(hi[0])])
    if feas:
        best = min(o for _, o in feas)
        out["best_obj"] = best
        for eps in (0.03, 0.05, 0.10):
            sel = [x[0] for x, o in feas if o <= best * (1 + eps)]
            z01 = [(np.log10(v) - np.log10(lo[0])) /
                   (np.log10(hi[0]) - np.log10(lo[0])) for v in sel]
            edge = float(np.mean([(z < EDGE_BAND) or (z > 1 - EDGE_BAND) for z in z01])) if sel else None
            out[f"band_eps{int(eps*100)}"] = dict(
                n=len(sel), L1_p5=float(np.percentile(sel, 5)),
                L1_med=float(np.median(sel)), L1_p95=float(np.percentile(sel, 95)),
                edge_frac=edge)
    tag = f"{a.space}_m{a.mass:g}_{a.method}_s{a.seed}"
    json.dump(out, open(os.path.join(a.out, f"er3_{tag}.json"), "w"), indent=1)
    np.save(os.path.join(a.out, f"er3_{tag}_log.npy"),
            np.array([x + [o if o is not None else np.nan, float(bool(ok))]
                      for x, o, ok in log], float))
    print(json.dumps(out, indent=1, ensure_ascii=False))

if __name__ == "__main__":
    main()

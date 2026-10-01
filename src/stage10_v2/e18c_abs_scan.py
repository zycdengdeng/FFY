# -*- coding: utf-8 -*-
"""E18c · 绝对腿长扫描(生成器无关、先验无关;E-R4b 分解矩阵的唯一数据源)

思路:与其为每个(锚点×指数×宽度)组合各跑一遍走廊,不如直接在**绝对腿长 L1** 的对数网格
上扫一遍可行率 f(工况, m, L1)。任何先验盒(中心 L_c(m)=L0·10^(kσ)·(m/m0)^b,半宽 w·u_max·σ)
的"盒内可行率"都是这张表上的一段平均——锚点、指数、宽度三个成分的分解全部变成后处理
(er4b_decompose.py),一次扫描,不再逐臂重跑。

每个探针除可行与否外还落盘 a_res / leg_stroke / leg_mass,所以"可行率最高的腿长"和
"峰值最低的腿长"两种口径都能从同一份数据读出(ER3 试跑显示两者方向可能相反)。

物理配置一律取自 evalcfg(D-2 之后的铁律),输出带 stamp。

用法(A100,单行):
  OMP_NUM_THREADS=1 python src/stage10_v2/e18c_abs_scan.py --workers 96 --nprobe 96 --out outputs/e18c_scan
成本:19 长度 × 9 质量 × 96 探针 × 6 工况 = 98,496 次评价(每次 2 遍),约为 g1b_e18b 的 0.6 倍。
ENG-prior 开发集(2026-10-01,R-H;与测试集质量/工况/种子互不重合):
  python src/stage10_v2/e18c_abs_scan.py --conds soil1.3,soil1.7,loam1.3,loam1.7 --nm 5 --mlo 4.5 --mhi 30 --nprobe 48 --seed 1 --out outputs/e18c_dev
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evalcfg as E                      # noqa: E402

# 违反项词表(位掩码);e18b 的分桶:前六条工程失效,deep_sink 模型失效,其余数值失败
VOCAB = ["gcap", "smax", "slenderness", "massbudget", "rebound", "slip",
         "deep_sink", "collapse", "solver", "nonfinite", "none", "unknown"]
BIT = {k: 1 << i for i, k in enumerate(VOCAB)}
CODE_OK, CODE_INFEAS, CODE_INVALID, CODE_UNSOLVED = 0, 1, 2, 3


def _seed(*parts):
    return zlib.crc32(("|".join(map(str, parts))).encode()) % (2 ** 31)


def classify(ok, why):
    if ok:
        return CODE_OK
    w = set(why)
    if w & {"solver", "nonfinite", "none", "collapse", "unknown"}:
        return CODE_UNSOLVED
    if "deep_sink" in w:
        return CODE_INVALID
    return CODE_INFEAS


def _probe_one(a):
    x, m, v0, kc = a
    try:
        r = E.evaluate(x, m, v0, kc)
    except Exception as e:                       # noqa: BLE001
        return (CODE_UNSOLVED, BIT["unknown"], np.nan, np.nan, np.nan, type(e).__name__)
    ok, why = E.gates(r)
    mask = 0
    for w in why:
        mask |= BIT.get(w, BIT["unknown"])
    g = lambda k: float(r.get(k, np.nan)) if isinstance(r, dict) else np.nan
    return (classify(ok, why), mask, g("a_res") if np.isfinite(g("a_res")) else g("peak_a"),
            g("leg_stroke"), g("leg_mass_kg"), None)


def run_cond(cname, Ls, ms, nprobe, workers, outdir, seed):
    cd = {**E.CONDS, **E.DEV_CONDS}[cname]
    kc, v0 = cd["kc"], cd["v0"]
    pr = E.prior("bio")                          # 只借它展开其余 9 维;L1 之后覆盖为绝对值
    jobs, tags = [], []
    for mi, m in enumerate(ms):
        for li, L in enumerate(Ls):
            U = E.lhs(nprobe, pr.ndim, np.random.default_rng(_seed("e18c", seed, cname, mi, li)))
            X = pr.expand(U, float(m))
            X[:, 0] = float(L)                   # 绝对腿长,与任何先验无关
            for x in X:
                jobs.append((tuple(map(float, x)), float(m), v0, kc))
                tags.append((mi, li))
    print(f"[{cname}] {len(ms)}级 × {len(Ls)}长 × {nprobe}探针 = {len(jobs)} 次评价", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_probe_one, jobs, chunksize=4))
    print(f"[{cname}] 完成 ({time.time() - t0:.0f}s)", flush=True)

    nm, nl = len(ms), len(Ls)
    code = np.full((nm, nl, nprobe), CODE_UNSOLVED, np.int8)
    mask = np.zeros((nm, nl, nprobe), np.int16)
    ares = np.full((nm, nl, nprobe), np.nan, np.float32)
    strk = np.full((nm, nl, nprobe), np.nan, np.float32)
    lmas = np.full((nm, nl, nprobe), np.nan, np.float32)
    cnt = {}
    unknown = {}
    for (mi, li), (c, mk, a, s, lm, exc) in zip(tags, out):
        k = cnt[(mi, li)] = cnt.get((mi, li), 0)
        code[mi, li, k], mask[mi, li, k] = c, mk
        ares[mi, li, k], strk[mi, li, k], lmas[mi, li, k] = a, s, lm
        cnt[(mi, li)] = k + 1
        if exc:
            unknown[exc] = unknown.get(exc, 0) + 1

    f = (code == CODE_OK).mean(2)
    f_judged = (code == CODE_OK).sum(2) / np.maximum((code <= CODE_INFEAS).sum(2), 1)
    r_invalid = (code == CODE_INVALID).mean(2)
    r_unsolved = (code == CODE_UNSOLVED).mean(2)
    crit = {k: ((mask & BIT[k]) > 0).mean(2).round(4).tolist() for k in VOCAB[:8]}
    feas = code == CODE_OK
    with np.errstate(all="ignore"):
        a_med = np.where(feas.sum(2) > 0, np.nanmedian(np.where(feas, ares, np.nan), 2), np.nan)
        a_min = np.where(feas.sum(2) > 0, np.nanmin(np.where(feas, ares, np.nan), 2), np.nan)
    blob = dict(cond=cname, kc=kc, v0=v0, label=cd["label"], nprobe=nprobe, seed=seed,
                m_grid=[round(float(v), 4) for v in ms], L_grid=[round(float(v), 3) for v in Ls],
                vocab=VOCAB, stamp=E.stamp(),
                f=f.round(4).tolist(), f_judged=f_judged.round(4).tolist(),
                r_invalid=r_invalid.round(4).tolist(), r_unsolved=r_unsolved.round(4).tolist(),
                crit=crit,
                a_res_med_feas=np.where(np.isfinite(a_med), a_med, -1).round(3).tolist(),
                a_res_min_feas=np.where(np.isfinite(a_min), a_min, -1).round(3).tolist(),
                unknown_exceptions=unknown)
    json.dump(blob, open(os.path.join(outdir, f"e18c_{cname}.json"), "w"), indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(outdir, f"e18c_{cname}_probes.npz"),
                        code=code, mask=mask, a_res=ares, leg_stroke=strk, leg_mass=lmas,
                        m_grid=np.asarray(ms), L_grid=np.asarray(Ls))
    # 现场可读:行=质量,列=腿长,单元=可行率%
    print("  m\\L " + "".join(f"{L:>5.0f}" for L in Ls))
    for mi, m in enumerate(ms):
        print(f"  {m:5.1f}" + "".join(f"{v*100:>5.0f}" for v in f[mi]))
    j = np.argmax(f, 1)
    print("  可行率最高腿长:", " ".join(f"{m:.1f}kg→{Ls[j[mi]]:.0f}mm({f[mi, j[mi]]:.2f})" for mi, m in enumerate(ms)))
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--conds", default=",".join(E.CONDS))
    ap.add_argument("--nl", type=int, default=19)
    ap.add_argument("--llo", type=float, default=30.0)
    ap.add_argument("--lhi", type=float, default=350.0)
    ap.add_argument("--nm", type=int, default=9)
    ap.add_argument("--mlo", type=float, default=4.0)
    ap.add_argument("--mhi", type=float, default=36.0)
    ap.add_argument("--nprobe", type=int, default=96)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--check-meta", default="outputs/v25_bird_e5/model_meta.json")
    ap.add_argument("--out", default="outputs/e18c_scan")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.check_meta and os.path.exists(a.check_meta):
        E.check_meta(a.check_meta)
    Ls = np.geomspace(a.llo, a.lhi, a.nl)
    ms = np.geomspace(a.mlo, a.mhi, a.nm)
    allc = {**E.CONDS, **E.DEV_CONDS}
    conds = [c for c in a.conds.split(",") if c in allc]
    assert conds, f"未知工况;可选:{list(allc)}"
    print(f"[e18c] stamp: {json.dumps(E.stamp(), ensure_ascii=False)}")
    print(f"[e18c] {len(conds)} 工况 × {a.nm}×{a.nl}×{a.nprobe} = {len(conds)*a.nm*a.nl*a.nprobe} 次评价\n")
    json.dump(dict(stamp=E.stamp(), args=vars(a)), open(os.path.join(a.out, "stamp.json"), "w"),
              indent=1, ensure_ascii=False)
    for c in conds:
        if os.path.exists(os.path.join(a.out, f"e18c_{c}.json")):
            print(f"[{c}] 已有结果,跳过"); continue
        run_cond(c, Ls, ms, a.nprobe, a.workers, a.out, a.seed)
        print()
    print(f"[e18c] 全部完成 → {a.out}")


if __name__ == "__main__":
    main()

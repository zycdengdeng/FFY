"""E6 · 多指标考试:η 进条件向量 + 回弹硬闸 + 按体重/速度分层出分。

三个升级(逐一回应"只看峰值"批评与 7 维回弹复核发现):
  ① 条件向量 4→5 维:c = (m, v0, g_cap, s_max, η_min)——客户可以点名要缓冲效率;
  ② 回弹硬闸:回弹 > 0 的设计一律不可行(指标现成,零成本;堵住弹跳设计蒙混);
  ③ 分层出分:考卷按 体重 {1-4, 4-8, 8-12}kg × 速度 {0.5-1.25, 1.25-2.0}m/s
     六格出分——对应飞机按吨位/下沉速度分级的工程叙事。

训练零新增仿真(95 万池的 η/回弹全部在案);评测新增:
  新考卷参考 + 验证参考 ≈ 3 万次,种子评测 ≈ 1.2 万次,BO-9-E6 ≈ 700 次。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage7_generative/e6_exam.py \
    --factory outputs/gen_data7/factory.jsonl --data outputs/gen_data7/gen_dataset.npz \
    --inc outputs/gen_e5/pool_increments.jsonl --out outputs/gen_e6 \
    --nseeds 3 --workers 128
产出:e6_results.json(总分 + 分层表 + BO 对照)+ cvae_e6_s*.pt + e6_refs.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import GCAP_RANGE, SMAX_RANGE, pareto2   # noqa: E402
from data_factory import _eval_one, KEYS                    # noqa: E402
from e5_loop import load_pools, lhs                         # noqa: E402
from train_cvae import CVAE, fit, norm                      # noqa: E402

iP, iS = KEYS.index("peak_a"), KEYS.index("stroke")
iE, iR = KEYS.index("eta"), KEYS.index("rebound")
iB = KEYS.index("n_bounce")
ETA_RANGE = (0.50, 0.80)        # η_min 抽样区间默认值(可用 --eta-lo/--eta-hi 覆盖)
REB_CAP = 0.05                  # 软闸:回能比上限(回弹高/等效落高;见《回弹判据调研》)

M_EDGES = [1.0, 4.0, 8.0, 12.0]
V_EDGES = [0.5, 1.25, 2.0]


def feas_mask(Y, gcap, smax, eta_min, v0):
    """考规 v2:峰值/行程/η 达标 + 两级回弹闸(硬:足端不离地;软:回能比≤REB_CAP)。"""
    ok = np.isfinite(Y[:, iP])
    h_eq = max(v0 * v0 / (2 * 9.81), 1e-9)
    er = np.nan_to_num(Y[:, iR], nan=np.inf) / h_eq
    nb = np.nan_to_num(Y[:, iB], nan=np.inf)
    return (ok & (Y[:, iP] <= gcap) & (Y[:, iS] <= smax)
            & (Y[:, iE] >= eta_min) & (nb <= 0.5) & (er <= REB_CAP))


def stratum(m, v0):
    mi = min(int(np.searchsorted(M_EDGES, m, "right")) - 1, len(M_EDGES) - 2)
    vi = min(int(np.searchsorted(V_EDGES, v0, "right")) - 1, len(V_EDGES) - 2)
    return f"m{M_EDGES[mi]:g}-{M_EDGES[mi+1]:g}kg × v{V_EDGES[vi]:g}-{V_EDGES[vi+1]:g}"


def build_pairs_e6(pools, test_cids, kscen):
    """池 → 5 维条件训练对(约束场景种子 88_000+cid,与旧 77_000 系独立)。"""
    C_tr, X_tr, n_empty = [], [], 0
    for p in pools.values():
        if p.cid in test_cids:
            continue
        ok = np.isfinite(p.Y[:, iP])
        X, Y = p.X[ok], p.Y[ok]
        crng = np.random.default_rng(88_000 + p.cid)
        for _ in range(kscen):
            gcap = crng.uniform(*GCAP_RANGE) * 9.81
            smax = crng.uniform(*SMAX_RANGE)
            emin = crng.uniform(*ETA_RANGE)
            feas = feas_mask(Y, gcap, smax, emin, p.v0)
            if feas.sum() < 3:
                n_empty += 1; continue
            front = pareto2(Y[feas][:, [iP, iS]])
            for x in X[feas][front]:
                C_tr.append([p.m, p.v0, gcap, smax, emin]); X_tr.append(x.tolist())
    return np.array(C_tr), np.array(X_tr), n_empty


def build_refs_e6(ex, conds, lo, hi, nref, cache_fp):
    """参考前沿:每题 300 LHS 实摔,**存全指标**(未来换考规可免费重判)。"""
    if os.path.exists(cache_fp):
        return json.load(open(cache_fp))
    jobs, slices, ofs = [], [], 0
    for si, c in enumerate(conds):
        rng = np.random.default_rng(700_000 + si)
        X = lo + (hi - lo) * lhs(nref, len(lo), rng)
        jobs += [(x, c[0], c[1]) for x in X]
        slices.append(slice(ofs, ofs + nref)); ofs += nref
    rows = list(ex.map(_eval_one, jobs, chunksize=8))
    Yall = np.array([[np.nan if v is None else v for v in r] for r in rows])
    cache = []
    for si, c in enumerate(conds):
        m, v0, gcap, smax, emin = c
        Y = Yall[slices[si]]
        feas = feas_mask(Y, gcap, smax, emin, v0)
        if not feas.any():
            continue
        cache.append(dict(
            m=float(m), v0=float(v0), gcap=float(gcap), smax=float(smax),
            eta_min=float(emin), ref=float(Y[feas, iP].min()),
            span=float(np.ptp(Y[feas, iS])) if feas.sum() > 1 else 0.0,
            n_feas=int(feas.sum()),
            Y=np.round(Y[:, [iP, iS, iE, iR, iB]], 5).tolist()))
    json.dump(cache, open(cache_fp, "w"))
    return cache


def eval_gen_e6(model, ex, refs, c_lo, c_hi, x_lo, x_hi, ngen):
    """生成 40/题 → 实摔 → 新考规打分;返回总分 + 每题明细。"""
    jobs, ofs = [], 0
    for r in refs:
        cn = torch.tensor(norm(np.array([r["m"], r["v0"], r["gcap"], r["smax"],
                                         r["eta_min"]]), c_lo, c_hi),
                          dtype=torch.float32)
        Xg = x_lo + (x_hi - x_lo) * model.sample(cn, ngen).numpy()
        jobs += [(x, r["m"], r["v0"]) for x in Xg]
    rows = list(ex.map(_eval_one, jobs, chunksize=8))
    Yall = np.array([[np.nan if v is None else v for v in r] for r in rows])
    out = []
    for si, r in enumerate(refs):
        Y = Yall[si * ngen:(si + 1) * ngen]
        feas = feas_mask(Y, r["gcap"], r["smax"], r["eta_min"], r["v0"])
        gap = ((float(Y[feas, iP].min()) - r["ref"]) / r["ref"]) if feas.any() else 1.0
        cov = (float(np.ptp(Y[feas, iS])) / r["span"]
               if feas.sum() > 1 and r["span"] > 0 else np.nan)
        out.append(dict(m=r["m"], v0=r["v0"], gap=float(gap),
                        feas=float(feas.mean()), cov=cov))
    gaps = np.array([o["gap"] for o in out])
    covs = [o["cov"] for o in out if o["cov"] is not None and np.isfinite(o["cov"])]
    summ = dict(median_gap=float(np.median(gaps)), mean_gap=float(gaps.mean()),
                fail=int((gaps >= 1.0).sum()),
                feas_rate=float(np.mean([o["feas"] for o in out])),
                coverage=float(np.mean(covs)) if covs else 0.0)
    return summ, out


def bo9_e6(ex, r, lo, hi, rng):
    """BO-9 改造版:罚函数加 η 与回弹,考规与生成臂完全一致。"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
    from scipy.stats import norm as N
    d = len(lo)
    X = lo + (hi - lo) * lhs(4, d, rng)
    Y = np.array([[np.nan if v is None else v for v in y]
                  for y in ex.map(_eval_one, [(x, r["m"], r["v0"]) for x in X])])

    def pen(y):
        if not np.isfinite(y[iP]):
            return 1e4
        v = (y[iP] + 1e3 * max(0, y[iP] - r["gcap"]) / r["gcap"]
             + 1e3 * max(0, y[iS] - r["smax"]) / r["smax"]
             + 1e3 * max(0, r["eta_min"] - y[iE]) / r["eta_min"])
        h_eq = max(r["v0"] ** 2 / (2 * 9.81), 1e-9)
        if np.nan_to_num(y[iB], nan=1.0) > 0.5:          # 硬闸:足端离地
            v += 1e3
        if np.nan_to_num(y[iR], nan=np.inf) / h_eq > REB_CAP:  # 软闸:回能比
            v += 1e3
        return v

    for _ in range(5):
        t = np.array([pen(y) for y in Y])
        g = GaussianProcessRegressor(Ck(1.0) * RBF([0.3] * d) + WhiteKernel(1e-4),
                                     normalize_y=True).fit(norm(X, lo, hi), t)
        cand = norm(lo + (hi - lo) * lhs(400, d, rng), lo, hi)
        mu, sd = g.predict(cand, return_std=True)
        z = (t.min() - mu) / (sd + 1e-9)
        ei = (t.min() - mu) * N.cdf(z) + sd * N.pdf(z)
        xnew = lo + (hi - lo) * cand[np.argmax(ei)]
        ynew = [np.nan if v is None else v
                for v in list(ex.map(_eval_one, [(xnew, r["m"], r["v0"])]))[0]]
        X = np.vstack([X, xnew]); Y = np.vstack([Y, ynew])
    feas = feas_mask(Y, r["gcap"], r["smax"], r["eta_min"], r["v0"])
    return ((float(Y[feas, iP].min()) - r["ref"]) / r["ref"]) if feas.any() else 1.0


def strata_table(rows):
    tab = {}
    for o in rows:
        key = stratum(o["m"], o["v0"])
        tab.setdefault(key, []).append(o)
    out = {}
    for k in sorted(tab):
        g = np.array([o["gap"] for o in tab[k]])
        out[k] = dict(n=len(g), median_gap=float(np.median(g)),
                      fail=int((g >= 1.0).sum()),
                      feas=float(np.mean([o["feas"] for o in tab[k]])))
    return out


def main():
    global ETA_RANGE, REB_CAP
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/gen_data7/factory.jsonl")
    ap.add_argument("--data", default="outputs/gen_data7/gen_dataset.npz")
    ap.add_argument("--inc", default="outputs/gen_e5/pool_increments.jsonl")
    ap.add_argument("--out", default="outputs/gen_e6")
    ap.add_argument("--nseeds", type=int, default=3)
    ap.add_argument("--nval", type=int, default=24)
    ap.add_argument("--nref", type=int, default=300)
    ap.add_argument("--ngen-eval", type=int, default=40)
    ap.add_argument("--kscen", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--zdim", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--skip-bo", action="store_true")
    ap.add_argument("--eta-lo", type=float, default=ETA_RANGE[0])
    ap.add_argument("--eta-hi", type=float, default=ETA_RANGE[1])
    ap.add_argument("--reb-cap", type=float, default=REB_CAP,
                    help="软闸回能比上限(硬闸 n_bounce=0 恒开)")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    ETA_RANGE = (args.eta_lo, args.eta_hi)
    REB_CAP = args.reb_cap

    meta = json.load(open(os.path.join(os.path.dirname(args.factory), "factory_meta.json")))
    ds_meta = json.load(open(args.data.replace("gen_dataset.npz", "gen_dataset_meta.json")))
    lo, hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    test_cids = set(ds_meta["test_cids"])
    c_lo = np.array(ds_meta["c_lo"] + [ETA_RANGE[0]])
    c_hi = np.array(ds_meta["c_hi"] + [ETA_RANGE[1]])

    # —— 池(工厂 + 全部回灌)与 η 分布体检 ——
    pools = load_pools(args.factory, lo, hi)
    if os.path.exists(args.inc):
        for line in open(args.inc):
            inc = json.loads(line)
            Yi = np.array([[np.nan if v is None else v for v in row]
                           for row in inc["Y"]], float)
            pools[inc["cid"]].absorb(np.array(inc["X"], float), Yi, lo, hi)
    eta_all = np.concatenate([p.Y[np.isfinite(p.Y[:, iP]), iE] for p in pools.values()])
    print(f"[e6] pool={sum(len(p.X) for p in pools.values())}  "
          f"η 分布 P10/P50/P90 = {np.percentile(eta_all, [10, 50, 90]).round(3)}  "
          f"η_min 抽样区间 {ETA_RANGE}")

    C_tr, X_tr, n_empty = build_pairs_e6(pools, test_cids, args.kscen)
    print(f"[e6] 5维训练对 {len(C_tr)}  空可行场景跳过 {n_empty}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # —— 新冻结考卷(留出 cid × 新约束抽签,含 η_min) ——
        test_conds = []
        for cid in sorted(test_cids):
            p = pools[cid]
            crng = np.random.default_rng(99_000 + cid)
            for _ in range(args.kscen):
                test_conds.append([p.m, p.v0, crng.uniform(*GCAP_RANGE) * 9.81,
                                   crng.uniform(*SMAX_RANGE), crng.uniform(*ETA_RANGE)])
        test_refs = build_refs_e6(ex, test_conds, lo, hi, args.nref,
                                  os.path.join(args.out, "e6_refs.json"))
        print(f"[e6] 冻结考卷 {len(test_refs)}/{len(test_conds)} 题有解  "
              f"({time.time()-t0:.0f}s)")

        # —— 验证考卷(全新工况,选模用) ——
        vrng = np.random.default_rng(987_101)
        Cv = lhs(args.nval, 2, vrng)
        val_conds = [[c_lo[0] + (c_hi[0] - c_lo[0]) * a,
                      c_lo[1] + (c_hi[1] - c_lo[1]) * b,
                      vrng.uniform(*GCAP_RANGE) * 9.81,
                      vrng.uniform(*SMAX_RANGE), vrng.uniform(*ETA_RANGE)]
                     for a, b in Cv]
        val_refs = build_refs_e6(ex, val_conds, lo, hi, args.nref,
                                 os.path.join(args.out, "e6_val_refs.json"))
        print(f"[e6] 验证考卷 {len(val_refs)} 题  ({time.time()-t0:.0f}s)")

        # —— N 种子训练 + 无泄露选模 ——
        rows = []
        for sd in range(args.nseeds):
            model, _ = fit(norm(C_tr, c_lo, c_hi), norm(X_tr, lo, hi),
                           epochs=args.epochs, zdim=args.zdim, seed=sd, verbose=False)
            val, _ = eval_gen_e6(model, ex, val_refs, c_lo, c_hi, lo, hi, args.ngen_eval)
            tst, det = eval_gen_e6(model, ex, test_refs, c_lo, c_hi, lo, hi, args.ngen_eval)
            rows.append(dict(seed=sd, val_gap=val["median_gap"], test=tst, detail=det))
            torch.save(dict(state=model.state_dict(),
                            meta=dict(ds_meta, c_lo=c_lo.tolist(), c_hi=c_hi.tolist(),
                                      x_lo=lo.tolist(), x_hi=hi.tolist(),
                                      c_order=["m", "v0", "gcap_ms2", "smax_m", "eta_min"]),
                            xd=len(lo), zdim=args.zdim),
                       os.path.join(args.out, f"cvae_e6_s{sd}.pt"))
            print(f"  seed {sd}: val {val['median_gap']*100:5.1f}%  "
                  f"test {tst['median_gap']*100:5.1f}%  fail {tst['fail']}  "
                  f"({time.time()-t0:.0f}s)")
        pick = int(np.argmin([r["val_gap"] for r in rows]))
        strata = strata_table(rows[pick]["detail"])

        # —— BO-9-E6 对照(同卷同考规) ——
        bo = None
        if not args.skip_bo:
            gaps = []
            for si, r in enumerate(test_refs):
                gaps.append(bo9_e6(ex, r, lo, hi, np.random.default_rng(880_000 + si)))
            gb = np.array(gaps)
            bo = dict(median_gap=float(np.median(gb)), mean_gap=float(gb.mean()),
                      fail=int((gb >= 1.0).sum()),
                      strata=strata_table([dict(m=r["m"], v0=r["v0"], gap=g, feas=np.nan)
                                           for r, g in zip(test_refs, gaps)]))
            print(f"[e6] BO-9-E6: 中位 {bo['median_gap']*100:.1f}%  "
                  f"崩盘 {bo['fail']}  ({time.time()-t0:.0f}s)")

    summary = dict(
        n_test=len(test_refs), eta_range=ETA_RANGE, kscen=args.kscen,
        n_pairs=len(C_tr), picked_seed=pick,
        picked_test=rows[pick]["test"], strata=strata,
        seeds=[dict(seed=r["seed"], val_gap=r["val_gap"],
                    test_gap=r["test"]["median_gap"], test_fail=r["test"]["fail"])
               for r in rows],
        bo9_e6=bo)
    json.dump(summary, open(os.path.join(args.out, "e6_results.json"), "w"),
              indent=2, ensure_ascii=False)

    print("\n== E6 多指标考试(η 条件 + 回弹闸)==")
    p = rows[pick]["test"]
    print(f"  选中 seed {pick}: gap 中位 {p['median_gap']*100:.1f}%  "
          f"崩盘 {p['fail']}  可行率 {p['feas_rate']*100:.0f}%  覆盖 {p['coverage']*100:.0f}%")
    print("  —— 分层出分(体重 × 速度)——")
    for k, v in strata.items():
        print(f"    {k:<26} n={v['n']:>3}  gap {v['median_gap']*100:5.1f}%  "
              f"崩盘 {v['fail']}")
    if bo:
        print(f"  BO-9-E6 同卷: {bo['median_gap']*100:.1f}%(崩盘 {bo['fail']})")
    print(f"[e6] done → {args.out}")


if __name__ == "__main__":
    main()

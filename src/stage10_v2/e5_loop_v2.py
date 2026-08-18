"""E5-v2 专家迭代:与 v1 同框架,但跑在 v2 物理与质量条件先验上。

与 `stage7_generative/e5_loop.py` 的五点不同(其余机制原样保留:三道防线、
冻结考卷、真值裁判 Exudyn、无代理模型):

 1. 设计池存**无量纲 u**,物理设计随用随由 `bioprior.expand(u, m)` 还原。
    于是"同一个 u 在不同 m 下是不同的腿",条件生成第一次有东西可条件化。
 2. 条件向量 5 维 [log10 m, v0, log10 k_c, g_cap, s_max]。m 与 k_c 取对数,
    因为两者按对数均匀采样且异速先验本身是 log-log 线性的。
 3. **训练/考卷按 bid(束)切,不按 cid 切**。路径束把同一批设计放进 K 个 cid,
    按 cid 切 = 同一个设计同时出现在两侧,静默泄漏。
 4. 可行性含结构判据:s_max 作用在**腿行程**(已扣除地面下陷),
    另加 slenderness 与质量预算,全是不随 m 缩放的绝对量。
 5. 考卷条件取自测试束;某题若 nref 个新鲜设计里一个可行的都没有,**整题剔除**
   (没有参考前沿的题会污染 gap 统计)。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage10_v2/e5_loop_v2.py \
    --factory outputs/v2_data_bio/factory.jsonl --out outputs/v2_e5_bio \
    --rounds 20 --kgen 24 --workers 128
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P                                    # noqa: E402
from bioprior import BioPrior                             # noqa: E402
from factory_v2 import KEYS_V2, lhs, zeta_of_kc           # noqa: E402
from dataset_v2 import (GCAP_RANGE, SMAX_RANGE, iP, iL,   # noqa: E402
                        iSO, iMO, feasible_mask, pareto2, split_by_bid)
from train_cvae import CVAE, fit, norm                    # noqa: E402

DU = 7                       # u 维


# ---------------------------------------------------------------- 数据结构
@dataclass
class PoolV2:
    cid: int
    bid: int
    m: float
    v0: float
    kc: float
    zc: float
    U: np.ndarray
    Y: np.ndarray
    seen: set = field(default_factory=set)

    def key(self, u):
        return tuple(np.round(u, 3))

    def absorb(self, U_new, Y_new):
        fresh = [i for i, u in enumerate(U_new) if self.key(u) not in self.seen]
        for i in fresh:
            self.seen.add(self.key(U_new[i]))
        if fresh:
            self.U = np.vstack([self.U, U_new[fresh]])
            self.Y = np.vstack([self.Y, Y_new[fresh]])
        return len(fresh)


def _eval_one(a):
    x7, m, v0, kc, zc, npass = a
    r = P.eval_v2(tuple(x7), m, v0, kc=kc, zeta_c=zc, npass=npass)
    if r is None or r.get("fail"):
        return [None] * len(KEYS_V2)
    out = []
    for k in KEYS_V2:
        v = r.get(k, np.nan)
        v = float(bool(v)) if isinstance(v, bool) else float(v)
        out.append(v if np.isfinite(v) else None)
    return out


def simulate(ex, jobs):
    rows = list(ex.map(_eval_one, jobs, chunksize=8))
    return np.array([[np.nan if v is None else v for v in r] for r in rows], float)


def scatter(items):
    """[(tag, X7, m, v0, kc, zc), ...] → (jobs, slices),跨工况打平吃满所有核。"""
    jobs, slices, ofs = [], {}, 0
    for tag, X, m, v0, kc, zc in items:
        jobs += [(x, m, v0, kc, zc, 2) for x in X]
        slices[tag] = slice(ofs, ofs + len(X)); ofs += len(X)
    return jobs, slices


def load_pools(fp):
    pools = {}
    for line in open(fp):
        c = json.loads(line)
        U = np.array(c["U"], float)
        Y = np.array([[np.nan if v is None else v for v in r] for r in c["Y"]], float)
        p = PoolV2(c["cid"], c["bid"], c["m"], c["v0"], c["kc"], c["zeta_c"], U, Y)
        p.seen = {p.key(u) for u in U}
        pools[c["cid"]] = p
    return pools


def cvec(p, gcap, smax):
    return [np.log10(p.m), p.v0, np.log10(p.kc), gcap, smax]


# ---------------------------------------------------------------- 训练对
def build_pairs(pools, train_bids, kscen, ktop=8):
    C, U = [], []
    for p in pools.values():
        if p.bid not in train_bids:
            continue
        crng = np.random.default_rng(77_000 + p.cid)
        for _ in range(kscen):
            gcap = float(crng.uniform(*GCAP_RANGE) * 9.81)
            smax = float(crng.uniform(*SMAX_RANGE))
            fe = feasible_mask(p.Y, gcap, smax)
            if fe.sum() < 3:
                continue
            idx = np.where(fe)[0]
            front = idx[pareto2(p.Y[idx, iP], p.Y[idx, iL])][:ktop]
            c = cvec(p, gcap, smax)
            for j in front:
                C.append(c); U.append(p.U[j])
    return np.array(C, float).reshape(-1, 5), np.array(U, float).reshape(-1, DU)


# ---------------------------------------------------------------- 冻结考卷
def build_exam(ex, pools, test_bids, prior, nref, nexam, cache_fp, seed=5,
               min_feas_ref=5):
    """考卷 = 测试束里的工况 + 每题一组设计要求 + nref 个新鲜设计给出的参考前沿。

    无解的题(nref 个里没有一个可行)整题剔除——没有参考就没有 gap 可言。
    """
    if os.path.exists(cache_fp):
        return json.load(open(cache_fp))
    cands = sorted([p for p in pools.values() if p.bid in test_bids],
                   key=lambda p: p.cid)
    n_avail = len(cands)
    rng = np.random.default_rng(seed)
    if len(cands) > nexam:
        cands = [cands[i] for i in sorted(rng.choice(len(cands), nexam, replace=False))]
    items, spec = [], []
    for si, p in enumerate(cands):
        crng = np.random.default_rng(500_000 + p.cid)
        gcap = float(crng.uniform(*GCAP_RANGE) * 9.81)
        smax = float(crng.uniform(*SMAX_RANGE))
        Uq = lhs(nref, DU, np.random.default_rng(600_000 + p.cid))
        Xq = prior.expand(Uq, p.m)
        items.append((si, Xq, p.m, p.v0, p.kc, p.zc)); spec.append((p, gcap, smax))
    jobs, slices = scatter(items)
    Y = simulate(ex, jobs)
    exam, n_dead, n_thin = [], 0, 0
    for si, (p, gcap, smax) in enumerate(spec):
        Yq = Y[slices[si]]
        ok = feasible_mask(Yq, gcap, smax)
        if not ok.any():
            n_dead += 1; continue          # 无解题剔除:没有参考前沿就没有 gap 可言
        if ok.sum() < min_feas_ref:
            n_thin += 1; continue          # 参考太薄:min 只由 1-2 个样本决定,gap 噪声大
        exam.append(dict(cid=p.cid, bid=p.bid, m=p.m, v0=p.v0, kc=p.kc, zc=p.zc,
                         gcap=gcap, smax=smax,
                         ref=float(Yq[ok, iP].min()),
                         span=float(np.ptp(Yq[ok, iL])) if ok.sum() > 1 else 0.0,
                         n_feas=int(ok.sum())))
    json.dump(exam, open(cache_fp, "w"), indent=2)
    print(f"[e5v2] 考卷:测试块 {n_avail} 个,抽 {len(spec)} 题 → "
          f"无解剔除 {n_dead},参考过薄(<{min_feas_ref})剔除 {n_thin} → **{len(exam)} 题**")
    if len(spec) < nexam:
        print(f"[e5v2] 提示:测试块只有 {n_avail} 个,不足 --nexam {nexam}。"
              f"要更多题就加大 --nglobal 或提高测试束比例,不是这里的 bug")
    return exam


def eval_model(model, ex, exam, meta, prior, ngen):
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    items = []
    for si, r in enumerate(exam):
        cn = torch.tensor(norm(np.array(
            [np.log10(r["m"]), r["v0"], np.log10(r["kc"]), r["gcap"], r["smax"]]),
            c_lo, c_hi), dtype=torch.float32)
        Ug = model.sample(cn, ngen).numpy()
        items.append((si, prior.expand(Ug, r["m"]), r["m"], r["v0"], r["kc"], r["zc"]))
    jobs, slices = scatter(items)
    Y = simulate(ex, jobs)
    gaps, feas, cov = [], [], []
    for si, r in enumerate(exam):
        Yg = Y[slices[si]]
        ok = feasible_mask(Yg, r["gcap"], r["smax"])
        feas.append(float(ok.mean()))
        gaps.append((float(Yg[ok, iP].min()) - r["ref"]) / r["ref"] if ok.any() else 1.0)
        cov.append(float(np.ptp(Yg[ok, iL])) / r["span"]
                   if ok.sum() > 1 and r["span"] > 0 else np.nan)
    cv = [c for c in cov if np.isfinite(c)]
    return dict(median_gap=float(np.median(gaps)), mean_gap=float(np.mean(gaps)),
                feas_rate=float(np.mean(feas)),
                coverage=float(np.mean(cv)) if cv else 0.0,
                fail=int(sum(g >= 1.0 for g in gaps)), n=len(gaps))


# ---------------------------------------------------------------- 主循环
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/v2_data_bio/factory.jsonl")
    ap.add_argument("--out", default="outputs/v2_e5_bio")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--kgen", type=int, default=24)
    ap.add_argument("--eps", type=float, default=0.3, help="探索比例(新鲜 LHS)")
    ap.add_argument("--kscen", type=int, default=6)
    ap.add_argument("--ngen-eval", type=int, default=40)
    ap.add_argument("--nref", type=int, default=300)
    ap.add_argument("--nexam", type=int, default=76)
    ap.add_argument("--min-feas-ref", type=int, default=5,
                    help="参考前沿至少要有几个可行设计,少于此数整题剔除")
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--zdim", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=32,
                    help="训练对少于此数就报错退出(防止在贫瘠池上闷头训练)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fmeta = json.load(open(os.path.join(os.path.dirname(args.factory),
                                        "factory_meta.json")))
    prior = BioPrior(fmeta["arm"], sigma=fmeta["prior"]["sigma"],
                     u_max=fmeta["prior"]["u_max"])
    pools = load_pools(args.factory)
    tr_b, va_b, te_b = split_by_bid([p.bid for p in pools.values()],
                                    np.random.default_rng(3))
    c_lo = [np.log10(fmeta["m_range"][0]), fmeta["v0_range"][0],
            np.log10(fmeta["kc_range"][0]), GCAP_RANGE[0] * 9.81, SMAX_RANGE[0]]
    c_hi = [np.log10(fmeta["m_range"][1]), fmeta["v0_range"][1],
            np.log10(fmeta["kc_range"][1]), GCAP_RANGE[1] * 9.81, SMAX_RANGE[1]]
    gmeta = dict(c_order=["log10_m", "v0", "log10_kc", "gcap_ms2", "smax_m"],
                 c_lo=c_lo, c_hi=c_hi, arm=fmeta["arm"], prior=prior.describe(),
                 keys=KEYS_V2, u_dim=DU, split_by="bid",
                 bids=dict(tr=sorted(tr_b), va=sorted(va_b), te=sorted(te_b)))
    json.dump(gmeta, open(os.path.join(args.out, "model_meta.json"), "w"),
              indent=2, ensure_ascii=False)

    traj_fp = os.path.join(args.out, "trajectory.json")
    traj = json.load(open(traj_fp)) if os.path.exists(traj_fp) else []
    start = len(traj)
    print(f"[e5v2] 臂={fmeta['arm']}  块 {len(pools)}  束 训练{len(tr_b)}/"
          f"验证{len(va_b)}/测试{len(te_b)}  从第 {start} 轮续跑")

    inc_fp = os.path.join(args.out, "pool_increments.jsonl")
    if os.path.exists(inc_fp):
        n = 0
        for line in open(inc_fp):
            inc = json.loads(line)
            Yi = np.array([[np.nan if v is None else v for v in r]
                           for r in inc["Y"]], float)
            n += pools[inc["cid"]].absorb(np.array(inc["U"], float), Yi)
        print(f"[e5v2] 重放历史回灌 {n} 条")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        exam = build_exam(ex, pools, te_b, prior, args.nref, args.nexam,
                          os.path.join(args.out, "exam.json"),
                          min_feas_ref=args.min_feas_ref)
        if not exam:
            raise SystemExit("[e5v2] 考卷为空:测试束里没有任何有解的题,检查工况范围")
        print(f"[e5v2] 冻结考卷就绪  ({time.time()-t0:.0f}s)")

        for rd in range(start, args.rounds + 1):
            C_tr, U_tr = build_pairs(pools, tr_b, args.kscen)
            if len(C_tr) < args.min_pairs:
                raise SystemExit(f"[e5v2] 训练对只有 {len(C_tr)} 条,池子太贫瘠")
            model, _ = fit(norm(C_tr, np.array(c_lo), np.array(c_hi)), U_tr,
                           epochs=args.epochs, zdim=args.zdim,
                           seed=args.seed * 101 + rd, verbose=False)
            sc = eval_model(model, ex, exam, gmeta, prior, args.ngen_eval)
            npool = int(sum(len(p.U) for p in pools.values()))
            traj.append(dict(round=rd, pairs=len(C_tr), pool=npool, **sc))
            json.dump(traj, open(traj_fp, "w"), indent=2)
            torch.save(dict(state=model.state_dict(), meta=gmeta, xd=DU,
                            zdim=args.zdim, arm=fmeta["arm"]),
                       os.path.join(args.out, f"cvae_r{rd}.pt"))
            print(f"[e5v2] r{rd}: 训练对={len(C_tr)} 池={npool} | "
                  f"gap 中位 {sc['median_gap']*100:5.1f}% 均值 {sc['mean_gap']*100:5.1f}% "
                  f"可行 {sc['feas_rate']*100:3.0f}% 覆盖 {sc['coverage']*100:3.0f}% "
                  f"崩盘 {sc['fail']}  ({time.time()-t0:.0f}s)")
            if rd == args.rounds:
                break

            n_exp = int(round(args.kgen * args.eps))
            n_gen = max(2, args.kgen - n_exp)
            items = []
            for p in pools.values():
                if p.bid not in tr_b:
                    continue
                rng = np.random.default_rng(rd * 1_000_003 + p.cid)
                crng = np.random.default_rng(rd * 2_000_003 + p.cid)
                cand = [lhs(n_exp, DU, rng)]                       # 探索
                for _ in range(2):                                 # 两个约束场景下开采
                    c = np.array(cvec(p, crng.uniform(*GCAP_RANGE) * 9.81,
                                      crng.uniform(*SMAX_RANGE)))
                    cn = torch.tensor(norm(c, np.array(c_lo), np.array(c_hi)),
                                      dtype=torch.float32)
                    cand.append(model.sample(cn, n_gen // 2).numpy())
                U_new = np.clip(np.vstack(cand), 0.0, 1.0)
                items.append((p.cid, prior.expand(U_new, p.m), p.m, p.v0, p.kc, p.zc))
            jobs, slices = scatter(items)
            Y = simulate(ex, jobs)
            added = 0
            with open(inc_fp, "a") as f:
                for cid, X_new, m_, _, _, _ in items:
                    Yn = Y[slices[cid]]
                    Un = prior.contract(X_new, m_)
                    added += pools[cid].absorb(Un, Yn)
                    f.write(json.dumps(dict(
                        cid=cid, round=rd, U=np.round(Un, 5).tolist(),
                        Y=[[None if not np.isfinite(v) else float(v) for v in r]
                           for r in Yn])) + "\n")
            print(f"[e5v2] r{rd}→r{rd+1}: 回灌 {added} 条(队列 {len(jobs)})"
                  f"  ({time.time()-t0:.0f}s)")

    print("\n== E5-v2 轨迹 ==")
    for t in traj:
        print(f"  r{t['round']}: gap {t['median_gap']*100:5.1f}%  可行 {t['feas_rate']*100:3.0f}%"
              f"  崩盘 {t['fail']}  池 {t['pool']}")
    print(f"[e5v2] done → {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

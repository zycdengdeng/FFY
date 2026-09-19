# -*- coding: utf-8 -*-
"""E5-v3 专家迭代：轮足。与 e5_loop_v2 同框架（三道防线、冻结考卷、真值裁判、
无代理模型、按 bid 切分），四点不同：

 1. 先验 = WheelPrior（14 维 u：10 维腿 + 4 维轮），评价 = physics_v3.eval_v3
    （双镜像腿、俯仰自由、真刹车+滚阻、停车距离）；
 2. 条件 8 维 [log10 m, v0, log10 kc, gcap, smax, Fr, pitch_cap, log10 Lstop]，
    判据 feasible_mask_v3（+翻倒 +停车，无 slip）；
 3. **考卷不再冻结在 Fr=0**：v3 的主场是滑跑，全 Fr=0 的考卷测不到轮子。每题在
    构卷时抽签一个 Fr（20% 精确取零）并连同要求四元组一起存进 exam.json——
    题面跨轮次一致由缓存文件保证，比"硬编码 Fr=0"更一般且同样冻结；
 4. 算力预算完全不同：eval_v3 单次 ~14 s（v2.5 是 ~1.5 s）。默认参数按此缩：
    nref 200 × nexam 60 的考卷 ≈ 130 核时（只建一次，缓存）；每轮
    eval(40×~50 题) + 回灌(kgen 24 × 训练块) ≈ 55 核时。128 workers 下
    20 轮 ≈ 8–10 h —— 过夜任务，别排在白天。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage10_v2/e5_loop_v3.py \
    --factory outputs/v3_data_bio/factory.jsonl --out outputs/v3_e5_bio \
    --rounds 20 --workers 128
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
from physics_v3 import WheelPrior, eval_v3               # noqa: E402
from factory_v3 import KEYS_V3, FR_ZERO_FRAC_V3          # noqa: E402
from factory_v2 import lhs, zeta_of_kc                   # noqa: E402
import froude as FR                                      # noqa: E402
from dataset_v2 import split_by_bid, pareto2             # noqa: E402
from dataset_v3 import (feasible_mask_v3, draw_req, cvec_v3, c_bounds,  # noqa: E402
                        C_ORDER, iAR, iLS)
from train_cvae import CVAE, fit, norm                   # noqa: E402

DU = 14
FR_MAX = 5.0


@dataclass
class PoolV3:
    cid: int; bid: int; m: float; v0: float; kc: float; zc: float
    U: np.ndarray; Y: np.ndarray; Fr: np.ndarray
    seen: set = field(default_factory=set)

    def key(self, u):
        return tuple(np.round(u, 3))

    def absorb(self, U_new, Y_new, Fr_new):
        fresh = [i for i, u in enumerate(U_new) if self.key(u) not in self.seen]
        for i in fresh:
            self.seen.add(self.key(U_new[i]))
        if fresh:
            self.U = np.vstack([self.U, U_new[fresh]])
            self.Y = np.vstack([self.Y, Y_new[fresh]])
            self.Fr = np.concatenate([self.Fr, np.asarray(Fr_new, float)[fresh]])
        return len(fresh)


def _eval_one(a):
    x14, m, v0, kc, zc, v_x = a
    try:
        r = eval_v3(tuple(x14), m, v0, kc, zeta_c=zc, v_x=v_x)
        if r.get("fail") == "solver":
            r = eval_v3(tuple(x14), m, v0, kc, zeta_c=zc, v_x=v_x, h=1e-4)
    except Exception:
        return [None] * len(KEYS_V3)
    if r is None or r.get("fail"):
        return [None] * len(KEYS_V3)
    out = []
    for k in KEYS_V3:
        v = r.get(k, np.nan)
        v = float(bool(v)) if isinstance(v, bool) else float(v)
        out.append(v if np.isfinite(v) else None)
    return out


def simulate(ex, jobs):
    rows = list(ex.map(_eval_one, jobs, chunksize=2))
    return np.array([[np.nan if v is None else v for v in r] for r in rows], float)


def scatter(items):
    """[(tag, X14, m, v0, kc, zc, v_x[]), ...] → (jobs, slices)。"""
    jobs, slices, ofs = [], {}, 0
    for tag, X, m, v0, kc, zc, vx in items:
        jobs += [(x, m, v0, kc, zc, float(v)) for x, v in zip(X, vx)]
        slices[tag] = slice(ofs, ofs + len(X)); ofs += len(X)
    return jobs, slices


def load_pools(fp):
    pools = {}
    for line in open(fp):
        c = json.loads(line)
        U = np.array(c["U"], float)
        Y = np.array([[np.nan if v is None else v for v in r] for r in c["Y"]], float)
        p = PoolV3(c["cid"], c["bid"], c["m"], c["v0"], c["kc"], c["zeta_c"],
                   U, Y, np.asarray(c["Fr"], float))
        p.seen = {p.key(u) for u in U}
        pools[c["cid"]] = p
    return pools


def sample_fr(n, seed):
    return FR.sample_fr(n, np.random.default_rng(seed), fr_max=FR_MAX,
                        zero_frac=FR_ZERO_FRAC_V3)


# ---------------------------------------------------------------- 训练对
def build_pairs(pools, train_bids, kscen, ktop=8):
    C, U = [], []
    for p in pools.values():
        if p.bid not in train_bids:
            continue
        crng = np.random.default_rng(77_000 + p.cid)
        for _ in range(kscen):
            gcap, smax, pc, ls = draw_req(crng)
            fe = feasible_mask_v3(p.Y, gcap, smax, pc, ls)
            if fe.sum() < 3:
                continue
            idx = np.where(fe)[0]
            front = idx[pareto2(p.Y[idx, iAR], p.Y[idx, iLS])][:ktop]
            for j in front:
                C.append(cvec_v3(p.m, p.v0, p.kc, gcap, smax,
                                 float(p.Fr[j]), pc, ls))
                U.append(p.U[j])
    return (np.array(C, float).reshape(-1, 8),
            np.array(U, float).reshape(-1, DU))


# ---------------------------------------------------------------- 冻结考卷
def build_exam(ex, pools, test_bids, prior, nref, nexam, cache_fp,
               min_feas_ref=5):
    """每题：测试束工况 + 抽签的 (gcap,smax,pitch_cap,Lstop) + 抽签的 Fr
    （20% 取零；连同参考前沿一起存缓存 —— 题面冻结由缓存文件保证）。"""
    if os.path.exists(cache_fp):
        return json.load(open(cache_fp))
    cands = sorted([p for p in pools.values() if p.bid in test_bids],
                   key=lambda p: p.cid)
    rng = np.random.default_rng(5)
    if len(cands) > nexam:
        cands = [cands[i] for i in sorted(rng.choice(len(cands), nexam, replace=False))]
    items, spec = [], []
    for si, p in enumerate(cands):
        crng = np.random.default_rng(500_000 + p.cid)
        gcap, smax, pc, ls = draw_req(crng)
        fr = float(sample_fr(1, 550_000 + p.cid)[0])
        vx = float(FR.fr_to_vx(fr, p.m))
        Uq = lhs(nref, DU, np.random.default_rng(600_000 + p.cid))
        Xq = prior.expand(Uq, p.m)
        items.append((si, Xq, p.m, p.v0, p.kc, p.zc, np.full(len(Xq), vx)))
        spec.append((p, gcap, smax, pc, ls, fr))
    jobs, slices = scatter(items)
    Y = simulate(ex, jobs)
    exam, n_dead, n_thin = [], 0, 0
    for si, (p, gcap, smax, pc, ls, fr) in enumerate(spec):
        Yq = Y[slices[si]]
        ok = feasible_mask_v3(Yq, gcap, smax, pc, ls)
        if not ok.any():
            n_dead += 1; continue
        if ok.sum() < min_feas_ref:
            n_thin += 1; continue
        exam.append(dict(cid=p.cid, bid=p.bid, m=p.m, v0=p.v0, kc=p.kc, zc=p.zc,
                         gcap=gcap, smax=smax, pitch_cap=pc, lstop=ls, fr=fr,
                         ref=float(Yq[ok, iAR].min()),
                         span=float(np.ptp(Yq[ok, iLS])) if ok.sum() > 1 else 0.0,
                         n_feas=int(ok.sum())))
    json.dump(exam, open(cache_fp, "w"), indent=2)
    print(f"[e5v3] 考卷：抽 {len(spec)} 题 → 无解剔 {n_dead}、参考过薄剔 {n_thin} "
          f"→ **{len(exam)} 题**（Fr 分布含 {sum(1 for e in exam if e['fr']==0)} 题 Fr=0）")
    return exam


def eval_model(model, ex, exam, meta, prior, ngen):
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    items = []
    for si, r in enumerate(exam):
        c = cvec_v3(r["m"], r["v0"], r["kc"], r["gcap"], r["smax"],
                    r["fr"], r["pitch_cap"], r["lstop"])
        cn = torch.tensor(norm(np.array(c), c_lo, c_hi), dtype=torch.float32)
        Ug = np.clip(model.sample(cn, ngen).numpy(), 0.0, 1.0)
        vx = float(FR.fr_to_vx(r["fr"], r["m"]))
        items.append((si, prior.expand(Ug, r["m"]), r["m"], r["v0"], r["kc"], r["zc"],
                      np.full(len(Ug), vx)))
    jobs, slices = scatter(items)
    Y = simulate(ex, jobs)
    gaps, feas, cov = [], [], []
    for si, r in enumerate(exam):
        Yg = Y[slices[si]]
        ok = feasible_mask_v3(Yg, r["gcap"], r["smax"], r["pitch_cap"], r["lstop"])
        feas.append(float(ok.mean()))
        gaps.append((float(Yg[ok, iAR].min()) - r["ref"]) / r["ref"] if ok.any() else 1.0)
        cov.append(float(np.ptp(Yg[ok, iLS])) / r["span"]
                   if ok.sum() > 1 and r["span"] > 0 else np.nan)
    cv = [c for c in cov if np.isfinite(c)]
    return dict(median_gap=float(np.median(gaps)), mean_gap=float(np.mean(gaps)),
                feas_rate=float(np.mean(feas)),
                coverage=float(np.mean(cv)) if cv else 0.0,
                fail=int(sum(g >= 1.0 for g in gaps)), n=len(gaps))


# ---------------------------------------------------------------- 主循环
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/v3_data_bio/factory.jsonl")
    ap.add_argument("--out", default="outputs/v3_e5_bio")
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--kgen", type=int, default=24)
    ap.add_argument("--eps", type=float, default=0.3)
    ap.add_argument("--kscen", type=int, default=6)
    ap.add_argument("--ngen-eval", type=int, default=40)
    ap.add_argument("--nref", type=int, default=200)
    ap.add_argument("--nexam", type=int, default=60)
    ap.add_argument("--min-feas-ref", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--zdim", type=int, default=4, help="设计 14 维，z 比 v2.5 的 3 加一")
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--min-pairs", type=int, default=32)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    fmeta = json.load(open(os.path.join(os.path.dirname(args.factory),
                                        "factory_meta.json")))
    global DU, FR_MAX
    DU = int(fmeta.get("u_dim", 14))
    FR_MAX = float(fmeta.get("fr_max", 5.0))
    prior = WheelPrior(fmeta["arm"])
    print(f"[e5v3] 设计维 {DU}  条件 8 维  Fr∈[0,{FR_MAX}]  材料 {fmeta.get('mat')}")
    pools = load_pools(args.factory)
    tr_b, va_b, te_b = split_by_bid([p.bid for p in pools.values()],
                                    np.random.default_rng(3))
    c_lo, c_hi = c_bounds(fmeta, FR_MAX)
    gmeta = dict(c_order=C_ORDER, c_lo=c_lo, c_hi=c_hi, arm=fmeta["arm"],
                 prior=prior.describe(), keys=KEYS_V3, u_dim=DU, v3=True,
                 split_by="bid",
                 bids=dict(tr=sorted(tr_b), va=sorted(va_b), te=sorted(te_b)))
    json.dump(gmeta, open(os.path.join(args.out, "model_meta.json"), "w"),
              indent=2, ensure_ascii=False)

    traj_fp = os.path.join(args.out, "trajectory.json")
    traj = json.load(open(traj_fp)) if os.path.exists(traj_fp) else []
    start = len(traj)
    print(f"[e5v3] 块 {len(pools)}  束 {len(tr_b)}/{len(va_b)}/{len(te_b)}  从第 {start} 轮续跑")

    inc_fp = os.path.join(args.out, "pool_increments.jsonl")
    if os.path.exists(inc_fp):
        n = 0
        for line in open(inc_fp):
            inc = json.loads(line)
            Yi = np.array([[np.nan if v is None else v for v in r]
                           for r in inc["Y"]], float)
            n += pools[inc["cid"]].absorb(np.array(inc["U"], float), Yi, inc["Fr"])
        print(f"[e5v3] 重放历史回灌 {n} 条")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        exam = build_exam(ex, pools, te_b, prior, args.nref, args.nexam,
                          os.path.join(args.out, "exam.json"),
                          min_feas_ref=args.min_feas_ref)
        if not exam:
            raise SystemExit("[e5v3] 考卷为空——测试束里没有有解的题")
        print(f"[e5v3] 冻结考卷就绪  ({time.time()-t0:.0f}s)")

        for rd in range(start, args.rounds + 1):
            C_tr, U_tr = build_pairs(pools, tr_b, args.kscen)
            if len(C_tr) < args.min_pairs:
                raise SystemExit(f"[e5v3] 训练对只有 {len(C_tr)} 条,池子太贫瘠")
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
            print(f"[e5v3] r{rd}: 对={len(C_tr)} 池={npool} | gap中位 "
                  f"{sc['median_gap']*100:5.1f}% 可行 {sc['feas_rate']*100:3.0f}% "
                  f"覆盖 {sc['coverage']*100:3.0f}% 崩盘 {sc['fail']} "
                  f"({time.time()-t0:.0f}s)", flush=True)
            if rd == args.rounds:
                break

            n_exp = int(round(args.kgen * args.eps))
            n_gen = max(2, args.kgen - n_exp)
            items, newfr = [], {}
            for p in pools.values():
                if p.bid not in tr_b:
                    continue
                rng = np.random.default_rng(rd * 1_000_003 + p.cid)
                crng = np.random.default_rng(rd * 2_000_003 + p.cid)
                cand = [lhs(n_exp, DU, rng)]
                for _ in range(2):
                    gcap, smax, pc, ls = draw_req(crng)
                    c = np.array(cvec_v3(p.m, p.v0, p.kc, gcap, smax,
                                         float(crng.uniform(0, FR_MAX)), pc, ls))
                    cn = torch.tensor(norm(c, np.array(c_lo), np.array(c_hi)),
                                      dtype=torch.float32)
                    cand.append(model.sample(cn, n_gen // 2).numpy())
                U_new = np.clip(np.vstack(cand), 0.0, 1.0)
                frn = sample_fr(len(U_new), 8_000_000 + rd * 10_000 + p.cid)
                vxn = FR.fr_to_vx(frn, p.m)
                items.append((p.cid, prior.expand(U_new, p.m), p.m, p.v0, p.kc, p.zc,
                              vxn))
                newfr[p.cid] = frn
            jobs, slices = scatter(items)
            Y = simulate(ex, jobs)
            added = 0
            with open(inc_fp, "a") as f:
                for it in items:
                    cid, X_new, m_ = it[0], it[1], it[2]
                    Yn = Y[slices[cid]]
                    Un = prior.contract(X_new, m_)
                    frn = newfr[cid]
                    added += pools[cid].absorb(Un, Yn, frn)
                    f.write(json.dumps(dict(
                        cid=cid, round=rd, U=np.round(Un, 5).tolist(),
                        Fr=np.round(frn, 5).tolist(),
                        Y=[[None if not np.isfinite(v) else float(v) for v in r]
                           for r in Yn])) + "\n")
            print(f"[e5v3] r{rd}→r{rd+1}: 回灌 {added} 条(队列 {len(jobs)})"
                  f"  ({time.time()-t0:.0f}s)", flush=True)

    print("\n== E5-v3 轨迹 ==")
    for t in traj:
        print(f"  r{t['round']}: gap {t['median_gap']*100:5.1f}%  可行 "
              f"{t['feas_rate']*100:3.0f}%  崩盘 {t['fail']}  池 {t['pool']}")
    print(f"[e5v3] done → {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

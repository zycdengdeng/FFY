"""E19 · 并集标尺互考:四臂的生成设计上同一张卷、对同一个参考前沿。

治 E17 讨论确认的评测缺陷——各臂对照自家盒子的参考,盒子效应被约掉。
本实验:
  ① 抽一套公共考题(m, v0, kc, gcap, smax),与任何臂无关;
  ② 每题的参考前沿 = 四个先验盒各撒 nref 个 LHS 设计**并集**的最优可行峰值
     ("迄今任何盒子里找到的最好"),四臂共用这一把尺;
  ③ 每臂的 r20 模型在每题生成 ngen 个设计实摔,gap 对并集参考计算。
副产品:逐盒参考值单独记录 → 直接回答"哪个盒子里绝对更好的设计更多"
(这个问题连模型都不需要——纯盒子性质)。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage10_v2/e19_union_exam.py \
      --workers 128 --out outputs/v2_e19
仿真量: 考题72 × 4盒 × nref(120) × 2遍 ≈ 69k(参考,占大头)
       + 4臂 × 72 × ngen(40) × 2遍 ≈ 23k  → 合计 ~92k ≈ 20 分钟@128核
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

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P                          # noqa: E402
from bioprior import BioPrior                   # noqa: E402
from factory_v2 import zeta_of_kc, lhs, KEYS_V2 # noqa: E402
from dataset_v2 import feasible_mask, iP, iL, GCAP_RANGE, SMAX_RANGE  # noqa: E402
from e17_emergent_b import load as load_ck      # noqa: E402
from train_cvae import norm                     # noqa: E402

ARM_LIST = ["bio", "geo", "elastic", "none"]


def _eval_one(a):
    x7, m, v0, kc, zc = a
    r = P.eval_v2(tuple(x7), m, v0, kc=kc, zeta_c=zc, npass=2)
    if r is None or r.get("fail"):
        return [None] * len(KEYS_V2)
    out = []
    for k in KEYS_V2:
        v = r.get(k, np.nan)
        v = float(bool(v)) if isinstance(v, bool) else float(v)
        out.append(v if np.isfinite(v) else None)
    return out


def simulate(ex, jobs):
    rows = list(ex.map(_eval_one, jobs, chunksize=4))
    return np.array([[np.nan if v is None else v for v in r] for r in rows], float)


def make_questions(n, seed=99):
    rng = np.random.default_rng(seed)
    qs = []
    for _ in range(n):
        qs.append(dict(m=float(10 ** rng.uniform(0, np.log10(12))),
                       v0=float(rng.uniform(0.5, 2.0)),
                       kc=float(10 ** rng.uniform(np.log10(5e4), 6.0)),
                       gcap=float(rng.uniform(*GCAP_RANGE) * 9.81),
                       smax=float(rng.uniform(*SMAX_RANGE))))
    return qs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs", help="含 v2_e5_{arm}/cvae_r20.pt")
    ap.add_argument("--round", default="r20")
    ap.add_argument("--suffix", default="", help="存档目录后缀,如 _s1(多种子)")
    ap.add_argument("--refs", default=None,
                    help="已有 union_refs.json 的路径:换种子重考时复用参考,只跑生成侧")
    ap.add_argument("--nq", type=int, default=72)
    ap.add_argument("--nref", type=int, default=120, help="每盒每题参考采样数")
    ap.add_argument("--ngen", type=int, default=40)
    ap.add_argument("--min-feas-ref", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_e19")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    priors = {a: BioPrior(a) for a in ARM_LIST}
    qs = make_questions(args.nq)
    t0 = time.time()

    # ---------------------------------------------------------- ① 并集参考
    ref_fp = args.refs or os.path.join(args.out, "union_refs.json")
    if os.path.exists(ref_fp):
        exam = json.load(open(ref_fp))
        print(f"[e19] 参考已缓存:{len(exam)} 题")
    else:
        jobs, tags = [], []
        for qi, q in enumerate(qs):
            zc = zeta_of_kc(q["kc"])
            for arm in ARM_LIST:
                U = lhs(args.nref, 7, np.random.default_rng(7_000_000 + qi * 31 + hash(arm) % 97))
                X = priors[arm].expand(U, q["m"])
                jobs += [(x, q["m"], q["v0"], q["kc"], zc) for x in X]
                tags += [(qi, arm)] * args.nref
        print(f"[e19] 参考仿真 {len(jobs)} 次...")
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            Y = simulate(ex, jobs)
        print(f"[e19] 参考完成 ({time.time()-t0:.0f}s)")
        exam = []
        for qi, q in enumerate(qs):
            idx_all, per_box = [], {}
            for arm in ARM_LIST:
                ii = [k for k, t in enumerate(tags) if t == (qi, arm)]
                per_box[arm] = ii; idx_all += ii
            Ya = Y[idx_all]
            fe = feasible_mask(Ya, q["gcap"], q["smax"])
            if fe.sum() < args.min_feas_ref:
                continue                                   # 无解/过薄题剔除
            d = dict(q, ref=float(Ya[fe, iP].min()), n_feas=int(fe.sum()))
            for arm in ARM_LIST:
                Yb = Y[per_box[arm]]
                fb = feasible_mask(Yb, q["gcap"], q["smax"])
                d[f"ref_{arm}"] = float(Yb[fb, iP].min()) if fb.any() else None
                d[f"feas_{arm}"] = float(fb.mean())
            exam.append(d)
        json.dump(exam, open(ref_fp, "w"), indent=2)
        print(f"[e19] 有效考题 {len(exam)}/{args.nq}")

    # 盒子本身的较量(零模型):谁提供并集最优、平均可行率多少
    print(f"\n{'盒子':<9}{'提供并集最优(题数)':>18}{'平均可行率':>11}{'盒最优/并集最优 中位':>20}")
    for arm in ARM_LIST:
        wins = sum(1 for d in exam if d[f"ref_{arm}"] is not None
                   and abs(d[f"ref_{arm}"] - d["ref"]) < 1e-9)
        fe = np.mean([d[f"feas_{arm}"] for d in exam])
        rr = [d[f"ref_{arm}"] / d["ref"] for d in exam if d[f"ref_{arm}"] is not None]
        print(f"{arm:<9}{wins:>14}/{len(exam)}{fe*100:>10.1f}%"
              f"{np.median(rr):>20.3f}")

    # ---------------------------------------------------------- ② 四臂互考
    results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for arm in ARM_LIST:
            fp = os.path.join(args.dir, f"v2_e5_{arm}{args.suffix}", f"cvae_{args.round}.pt")
            if not os.path.exists(fp):
                print(f"[e19] 缺 {fp},跳过"); continue
            model, meta = load_ck(fp)
            pr = BioPrior(arm, sigma=meta["prior"]["sigma"], u_max=meta["prior"]["u_max"])
            c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
            jobs, slices, ofs = [], [], 0
            torch.manual_seed(0)
            for d in exam:
                c = np.array([np.log10(d["m"]), d["v0"], np.log10(d["kc"]),
                              d["gcap"], d["smax"]])
                cn = torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32)
                X = pr.expand(np.clip(model.sample(cn, args.ngen).numpy(), 0, 1), d["m"])
                zc = zeta_of_kc(d["kc"])
                jobs += [(x, d["m"], d["v0"], d["kc"], zc) for x in X]
                slices.append(slice(ofs, ofs + len(X))); ofs += len(X)
            Y = simulate(ex, jobs)
            gaps, feas = [], []
            for d, sl in zip(exam, slices):
                Yg = Y[sl]
                fe = feasible_mask(Yg, d["gcap"], d["smax"])
                feas.append(float(fe.mean()))
                gaps.append((float(Yg[fe, iP].min()) - d["ref"]) / d["ref"]
                            if fe.any() else 1.0)
            results[arm] = dict(median_gap=float(np.median(gaps)),
                                mean_gap=float(np.mean(gaps)),
                                feas_rate=float(np.mean(feas)),
                                fail=int(sum(g >= 1.0 for g in gaps)),
                                gaps=[round(g, 4) for g in gaps])
            print(f"[e19] {arm:<8} 并集 gap 中位 {results[arm]['median_gap']*100:6.1f}%  "
                  f"可行 {results[arm]['feas_rate']*100:3.0f}%  崩盘 {results[arm]['fail']}"
                  f"  ({time.time()-t0:.0f}s)")

    json.dump(dict(n_exam=len(exam), nref=args.nref, ngen=args.ngen,
                   results=results),
              open(os.path.join(args.out, "e19_results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[e19] → {args.out}")


if __name__ == "__main__":
    main()

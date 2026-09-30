"""E18 · 可行体重走廊:某个形态型沿某条标度律,从多轻到多重都能安全着陆?

与模型无关(不加载网络)——纯物理+先验几何,是 E17(生成侧)的独立互证。
预扫发现可行率沿 m **先升后降**(轻端被过载卡、重端被行程/结构卡),
故不用二分,改用**梯子扫描**:对每个 (臂 b, 形态 u),沿对数 m 网格逐级实测
"刚度/阻尼维撒 nprobe 个样本的可行占比 f(m)",得到:
    m_lo*(u,b) = 最轻的可行体重(过载边界)
    m_hi*(u,b) = 最重的可行体重(结构/行程边界)
    走廊宽度  = m_hi*/m_lo*(倍数)
四臂对比:哪条标度律的走廊更宽、更靠哪边——E17 说物理偏好 b≈0.2,
这里应看到相应臂的走廊不吃亏。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage10_v2/e18_mass_limit.py \
      --workers 128 --out outputs/v2_e18
仿真量: 4臂 × 9u × 14级 × nprobe(32) ≈ 16,000 次评价(每次2遍) ≈ 授时 12 分钟
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import physics_v2 as P                          # noqa: E402
from bioprior import BioPrior                   # noqa: E402
from factory_v2 import zeta_of_kc, lhs          # noqa: E402

KC0 = 1.0e5          # 中等地面(草地/压实土);预扫显示该场景动态范围最好
V00 = 1.2
GCAP_G = 10.0
SMAX = 0.024


def _probe_one(a):
    x7, m = a
    r = P.eval_v2(tuple(x7), m, V00, kc=KC0, zeta_c=zeta_of_kc(KC0), npass=2)
    ok, _ = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    return bool(ok)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bio,geo,elastic,none")
    ap.add_argument("--nu", type=int, default=9, help="u 网格点数(±2σ)")
    ap.add_argument("--nm", type=int, default=14, help="体重梯级数(mlo–mhi 对数)")
    ap.add_argument("--nprobe", type=int, default=32, help="每级撒多少刚度/阻尼样本")
    ap.add_argument("--mlo", type=float, default=0.5, help="体重梯子下端 (kg)")
    ap.add_argument("--mhi", type=float, default=60.0, help="体重梯子上端 (kg);60 处饱和时提到 120")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_e18")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    uls = np.array([0.0]) if args.nu == 1 else np.linspace(-2.0, 2.0, args.nu)
    ms = 10 ** np.linspace(np.log10(args.mlo), np.log10(args.mhi), args.nm)
    arms = args.arms.split(",")

    # 一次性打平全部任务,吃满核
    jobs, tags = [], []
    for arm in arms:
        prior = BioPrior(arm)
        for ui, uL in enumerate(uls):
            for mi, m in enumerate(ms):
                U = lhs(args.nprobe, 7, np.random.default_rng(
                    hash((arm, ui, mi)) % (2**31)))
                U[:, 0] = 0.5 * (uL / prior.u_max + 1.0)
                X = prior.expand(U, float(m))
                jobs += [(tuple(x), float(m)) for x in X]
                tags += [(arm, ui, mi)] * args.nprobe
    print(f"[e18] {len(arms)}臂 × {len(uls)}u × {len(ms)}级 × {args.nprobe}探针 "
          f"= {len(jobs)} 次评价(每次2遍仿真)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        flags = list(ex.map(_probe_one, jobs, chunksize=4))
    print(f"[e18] 完成 ({time.time()-t0:.0f}s)")

    F = {}
    for tg, fl in zip(tags, flags):
        F.setdefault(tg, []).append(fl)

    res = {}
    print(f"\n{'臂':<9}{'b':>7}{'走廊(u=0)':>16}{'宽度倍数':>9}{'走廊最宽的u':>11}")
    for arm in arms:
        prior = BioPrior(arm)
        rows = []
        for ui, uL in enumerate(uls):
            f = np.array([np.mean(F[(arm, ui, mi)]) for mi in range(len(ms))])
            feas_idx = np.where(f > 0)[0]
            if len(feas_idx) == 0:
                rows.append(dict(uL=float(uL), m_lo=None, m_hi=None, width=0.0,
                                 f=[round(v, 3) for v in f]))
                continue
            m_lo, m_hi = float(ms[feas_idx[0]]), float(ms[feas_idx[-1]])
            rows.append(dict(uL=float(uL), m_lo=m_lo, m_hi=m_hi,
                             width=m_hi / m_lo, f=[round(v, 3) for v in f]))
        res[arm] = dict(b=prior.b, rows=rows)
        mid = rows[len(uls) // 2]
        widths = [r["width"] for r in rows]
        best = rows[int(np.argmax(widths))]
        if mid["m_lo"]:
            print(f"{arm:<9}{prior.b:>7.3f}"
                  f"{mid['m_lo']:>7.1f}–{mid['m_hi']:<6.1f}kg{mid['width']:>8.1f}×"
                  f"{best['uL']:>+10.2f}")
        else:
            print(f"{arm:<9}{prior.b:>7.3f}  (u=0 全不可行)")

    json.dump(dict(kc=KC0, v0=V00, gcap_g=GCAP_G, smax=SMAX,
                   m_grid=[round(float(v), 3) for v in ms],
                   u_grid=[round(float(v), 3) for v in uls],
                   nprobe=args.nprobe, arms=res),
              open(os.path.join(args.out, "e18_corridor.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[e18] → {os.path.join(args.out, 'e18_corridor.json')}")


if __name__ == "__main__":
    main()

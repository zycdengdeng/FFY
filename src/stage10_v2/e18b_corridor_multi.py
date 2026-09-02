# -*- coding: utf-8 -*-
"""E18b · 可行体重走廊(多工况版)

相对 e18_mass_limit.py 的三处改动,都是为了让"升维到多重"这句话经得起追问:

  1) **多工况**。原版只在 kc=1e5 / v0=1.2 单点上扫,结论只能说"在草地上"。
     这里做 kc × v0 的因子网格,走廊从单点声明变成趋势。
  2) **分离模型失效与物理失效**。原版把 deep_sink(罚接触模型侵入超界,
     非物理)和真实的过载/结构失效一起记成"不可行"。重端质量大、侵入深,
     这两者混在一起会让"重端塌方"看起来比实际更硬。这里逐条判据分开统计。
  3) **可复现的随机种子**。原版用 hash((arm, ui, mi)),Python 对字符串的
     hash 每进程随机化(除非设 PYTHONHASHSEED),所以原版结果无法逐位复现。
     这里改用 zlib.crc32 的确定性哈希。

判据分解按 feasible_v2 的返回:每次评价可能同时违反多条,所以各条违反率
**不可叠加成 100%**——图里按多条独立曲线画,不画堆叠面积。

用法(A100,单条命令,勿换行):
  OMP_NUM_THREADS=1 python src/stage10_v2/e18b_corridor_multi.py --workers 128 --nprobe 48 --out outputs/v2_e18b

快速预览(约 1/4 成本):
  OMP_NUM_THREADS=1 python src/stage10_v2/e18b_corridor_multi.py --workers 128 --nprobe 24 --conds turf1.2,concrete1.2 --out outputs/v2_e18b_quick
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
import physics_v2 as P                          # noqa: E402
from bioprior import BioPrior                   # noqa: E402
from factory_v2 import zeta_of_kc, lhs          # noqa: E402

# 验收要求固定,只让工况变——否则四臂比较里混进第五个自由度
GCAP_G, SMAX = 10.0, 0.024

# 工况网格。kc 三档覆盖 KC_RANGE 全程(硬地 ↔ 湿沙),v0 两档。
CONDS = {
    "concrete1.2": dict(kc=1.0e6, v0=1.2, label="硬地 k_c=1e6 · v0=1.2"),
    "turf1.2":     dict(kc=1.0e5, v0=1.2, label="草地 k_c=1e5 · v0=1.2"),
    "wetsand1.2":  dict(kc=5.0e4, v0=1.2, label="湿沙 k_c=5e4 · v0=1.2"),
    "concrete2.0": dict(kc=1.0e6, v0=2.0, label="硬地 k_c=1e6 · v0=2.0"),
    "turf2.0":     dict(kc=1.0e5, v0=2.0, label="草地 k_c=1e5 · v0=2.0"),
    "wetsand2.0":  dict(kc=5.0e4, v0=2.0, label="湿沙 k_c=5e4 · v0=2.0"),
}
# 判据分桶。前四条是真实的工程失效,deep_sink 是模型失效,other 是求解器问题。
CRIT = ["gcap", "smax", "slenderness", "massbudget"]
MODELFAIL = ["deep_sink"]
OTHER = ["collapse", "solver", "nonfinite", "none"]


BASE_V21 = None   # 由 --v21 设定;None 时完全走老路径,既有结果可复现


def _base():
    if BASE_V21 is None:
        return None
    return {**P.SCEN_BIRD_X, "hip_damp_unified": True}


# --- 统计口径修正(2026-09-02):把三件事分开,不再塞进同一个分母 ---
#  ok          可行
#  infeasible  求解成功、模型有效,但违反 gcap/smax/slenderness/massbudget
#  invalid     模型失效(deep_sink:侵入超过足端球半径,罚接触模型不适用)
#  unsolved    数值失败(solver / nonfinite / none / collapse)
# 真实可行率 f_judged = ok / (ok + infeasible) —— 分母只含"能判"的样本。
# 旧口径 f_raw = ok / n 一并保留,便于与既有结果对照。
CLS_OK, CLS_INFEAS, CLS_INVALID, CLS_UNSOLVED = "ok", "infeasible", "invalid", "unsolved"


def classify(ok, why):
    """feasible_v2 对失败是提前返回,why 为单元素,所以三档可以干净分开。"""
    if ok:
        return CLS_OK
    w = set(why)
    if w & set(OTHER):
        return CLS_UNSOLVED
    if w & set(MODELFAIL):
        return CLS_INVALID
    return CLS_INFEAS


def _seed(*parts):
    """确定性种子:不依赖 Python 的字符串 hash 随机化。"""
    return zlib.crc32(("|".join(map(str, parts))).encode()) % (2 ** 31)


def _probe_one(a):
    """返回 (是否可行, 违反判据元组)。失败样本也带回原因,不吞掉。"""
    x7, m, v0, kc = a
    r = P.eval_v2(tuple(x7), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=_base())
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    return (bool(ok), tuple(why))


def run_cond(cname, arms, uls, ms, nprobe, workers, outdir):
    cd = CONDS[cname]
    kc, v0 = cd["kc"], cd["v0"]
    jobs, tags = [], []
    for arm in arms:
        prior = BioPrior(arm, v21=(BASE_V21 is not None))
        for ui, uL in enumerate(uls):
            for mi, m in enumerate(ms):
                U = lhs(nprobe, (9 if BASE_V21 is not None else 7), np.random.default_rng(_seed(cname, arm, ui, mi)))
                U[:, 0] = 0.5 * (uL / prior.u_max + 1.0)   # 锁定形态型,只撒刚度/阻尼
                for x in prior.expand(U, float(m)):
                    jobs.append((tuple(x), float(m), v0, kc))
                    tags.append((arm, ui, mi))

    print(f"[{cname}] {len(arms)}臂 × {len(uls)}u × {len(ms)}级 × {nprobe}探针 "
          f"= {len(jobs)} 次评价", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        out = list(ex.map(_probe_one, jobs, chunksize=4))
    print(f"[{cname}] 完成 ({time.time() - t0:.0f}s)", flush=True)

    # 汇总:每个 (臂, u, 体重) 格子记可行率 + 各判据违反率
    acc = {}
    for tg, (ok, why) in zip(tags, out):
        d = acc.setdefault(tg, dict(n=0, ok=0, infeasible=0, invalid=0, unsolved=0,
                                    **{c: 0 for c in CRIT + MODELFAIL + ["other"]}))
        d["n"] += 1
        d["ok"] += int(ok)
        d[classify(ok, why)] += 0 if ok else 1
        for w in why:
            if w in CRIT or w in MODELFAIL:
                d[w] += 1
            elif w in OTHER:
                d["other"] += 1

    res = {}
    for arm in arms:
        rows = []
        for ui, uL in enumerate(uls):
            g = [acc[(arm, ui, mi)] for mi in range(len(ms))]
            rows.append(dict(
                uL=float(uL),
                f=[round(d["ok"] / d["n"], 4) for d in g],                    # 旧口径,保留可比
                f_judged=[round(d["ok"] / max(d["ok"] + d["infeasible"], 1), 4) for d in g],
                r_invalid=[round(d["invalid"] / d["n"], 4) for d in g],
                r_unsolved=[round(d["unsolved"] / d["n"], 4) for d in g],
                **{c: [round(d[c] / d["n"], 4) for d in g]
                   for c in CRIT + MODELFAIL + ["other"]}))
        # 池合并 u:整个设计盒的可行率,统计误差比单条 u 曲线小 3 倍
        _sum = lambda mi, k: sum(acc[(arm, ui, mi)][k] for ui in range(len(uls)))
        pooled = [_sum(mi, "ok") / _sum(mi, "n") for mi in range(len(ms))]
        pooled_judged = [_sum(mi, "ok") / max(_sum(mi, "ok") + _sum(mi, "infeasible"), 1)
                         for mi in range(len(ms))]
        pooled_unsolved = [_sum(mi, "unsolved") / _sum(mi, "n") for mi in range(len(ms))]
        pooled_invalid = [_sum(mi, "invalid") / _sum(mi, "n") for mi in range(len(ms))]
        res[arm] = dict(b=BioPrior(arm, v21=(BASE_V21 is not None)).b, rows=rows,
                        pooled=[round(v, 4) for v in pooled],
                        pooled_judged=[round(v, 4) for v in pooled_judged],
                        pooled_unsolved=[round(v, 4) for v in pooled_unsolved],
                        pooled_invalid=[round(v, 4) for v in pooled_invalid])

    blob = dict(cond=cname, kc=kc, v0=v0, label=cd["label"],
                gcap_g=GCAP_G, smax=SMAX, nprobe=nprobe,
                m_grid=[round(float(v), 3) for v in ms],
                u_grid=[round(float(v), 3) for v in uls],
                crit=CRIT, modelfail=MODELFAIL, arms=res)
    fp = os.path.join(outdir, f"e18b_{cname}.json")
    json.dump(blob, open(fp, "w"), indent=1, ensure_ascii=False)
    print(f"[{cname}] → {fp}", flush=True)

    # 现场打一份人眼可读的池合并曲线,跑完不用等画图就能看
    print(f"  {'臂':<9}" + "".join(f"{m:>6.1f}" for m in ms))
    for arm in arms:
        print(f"  {arm:<9}" + "".join(f"{v*100:>6.0f}" for v in res[arm]["pooled"]))
    return blob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bio,geo,elastic,none")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--nu", type=int, default=9)
    ap.add_argument("--nm", type=int, default=16)
    ap.add_argument("--nprobe", type=int, default=48)
    ap.add_argument("--mlo", type=float, default=0.5)
    ap.add_argument("--mhi", type=float, default=120.0)
    ap.add_argument("--v21", action="store_true",
                    help="用 v2.1/v2.2 物理:9 维设计(含姿态)+ 髋阻尼统一式")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_e18b")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    uls = np.linspace(-2.0, 2.0, args.nu)
    ms = 10 ** np.linspace(np.log10(args.mlo), np.log10(args.mhi), args.nm)
    arms = args.arms.split(",")
    conds = [c for c in args.conds.split(",") if c in CONDS]
    assert conds, f"未知工况;可选:{list(CONDS)}"

    n1 = len(arms) * args.nu * args.nm * args.nprobe
    print(f"[e18b] {len(conds)} 工况 × {n1} = {len(conds) * n1} 次评价(每次 2 遍仿真)\n")
    for cname in conds:
        run_cond(cname, arms, uls, ms, args.nprobe, args.workers, args.out)
        print()
    print(f"[e18b] 全部完成 → {args.out}")


if __name__ == "__main__":
    main()

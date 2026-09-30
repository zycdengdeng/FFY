"""E11 · 公共标尺:三个几何盒子的并集参考前沿。

问题:消融三臂(bio/wide/shift)设计空间不同,若各用各的参考前沿,gap 不可比
(宽盒臂会被自己更强的参考"压分",窄盒臂被自己更弱的参考"抬分")。
解法:同一道考题下,从三个盒子各撒 nref 个 LHS 设计全部实摔,取**并集**的
最优可行峰值为参考——"迄今任何人找到的最好",三臂共用同一把尺子。
副产品:逐盒参考值单独记录,直接回答"宽盒里是否真的存在更好的设计"。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage7_generative/build_union_refs.py \
    --src-refs outputs/gen_e5/refs.json --out outputs/gen_abl/refs_union.json \
    --nref 300 --workers 128
产出: refs_union.json(e5_loop 可直接当 refs.json 用)+ 同目录 union_diag.json
仿真量: 题数 × 3 × nref(76 题 × 900 ≈ 6.8 万次)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage6_surrogate"))
import models as M                              # noqa: E402
from data_factory import _eval_one, KEYS        # noqa: E402
from e5_loop import lhs                         # noqa: E402

iP, iS = KEYS.index("peak_a"), KEYS.index("stroke")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-refs", default="outputs/gen_e5/refs.json",
                    help="取其中的考题条件 (m, v0, gcap, smax)")
    ap.add_argument("--out", default="outputs/gen_abl/refs_union.json")
    ap.add_argument("--nref", type=int, default=300, help="每盒每题采样数")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    src = json.load(open(args.src_refs))
    boxes = list(M.BOXES7.items())
    print(f"[union] 考题 {len(src)}  盒子 {[b for b, _ in boxes]}  "
          f"每盒每题 {args.nref} → 总仿真 {len(src) * len(boxes) * args.nref}")

    jobs, tags = [], []
    for si, r in enumerate(src):
        for bi, (bname, (blo, bhi)) in enumerate(boxes):
            lo, hi = np.array(blo), np.array(bhi)
            rng = np.random.default_rng(600_000 + si * 17 + bi)   # 每题每盒独立
            X = lo + (hi - lo) * lhs(args.nref, len(lo), rng)
            jobs += [(x, r["m"], r["v0"]) for x in X]
            tags += [(si, bname)] * args.nref

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_eval_one, jobs, chunksize=8))
    Y = np.array([[np.nan if v is None else v for v in r] for r in rows])
    print(f"[union] 仿真完成 ({time.time() - t0:.0f}s)")

    per = {}                       # (si, box) → 可行子集指标
    for k, (si, bname) in enumerate(tags):
        per.setdefault((si, bname), []).append(k)

    refs, diag = [], []
    for si, r in enumerate(src):
        idx_all = [k for b, _ in boxes for k in per[(si, b)]]
        Ya = Y[idx_all]
        ok = (np.isfinite(Ya[:, iP]) & (Ya[:, iP] <= r["gcap"])
              & (Ya[:, iS] <= r["smax"]))
        if not ok.any():
            continue
        d = dict(m=r["m"], v0=r["v0"], gcap=r["gcap"], smax=r["smax"],
                 ref=float(Ya[ok, iP].min()),
                 span=float(np.ptp(Ya[ok, iS])) if ok.sum() > 1 else 0.0)
        refs.append(d)
        row = dict(sc=si, m=r["m"], v0=r["v0"], ref_union=d["ref"],
                   ref_old=r.get("ref"), n_feas_union=int(ok.sum()))
        for bname, _ in boxes:
            Yb = Y[per[(si, bname)]]
            okb = (np.isfinite(Yb[:, iP]) & (Yb[:, iP] <= r["gcap"])
                   & (Yb[:, iS] <= r["smax"]))
            row[f"ref_{bname}"] = float(Yb[okb, iP].min()) if okb.any() else None
            row[f"feas_{bname}"] = float(okb.mean())
        diag.append(row)

    json.dump(refs, open(args.out, "w"))
    json.dump(diag, open(os.path.join(os.path.dirname(args.out) or ".",
                                      "union_diag.json"), "w"), indent=2)

    print(f"[union] 有解考题 {len(refs)}/{len(src)} → {args.out}")
    print("[union] 逐盒诊断(该盒是否含更好设计 / 可行率):")
    for bname, _ in boxes:
        wins = sum(1 for d in diag
                   if d[f"ref_{bname}"] is not None
                   and abs(d[f"ref_{bname}"] - d["ref_union"]) < 1e-9)
        fe = np.mean([d[f"feas_{bname}"] for d in diag])
        rr = [d[f"ref_{bname}"] / d["ref_union"] for d in diag
              if d[f"ref_{bname}"] is not None]
        print(f"  {bname:6s}: 提供并集最优 {wins}/{len(diag)} 题  "
              f"平均可行率 {fe * 100:.1f}%  该盒最优/并集最优 中位 "
              f"{np.median(rr) if rr else float('nan'):.3f}")


if __name__ == "__main__":
    main()

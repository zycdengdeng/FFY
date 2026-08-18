"""E11b · 消融的结构判定:三个几何盒子的"赢家"到底能不能造出来?

背景:E11 在并集标尺上显示工程朴素宽盒(体积 94× 生物盒)胜出。但该标尺
只看峰值过载与行程,**目标函数里没有质量与结构**——宽盒允许 L1≤250mm、r2≤4,
对一只 2kg 机体可给出总长近 2 m 的腿:峰值当然低,却造不出来。
诊断线索:宽盒优势随体重递减(m<4kg 时 1.32×,m≥8kg 时 1.06×),
正是"轻机体配超长腿"在刷分。

本实验:对与 build_union_refs 完全相同的采样(同种子),取每盒每题的最优可行设计,
过 E8 结构链条(关节力矩→薄壁管截面→衍生质量),比较三盒赢家的
**腿长、腿质量、质量占机体比**。判据:若宽盒赢家在质量上不可接受,
则"生物盒子编码了浅层目标看不见的结构合理性"成立;否则如实报负结果。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage8_struct/e11b_struct_check.py \
    --src-refs outputs/gen_e5/refs.json --out outputs/gen_e11b \
    --nref 300 --workers 128
成本: 题数 × 3 盒 × nref 次带传感器仿真(76×900 ≈ 6.8 万,约 25 分钟)
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
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage7_generative"))
import models as M                                   # noqa: E402
from e8_struct import _job, size_leg, MATERIALS      # noqa: E402
from e5_loop import lhs                              # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src-refs", default="outputs/gen_e5/refs.json")
    ap.add_argument("--out", default="outputs/gen_e11b")
    ap.add_argument("--nref", type=int, default=300)
    ap.add_argument("--material", default="cfnylon", choices=list(MATERIALS))
    ap.add_argument("--sf", type=float, default=2.0)
    ap.add_argument("--dmin", type=float, default=0.004)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    mat = MATERIALS[args.material]

    src = json.load(open(args.src_refs))
    boxes = list(M.BOXES7.items())
    print(f"[e11b] 考题 {len(src)} × 盒 {len(boxes)} × {args.nref} "
          f"= {len(src) * len(boxes) * args.nref} 次带传感器仿真")

    jobs, tags = [], []
    for si, r in enumerate(src):
        for bi, (bname, (blo, bhi)) in enumerate(boxes):
            lo, hi = np.array(blo), np.array(bhi)
            rng = np.random.default_rng(600_000 + si * 17 + bi)   # 与 build_union_refs 同种子
            X = lo + (hi - lo) * lhs(args.nref, len(lo), rng)
            jobs += [(x, r["m"], r["v0"]) for x in X]
            tags += [(si, bname, x) for x in X]

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        res = list(ex.map(_job, jobs, chunksize=8))
    print(f"[e11b] 仿真完成 ({time.time() - t0:.0f}s)")

    per = {}
    for (si, bname, x), mt in zip(tags, res):
        if mt is None:
            continue
        r = src[si]
        if mt["peak_a"] <= r["gcap"] and mt["stroke"] <= r["smax"]:
            key = (si, bname)
            if key not in per or mt["peak_a"] < per[key][1]["peak_a"]:
                per[key] = (x, mt)

    rows = []
    for si, r in enumerate(src):
        row = dict(sc=si, m=r["m"], v0=r["v0"])
        for bname, _ in boxes:
            e = per.get((si, bname))
            if e is None:
                row[bname] = None; continue
            x, mt = e
            segs, tot = size_leg(mt, mat, args.sf, 1, args.dmin)
            Lleg = sum(mt["seg_len"]) * 1e3
            row[bname] = dict(peak_g=mt["peak_a"] / 9.81, L_leg_mm=Lleg,
                              leg_mass_g=tot * 1e3, mass_frac_pct=100 * tot / r["m"],
                              L1_mm=float(x[0]), r2=float(x[1]), r3=float(x[2]),
                              D_max_mm=max(s["D_mm"] for s in segs))
        rows.append(row)

    json.dump(dict(material=args.material, sf=args.sf, nref=args.nref, rows=rows),
              open(os.path.join(args.out, "e11b_struct.json"), "w"),
              indent=2, ensure_ascii=False)

    print(f"\n{'盒':<7}{'有解':>5}{'峰值中位':>10}{'腿长中位':>11}{'腿质量中位':>12}"
          f"{'占机体中位':>12}{'最粗管中位':>12}")
    for bname, _ in boxes:
        v = [r[bname] for r in rows if r.get(bname)]
        if not v:
            print(f"{bname:<7}  无"); continue
        f = lambda k: np.median([e[k] for e in v])
        print(f"{bname:<7}{len(v):>5}{f('peak_g'):>9.2f}g{f('L_leg_mm'):>10.0f}mm"
              f"{f('leg_mass_g'):>11.0f}g{f('mass_frac_pct'):>11.1f}%"
              f"{f('D_max_mm'):>11.1f}mm")

    # 轻机体子集:宽盒优势最大处,也是"超长腿"最可疑处
    print("\n按体重分层的腿长/质量占比中位(暴露不合理设计):")
    for lab, sel in [("m<4kg", lambda r: r["m"] < 4), ("4-8kg", lambda r: 4 <= r["m"] < 8),
                     ("m>=8kg", lambda r: r["m"] >= 8)]:
        sub = [r for r in rows if sel(r)]
        parts = []
        for bname, _ in boxes:
            v = [r[bname] for r in sub if r.get(bname)]
            if v:
                parts.append(f"{bname} 腿{np.median([e['L_leg_mm'] for e in v]):.0f}mm/"
                             f"{np.median([e['mass_frac_pct'] for e in v]):.1f}%体重")
        print(f"  {lab:8s}(n={len(sub):>2}): " + "  |  ".join(parts))
    print(f"[e11b] done → {args.out}/e11b_struct.json")


if __name__ == "__main__":
    main()

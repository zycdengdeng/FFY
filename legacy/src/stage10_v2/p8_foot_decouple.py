# -*- coding: utf-8 -*-
"""P8 · 足端解绑验证:混杂消失了吗?

背景:r_foot = 0.20·L1 把"腿长"和"接地面积"绑成同一个变量。
四臂消融里 none 臂(b=0)因此自动获得一只不随质量长大的小脚,
在软地面上侵入超限 → deep_sink → 模型失效。这个劣势**与异速律假设无关**,
是我们建模抄近路带来的混杂。

v2.3 把 r_foot 改为由 (质量, 地面刚度) 派生(physics_v2.foot_radius 的 "bearing" 模式),
四臂在同一工况下拿到完全一样的脚。

本脚本回答两问:
  ① 混杂消失了吗 —— 四臂的 deep_sink 率是否收敛到同一条曲线?
  ② 代价是什么 —— bio 臂自己的可行率变了多少?(脚变了,接触几何也变了)

用法:  python src/stage10_v2/p8_foot_decouple.py --workers 128 --out outputs/v2_p8
"""
from __future__ import annotations
import argparse, json, os, sys, zlib
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import physics_v2 as P
from bioprior import BioPrior
from factory_v2 import zeta_of_kc, lhs

GCAP_G, SMAX = 10.0, 0.024
ARMS = ("bio", "geo", "elastic", "none")
TERR = {"硬地": 1.0e6, "草地": 1.0e5, "湿沙": 5.0e4}
CRIT = ["gcap", "smax", "slenderness", "massbudget"]
MODELFAIL = ["deep_sink"]
OTHER = ["collapse", "solver", "nonfinite", "none"]


def classify(ok, why):
    if ok:
        return "ok"
    w = set(why)
    if w & set(OTHER):
        return "unsolved"
    if w & set(MODELFAIL):
        return "invalid"
    return "infeasible"


def _ev(a):
    x9, m, v0, kc, mode = a
    base = {**P.SCEN_BIRD_X, "hip_damp_unified": True, "foot_mode": mode}
    r = P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=base)
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    return classify(bool(ok), list(why))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--masses", default="5,8,12,20,30")
    ap.add_argument("--v0", type=float, default=1.2)
    ap.add_argument("--nprobe", type=int, default=64)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_p8")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ms = [float(v) for v in a.masses.split(",")]

    print("=== 足端半径对照(mm):绑定 vs 解绑 ===")
    print(f"{'工况':<7}{'m':>5}" + "".join(f"{x:>9}" for x in ARMS) + f"{'解绑后(四臂同)':>16}")
    for tn, kc in TERR.items():
        for m in ms:
            row = f"{tn:<7}{m:>5.0f}"
            for arm in ARMS:
                L1 = BioPrior(arm, v21=True).l1_center(m)
                row += f"{P.foot_radius('leg', L1 / 1000, m, kc) * 1000:>9.1f}"
            row += f"{P.foot_radius('bearing', None, m, kc) * 1000:>14.1f}"
            print(row)

    jobs, tags = [], []
    for mode in ("leg", "bearing"):
        for arm in ARMS:
            pr = BioPrior(arm, v21=True)
            for tn, kc in TERR.items():
                for mi, m in enumerate(ms):
                    sd = zlib.crc32(f"{arm}|{tn}|{mi}".encode()) % (2 ** 31)
                    U = lhs(a.nprobe, 9, np.random.default_rng(sd))
                    for x in pr.expand(U, m):
                        jobs.append((tuple(x), m, a.v0, kc, mode))
                        tags.append((mode, arm, tn, mi))
    print(f"\n[p8] {len(jobs)} 次评价(2 模式 × 4 臂 × {len(TERR)} 地面 × "
          f"{len(ms)} 质量 × {a.nprobe} 探针)", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_ev, jobs, chunksize=8))

    acc = {}
    for tg, c in zip(tags, res):
        d = acc.setdefault(tg, dict(n=0, ok=0, infeasible=0, invalid=0, unsolved=0))
        d["n"] += 1; d[c] += 1

    for mode, title in (("leg", "① 绑定(r_foot = 0.20·L1,v2.2 口径)"),
                        ("bearing", "② 解绑(r_foot 由 m 与 k_c 派生,v2.3)")):
        print(f"\n===== {title} · deep_sink + 数值失败率 (%) =====")
        print(f"{'工况':<7}{'臂':<9}" + "".join(f"{m:>7.0f}" for m in ms))
        for tn in TERR:
            for arm in ARMS:
                v = [(acc[(mode, arm, tn, i)]["invalid"] + acc[(mode, arm, tn, i)]["unsolved"])
                     / acc[(mode, arm, tn, i)]["n"] * 100 for i in range(len(ms))]
                print(f"{tn:<7}{arm:<9}" + "".join(f"{x:>7.0f}" for x in v))

    print(f"\n===== 判据 ① 混杂是否消失:四臂废题率的**极差** (%) =====")
    print(f"{'工况':<7}{'m':>5}{'绑定':>9}{'解绑':>9}   ← 解绑后应趋近 0")
    for tn in TERR:
        for i, m in enumerate(ms):
            sp = {}
            for mode in ("leg", "bearing"):
                v = [(acc[(mode, arm, tn, i)]["invalid"] + acc[(mode, arm, tn, i)]["unsolved"])
                     / acc[(mode, arm, tn, i)]["n"] * 100 for arm in ARMS]
                sp[mode] = max(v) - min(v)
            print(f"{tn:<7}{m:>5.0f}{sp['leg']:>8.0f}%{sp['bearing']:>8.0f}%")

    print(f"\n===== 判据 ② 代价:bio 臂自己的可行率(判得了的样本) (%) =====")
    print(f"{'工况':<7}{'m':>5}{'绑定':>9}{'解绑':>9}{'Δ':>8}")
    for tn in TERR:
        for i, m in enumerate(ms):
            f = {}
            for mode in ("leg", "bearing"):
                d = acc[(mode, "bio", tn, i)]
                f[mode] = d["ok"] / max(d["ok"] + d["infeasible"], 1) * 100
            print(f"{tn:<7}{m:>5.0f}{f['leg']:>8.0f}%{f['bearing']:>8.0f}%"
                  f"{f['bearing']-f['leg']:>+7.0f}%")

    json.dump({f"{k[0]}|{k[1]}|{k[2]}|{k[3]}": v for k, v in acc.items()},
              open(os.path.join(a.out, "p8_foot.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n[p8] → {a.out}/p8_foot.json")


if __name__ == "__main__":
    main()

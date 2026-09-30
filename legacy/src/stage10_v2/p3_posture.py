# -*- coding: utf-8 -*-
"""P3 · 触地姿态对照:冻结姿态(踝120°/膝90°) vs Duong 视频中位(踝144°/膝133°)。

姿态来源:组内 Duong 12 段着水视频 water_contact 帧,踝 113–160°(中位 144°)、
膝 118–157°(中位 133°);跗跖-水平角 ψ 触水后稳在 ~50°(自家视频实测),保持不变。
腿越伸直可用行程越小 → 预期峰值升高,现取法偏乐观。

与 P1 的髋阻尼统一式做 2×2 —— 两者都将进入 v2.1 的同一次重跑,须看叠加效果。

用法:  python src/stage10_v2/p3_posture.py --designs outputs/v2_viz/designs.json --out outputs/v2_p3
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import physics_v2 as P
from factory_v2 import zeta_of_kc
from e18b_corridor_multi import GCAP_G, SMAX

TERR = {"硬地": 1.0e6, "草地": 1.0e5, "湿沙": 5.0e4}
V0 = 1.2
POST = {"冻结(踝120/膝90)": dict(thetaA=np.radians(120.0), thetaK=np.radians(90.0)),
        "Duong中位(踝144/膝133)": dict(thetaA=np.radians(144.0), thetaK=np.radians(133.0))}


def _ev(a):
    x7, m, kc, post, unified = a
    base = {**P.SCEN_BIRD_X, **POST[post], "hip_damp_unified": bool(unified)}
    r = P.eval_v2(tuple(x7), m, V0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=base)
    if r is None or r.get("fail"):
        return dict(fail=(r or {}).get("fail", "?"))
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    return dict(g=r["peak_a"] / 9.81, st=r["leg_stroke_mm"], sk=r["sink_mm"],
                D=r["D_mm"][2], ok=bool(ok), why=list(why))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="outputs/v2_viz/designs.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_p3")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = json.load(open(a.designs))

    jobs, tags = [], []
    for d in D:
        for tn, kc in TERR.items():
            for pn in POST:
                for u in (False, True):
                    jobs.append((d["x7"], d["m"], kc, pn, u))
                    tags.append((d["m"], tn, pn, u))
    print(f"[p3] {len(jobs)} 次评价", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_ev, jobs, chunksize=1))
    out = [dict(m=t[0], terr=t[1], posture=t[2], unified=t[3], **r)
           for t, r in zip(tags, res)]
    json.dump(out, open(os.path.join(a.out, "p3_posture.json"), "w"),
              indent=1, ensure_ascii=False)

    G = lambda m, tn, pn, u: next(o for o in out if o["m"] == m and o["terr"] == tn
                                  and o["posture"] == pn and o["unified"] == u)
    P0, P1 = list(POST)
    print("\n=== 姿态对照(髋阻尼=现状特例式) ===")
    print(f"{'m':>5} {'地面':<5}{'冻结 g':>8}{'Duong g':>9}{'Δ峰值':>8}{'冻结行程':>9}{'Duong行程':>10}   判定变化")
    for d in D:
        for tn in TERR:
            p, q = G(d["m"], tn, P0, False), G(d["m"], tn, P1, False)
            if "g" not in p or "g" not in q:
                print(f"{d['m']:>5.0f} {tn:<5}  失败 {p.get('fail','-')}/{q.get('fail','-')}"); continue
            v = ("✓" if p["ok"] else "✗") + "→" + ("✓" if q["ok"] else "✗" + ",".join(q["why"]))
            print(f"{d['m']:>5.0f} {tn:<5}{p['g']:>8.2f}{q['g']:>9.2f}"
                  f"{(q['g']-p['g'])/p['g']*100:>7.1f}%{p['st']:>9.1f}{q['st']:>10.1f}   {v}")
    print("\n=== 2×2 全景(12 kg,峰值 g / 判定) ===")
    print(f"{'地面':<5}{'冻结+特例':>12}{'冻结+统一':>12}{'Duong+特例':>12}{'Duong+统一':>12}")
    for tn in TERR:
        row = f"{tn:<5}"
        for pn in (P0, P1):
            for u in (False, True):
                o = G(12.0, tn, pn, u)
                row += (f"{o['g']:>9.2f}{'✓' if o['ok'] else '✗':>3}"
                        if "g" in o else f"{'fail':>12}")
        print(row)
    print(f"\n[p3] → {a.out}/p3_posture.json")


if __name__ == "__main__":
    main()

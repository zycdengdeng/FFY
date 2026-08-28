# -*- coding: utf-8 -*-
"""P1 · 髋阻尼特例式对照。

v1 遗留:c_hip = (τ/0.03)·0.2·k_hip —— 等效松弛时间为踝/膝的 6.67 倍,
换算损耗因子 tan δ ≈ 16–25,超过任何已知材料。两问:

  A. 偏置:统一成 c = τ·k 后,现有 AI 设计的峰值/可行性变多少?
     (若特例式在压峰值,则现有结果整体偏乐观)
  B. 死通道:κ髋 在六工况间几乎不动(1.10×)。固定设计扫 κ髋 ∈ [6,32],
     两种阻尼式下看峰值的响应幅度 —— 若特例式下平、统一式下变,
     则"网络不调髋"是被阻尼式压死的,修好后髋是一个可用的主动控制通道。

用法(A100,单行):
  OMP_NUM_THREADS=1 python src/stage10_v2/p1_hipdamp.py --workers 64 --out outputs/v2_p1
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


def _ev(a):
    x7, m, kc, unified = a
    base = {**P.SCEN_BIRD_X, "hip_damp_unified": bool(unified)}
    r = P.eval_v2(tuple(x7), m, V0, kc=kc, zeta_c=zeta_of_kc(kc),
                  npass=2, base=base)
    if r is None or r.get("fail"):
        return dict(fail=(r or {}).get("fail", "?"))
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    return dict(g=r["peak_a"] / 9.81, st=r["leg_stroke_mm"],
                Mh=r["M_hip"], D=r["D_mm"][2], ok=bool(ok), why=list(why))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--designs", default="outputs/v2_viz/designs.json")
    ap.add_argument("--nk", type=int, default=9, help="κ髋 扫描级数")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_p1")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    D = json.load(open(a.designs))

    jobs, tags = [], []
    # ---- A. 偏置:5 个 AI 设计 × 3 地面 × 2 阻尼式
    for d in D:
        for tn, kc in TERR.items():
            for u in (False, True):
                jobs.append((d["x7"], d["m"], kc, u)); tags.append(("A", d["m"], tn, u, None))
    # ---- B. 死通道:AI 12kg 设计,扫 κ髋,草地+硬地 × 2 阻尼式
    x12 = list(D[-1]["x7"]); ks = np.linspace(6.0, 32.0, a.nk)
    for tn in ("草地", "硬地"):
        for u in (False, True):
            for kh in ks:
                x = list(x12); x[5] = float(kh)
                jobs.append((x, 12.0, TERR[tn], u)); tags.append(("B", 12.0, tn, u, float(kh)))

    print(f"[p1] {len(jobs)} 次评价", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        res = list(ex.map(_ev, jobs, chunksize=1))
    out = [dict(part=t[0], m=t[1], terr=t[2], unified=t[3], kap_hip=t[4], **r)
           for t, r in zip(tags, res)]
    json.dump(out, open(os.path.join(a.out, "p1_hipdamp.json"), "w"),
              indent=1, ensure_ascii=False)

    # ---------- 现场汇总 ----------
    G = lambda part, m, tn, u, kh=None: next(
        (o for o in out if o["part"] == part and o["m"] == m and o["terr"] == tn
         and o["unified"] == u and (kh is None or o.get("kap_hip") == kh)), None)
    print("\n=== A · 偏置:特例式(现状) vs 统一式 c=τ·k ===")
    print(f"{'m':>5} {'地面':<5}{'现状 g':>9}{'统一 g':>9}{'Δ':>8}{'现状髋M':>9}{'统一髋M':>9}   判定变化")
    for d in D:
        for tn in TERR:
            p0, p1 = G("A", d["m"], tn, False), G("A", d["m"], tn, True)
            if not p0 or "g" not in p0 or "g" not in p1:
                print(f"{d['m']:>5.0f} {tn:<5}  失败 {p0.get('fail','?')}/{p1.get('fail','?')}"); continue
            v = ("✓" if p0["ok"] else "✗") + "→" + ("✓" if p1["ok"] else "✗")
            print(f"{d['m']:>5.0f} {tn:<5}{p0['g']:>9.2f}{p1['g']:>9.2f}"
                  f"{(p1['g']-p0['g'])/p0['g']*100:>7.1f}%{p0['Mh']:>9.1f}{p1['Mh']:>9.1f}   {v}")
    print("\n=== B · κ髋 死通道检验(AI-12kg,扫 κ髋 6→32) ===")
    ks = sorted({o["kap_hip"] for o in out if o["part"] == "B"})
    for tn in ("草地", "硬地"):
        for u, lab in ((False, "现状特例式"), (True, "统一 c=τ·k")):
            gs = [G("B", 12.0, tn, u, k) for k in ks]
            gg = [q["g"] for q in gs if q and "g" in q]
            row = "  ".join(f"{q['g']:.2f}" if q and "g" in q else " ×  " for q in gs)
            span = (max(gg) / min(gg) - 1) * 100 if len(gg) > 1 else float("nan")
            print(f"{tn} {lab:<9} 峰值g: {row}   响应幅度 {span:.1f}%")
    print(f"\n[p1] → {a.out}/p1_hipdamp.json")


if __name__ == "__main__":
    main()

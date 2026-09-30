# -*- coding: utf-8 -*-
"""跑三条对比时程并存 npz。硬地 12kg,刚度全部用 AI 的,只变几何。"""
import sys, json, numpy as np
sys.path.insert(0, '/tmp/pipeline_code/src/stage10_v2')
import physics_v2 as P
from factory_v2 import zeta_of_kc

M, V0, KC = 12.0, 1.2, 1.0e6
GCAP, SMAX = 10 * 9.81, 0.024
AI = json.load(open("/tmp/designs.json"))[-1]["x7"]
KAP = list(AI[3:7])                      # 同一套 κ踝/κ膝/κ髋/ζ

CASES = [
    ("ai",    "AI 生成设计",            list(AI)),
    ("swan",  "真鸟·疣鼻天鹅几何",       [93.6, 1.764, 0.951] + KAP),
    ("duck",  "朴素仿生·绿头鸭腿原样",   [39.75, 1.80, 1.06] + KAP),
]

out = {}
for key, label, x7 in CASES:
    r = P.eval_v2(tuple(x7), M, V0, kc=KC, zeta_c=zeta_of_kc(KC),
                  npass=2, keep_history=True)
    assert not r.get("fail"), (key, r)
    ok, why = P.feasible_v2(r, GCAP, SMAX)
    h = r.pop("hist")
    meta = dict(key=key, label=label, x7=x7,
                peak_g=r["peak_a"] / 9.81, leg_stroke_mm=r["leg_stroke_mm"],
                sink_mm=r["sink_mm"], D_mm=r["D_mm"], leg_mass_g=r["leg_mass_kg"] * 1e3,
                mass_frac=r["mass_frac"], F_peak=r["F_peak"], ok=bool(ok), why=list(why),
                L_mm=[x7[0], x7[0] * x7[1], x7[0] * x7[2]])
    out[key] = dict(meta=meta, hist=h)
    print(f"{label:<22} {meta['peak_g']:6.2f} g  行程 {meta['leg_stroke_mm']:5.1f}mm  "
          f"{'可行' if ok else '✗ ' + ','.join(why)}", flush=True)

np.savez_compressed("/tmp/anim/hist.npz",
                    meta=json.dumps({k: v["meta"] for k, v in out.items()}),
                    **{f"{k}__{n}": np.asarray(a)
                       for k, v in out.items() for n, a in v["hist"].items()})
print("\nsaved /tmp/anim/hist.npz",
      __import__("os").path.getsize("/tmp/anim/hist.npz") // 1024, "KB")

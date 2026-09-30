# -*- coding: utf-8 -*-
"""v3_grid · M3 相图重跑（正确刹车符号）+ M4 工厂基线标定

⚠ 为什么重跑：M1–M3 的 wheel_probe 刹车用户函数写了 −tanh(ω)，
   但 exudyn 的 springTorqueUserFunction 返回值施加时取反（2026-09-19 台架
   实测：+c·ω 才减速，误差 0），所以那套「刹车」其实是驱动力矩，轮子从不真刹车
   ——M3「轮足滚 8 m 停不下」正是这个 bug 的指纹。physics_v3.eval_v3 已修正符号，
   并补上滚动阻力与停车距离。本脚本在正确物理下重画相图、定 v3 盒子。

矩阵：m{4.7,12} × wb{0.7,0.8,0.9} × τ{1,2,4,6} × ω_th{10,40} × r_w{0.03}
       × Fr{0,0.5,1,2,3,5}   （+ r_w{0.025,0.045} 仅在 wb=0.9,τ=2 抽查敏感度）
基线臂：v2.5 al7075 裸足单腿（physics_v2.eval_v2，同工况）——三方对比用。

用法（单行，A100）：
  cd /mnt/zihanw/FFY && python v3_grid_probe.py --workers 32 2>&1 | tee outputs/v3_grid/run.log
"""
from __future__ import annotations
import os, sys, json, time, argparse, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, ".")
sys.path.insert(0, "src/stage10_v2")

CHOSEN_X = [113.1004, 2.0272, 1.2446, 1.2371, 4.6829, 16.2987,
            0.0199, 125.6335, 129.8789, 29.1107]
V0, KC = 1.3718, 8.681e5

FR_LIST = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]
WB_LIST = [0.7, 0.8, 0.9]
TAU_LIST = [1.0, 2.0, 4.0, 6.0]
OMTH_LIST = [10.0, 40.0]
M_LIST = [4.7305, 12.0]


def _fr_vx(fr, m):
    import froude as FR
    return float(FR.fr_to_vx(fr, m))


def _one(job):
    import physics_v3 as P3
    import physics_v2 as P2
    t0 = time.time()
    vx = _fr_vx(job["fr"], job["m"])
    try:
        if job["kind"] == "bare":               # v2.5 裸足单腿基线（锁俯仰，v2.5 口径）
            r = P2.eval_v2(tuple(CHOSEN_X), job["m"], V0, kc=KC, base=None,
                           v_x=vx, planar=True, mat="al7075", npass=1)
            out = dict(a_res_g=r.get("a_res", r["peak_a"]) / 9.81,
                       peak_g=r["peak_a"] / 9.81,
                       leg_stroke_mm=r.get("leg_stroke", 0) * 1e3,
                       mass_frac=r.get("mass_frac"), fail=r.get("fail"))
        else:
            x = list(CHOSEN_X) + [job["wb"], job["tau"], job["r_w"], job["om_th"]]
            r = P3.eval_v3(x, job["m"], V0, KC, v_x=vx)
            if r.get("fail") == "solver":
                r = P3.eval_v3(x, job["m"], V0, KC, v_x=vx, h=1e-4)
                r["retried"] = True
            out = {k: r.get(k) for k in
                   ("fail", "a_res", "peak_a", "pitch_max_deg", "roll_dist_m",
                    "stop_dist_m", "stopped", "stop_extrap", "vx_end",
                    "leg_stroke", "mass_frac", "wheel_spin_max", "latch_f", "latch_r")}
    except Exception as e:
        out = dict(fail="except", err=str(e)[:100])
    out.update(job); out["wall_s"] = round(time.time() - t0, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="outputs/v3_grid")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frs, wbs, taus, omths, ms = FR_LIST, WB_LIST, TAU_LIST, OMTH_LIST, M_LIST
    if a.quick:
        frs, wbs, taus, omths, ms = [0.0, 1.0, 2.0], [0.9], [2.0], [40.0], [4.7305]

    jobs = []
    for m in ms:
        for fr in frs:
            jobs.append(dict(kind="bare", m=m, fr=fr, wb=None, tau=None,
                             r_w=None, om_th=None))
            for wb, tau, omth in itertools.product(wbs, taus, omths):
                jobs.append(dict(kind="v3", m=m, fr=fr, wb=wb, tau=tau,
                                 r_w=0.03, om_th=omth))
            if not a.quick:                     # r_w 敏感度抽查
                for rw in (0.025, 0.045):
                    jobs.append(dict(kind="v3", m=m, fr=fr, wb=0.9, tau=2.0,
                                     r_w=rw, om_th=40.0))
    print(f"[v3_grid] jobs={len(jobs)} workers={a.workers}", flush=True)

    t0 = time.time(); res = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=2)):
            res.append(r)
            if (i + 1) % 40 == 0:
                ok = sum(1 for x in res if x["kind"] == "v3" and not x.get("fail"))
                print(f"  {i+1}/{len(jobs)}  v3_ok={ok}  {time.time()-t0:.0f}s", flush=True)
    json.dump(res, open(os.path.join(a.out, "v3_grid.json"), "w"), ensure_ascii=False)
    print(f"[v3_grid] done {time.time()-t0:.0f}s → {a.out}/v3_grid.json", flush=True)

    # ---- 摘要：翻倒/塌陷相图 + 停车距离 ----
    v3 = [r for r in res if r["kind"] == "v3"]
    from collections import Counter
    print("\n失败分布:", Counter(r.get("fail") or "ok" for r in v3))
    print("\n== 存活相图（wb×Fr，best-of τ×ω_th；值=停车距离 m / × = 全灭）m=4.73 ==")
    print("        Fr:  " + " ".join(f"{fr:>6}" for fr in frs))
    for wb in wbs:
        row = []
        for fr in frs:
            cand = [r for r in v3 if r["m"] == 4.7305 and r["wb"] == wb
                    and r["fr"] == fr and not r.get("fail")
                    and np.isfinite(r.get("stop_dist_m", np.inf))]
            row.append(f"{min(c['stop_dist_m'] for c in cand):>6.2f}" if cand else "     ×")
        print(f"  wb={wb:.1f}:  " + " ".join(row))
    # 裸足基线
    print("\n== 裸足基线 a_res_g（v2.5 al 单腿，锁俯仰）==")
    for m in ms:
        row = [f"Fr{r['fr']}:{r.get('a_res_g',float('nan')):.2f}"
               for r in sorted([x for x in res if x["kind"] == "bare" and x["m"] == m],
                               key=lambda r: r["fr"])]
        print(f"  m={m:.1f}: " + "  ".join(row))
    # 可行样本停车距离
    good = [r for r in v3 if not r.get("fail") and r.get("stopped")]
    if good:
        print(f"\n停住样本 {len(good)}/{len(v3)}；停车距离中位 "
              f"{np.median([r['stop_dist_m'] for r in good]):.2f} m")
        for fr in frs:
            g = [r for r in good if r["fr"] == fr]
            if g:
                print(f"  Fr={fr}: {len(g)} 停，停车 "
                      f"{np.median([r['stop_dist_m'] for r in g]):.2f} m，"
                      f"俯仰中位 {np.median([r['pitch_max_deg'] for r in g]):.1f}°")


if __name__ == "__main__":
    main()

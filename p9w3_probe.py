# -*- coding: utf-8 -*-
"""P9w-3 · 轮足 v3 全标定（M3）—— A100 版

前置（本地已验，A100 复验）：
  Phase 0 = M1 回归：单腿锁俯仰 4 模式 × 3 Fr（quadPoints 修复后 12/12 应全过）
  Phase 1 = M2/M3 主矩阵：双腿镜像 + 俯仰自由 + 翻倒判据
              design{m} × mode{locked,bc} × wb × Fr × (τ_max × r_w | bc)
标定目标：
  · 翻倒相图：wb–Fr 边界（locked 基线 vs bc）
  · bc 的 τ_max / r_w 甜点；停车距离 / 是否停住
  · Fr 上界：v2.5 裸足 Fr≥4 全崩，轮足能推到哪
用法（单行）：
  cd /mnt/zihanw/FFY && python p9w3_probe.py --workers 32 2>&1 | tee outputs/v3_p9w3/run.log
"""
from __future__ import annotations
import os, sys, json, time, argparse, itertools
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, ".")
sys.path.insert(0, "src/stage10_v2")

# ---- 锚定设计（v2.5 al7075 标准工况可行解，figs_v25al/fig1 同源）----
CHOSEN = dict(
    X=[113.1004, 2.0272, 1.2446, 1.2371, 4.6829, 16.2987,
       0.0199, 125.6335, 129.8789, 29.1107],
    m=4.7305, v0=1.3718, kc=8.681e5)

FR_LIST   = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
WB_LIST   = [0.3, 0.5, 0.7, 0.9]
TAU_LIST  = [2.0, 4.0, 8.0, 16.0]
RW_LIST   = [0.02, 0.03, 0.045]
M_LIST    = [4.7305, 12.0]          # 5 kg 级（M350）与 12 kg 级（CW-15）
V0, KC    = CHOSEN["v0"], CHOSEN["kc"]


def _fr_to_vx(fr, m):
    lref = (10**(0.479 + 0.391*np.log10(m*1000.))/1000.) * 3.85
    return fr*np.sqrt(9.81*lref)


def _one(job):
    import wheel_probe as W
    import physics_v2 as P
    base = {**P.SCEN_BIRD_X, "hip_damp_unified": True}
    kind = job["kind"]
    vx = _fr_to_vx(job["fr"], job["m"])
    t0 = time.time()
    try:
        if kind == "m1":
            r = W.eval_wheel(CHOSEN["X"], job["m"], V0, KC, 0.15, base, v_x=vx,
                             mode=job["mode"], pitch_free=False, guards=True)
        else:
            wheel = dict(r_w=job["r_w"], m_w=0.04, tau_max=job["tau"], omega_th=20.0)
            r = W.eval_wheel2(CHOSEN["X"], job["m"], V0, KC, 0.15, base, v_x=vx,
                              mode=job["mode"], wheel=wheel, wb=job["wb"], T=1.2, h=2e-4)
            if r.get("fail") == "solver":       # 一次半步长重试
                r = W.eval_wheel2(CHOSEN["X"], job["m"], V0, KC, 0.15, base, v_x=vx,
                                  mode=job["mode"], wheel=wheel, wb=job["wb"], T=1.2, h=1e-4)
                r["retried"] = True
    except Exception as e:
        r = dict(fail="except", err=str(e)[:100])
    r = {k: v for k, v in r.items() if not k.startswith("_")}
    r.update(job); r["wall_s"] = round(time.time()-t0, 1)
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--out", default="outputs/v3_p9w3")
    ap.add_argument("--quick", action="store_true", help="缩小矩阵冒烟")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    frs, wbs, taus, rws, ms = FR_LIST, WB_LIST, TAU_LIST, RW_LIST, M_LIST
    if a.quick:
        frs, wbs, taus, rws, ms = [0.0, 2.0], [0.7], [8.0], [0.03], [4.7305]

    jobs = []
    # Phase 0 · M1 回归（单腿锁俯仰，m=chosen）
    for mode in ("rigid", "locked", "brake", "bc"):
        for fr in (0.0, 0.5, 2.0):
            jobs.append(dict(kind="m1", mode=mode, fr=fr, m=CHOSEN["m"],
                             wb=None, tau=8.0, r_w=0.03))
    # Phase 1 · 主矩阵
    for m in ms:
        for wb in wbs:
            for fr in frs:
                jobs.append(dict(kind="m2", mode="locked", m=m, wb=wb, fr=fr,
                                 tau=0.0, r_w=0.03))
                for tau, rw in itertools.product(taus, rws):
                    jobs.append(dict(kind="m2", mode="bc", m=m, wb=wb, fr=fr,
                                     tau=tau, r_w=rw))
    print(f"[p9w3] jobs={len(jobs)} workers={a.workers}", flush=True)

    t0 = time.time(); out = []
    with Pool(a.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_one, jobs, chunksize=2)):
            out.append(r)
            if (i+1) % 50 == 0:
                nok = sum(1 for x in out if not x.get("fail"))
                print(f"  {i+1}/{len(jobs)}  ok={nok}  {time.time()-t0:.0f}s", flush=True)
    with open(os.path.join(a.out, "p9w3.json"), "w") as f:
        json.dump(out, f, ensure_ascii=False)
    print(f"[p9w3] done {time.time()-t0:.0f}s → {a.out}/p9w3.json", flush=True)

    # ---- 摘要 ----
    m1 = [r for r in out if r["kind"] == "m1"]
    print(f"\n== Phase0 M1 回归: {sum(1 for r in m1 if not r.get('fail'))}/{len(m1)} 通过 ==")
    for r in sorted(m1, key=lambda r: (r['mode'], r['fr'])):
        if r.get("fail"): print(f"   !! {r['mode']} Fr={r['fr']}: {r['fail']}")

    m2 = [r for r in out if r["kind"] == "m2"]
    print("\n== Phase1 翻倒相图（best-of τ×r_w；格值=peak_g，×=全配置皆败）==")
    for m in ms:
        for mode in ("locked", "bc"):
            print(f"-- m={m:.1f}kg {mode} --   Fr: " +
                  " ".join(f"{fr:>5}" for fr in frs))
            for wb in wbs:
                row = []
                for fr in frs:
                    cand = [r for r in m2 if r["m"] == m and r["mode"] == mode
                            and r["wb"] == wb and r["fr"] == fr and not r.get("fail")]
                    row.append(f"{min(c['peak_g'] for c in cand):>5.1f}" if cand else "    ×")
                print(f"   wb={wb:.1f}: " + " ".join(row))
    # bc 甜点
    ok_bc = [r for r in m2 if r["mode"] == "bc" and not r.get("fail")]
    if ok_bc:
        from collections import Counter
        cnt = Counter((r["tau"], r["r_w"]) for r in ok_bc)
        print("\n== bc 通过数最多的 (τ_max, r_w) ==")
        for (tau, rw), n in cnt.most_common(5):
            print(f"   τ={tau:>4} r_w={rw}: {n} 过")
        stop = [r for r in ok_bc if r.get("stopped")]
        print(f"\n停住率（bc 通过样本内）: {len(stop)}/{len(ok_bc)}")
        hi = [r for r in ok_bc if r["fr"] >= 3.0]
        if hi:
            print(f"Fr≥3 存活: {len(hi)} 例，peak_g 中位 "
                  f"{np.median([r['peak_g'] for r in hi]):.2f}，"
                  f"滚距中位 {np.median([r['roll_dist_mm'] for r in hi])/1000:.2f} m")


if __name__ == "__main__":
    main()

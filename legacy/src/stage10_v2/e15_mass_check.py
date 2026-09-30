"""E15 · G1 验收门:v2 物理里质量是否真的活了?

判据(《质量耦合改造方案_v2.md》§三):
  G1-a  同一设计沿 m 扫,peak_a 相对极差 ≥ 10%
  G1-b  可行率随 m 单调下降
不过就回头调物理,不进小试——这是"一次改到位"与"盲目重跑"的分界线。

同时报告:各地形分别贡献多少质量效应、哪条约束在卡人、两遍质量回代的影响。

用法:
  python src/stage10_v2/e15_mass_check.py --ndes 12 --workers 2          # 沙箱
  OMP_NUM_THREADS=1 python src/stage10_v2/e15_mass_check.py \
      --ndes 60 --workers 128 --out outputs/gen_v2_g1                    # A100
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
sys.path.insert(0, os.path.join(HERE, "..", "stage6_surrogate"))
import models as M                                       # noqa: E402
import physics_v2 as P                                   # noqa: E402

# mud/water 超出球-面罚接触模型的有效范围(侵入深度 > 足端球半径),故不在默认表内
TERRAINS = ["concrete", "asphalt", "turf", "wetsand"]
KC_GRID = [2.0e6, 8.0e5, 3.0e5, 1.2e5, 5.0e4, 2.0e4]
MASSES = [1.0, 2.0, 4.0, 8.0, 12.0]


def _job(a):
    xi, m, v0, tname, npass = a[:5]
    lk, lm, lc = (list(a[5:8]) + [False])[:3] if len(a) > 5 else (False, False, False)
    t = P.TERRAIN[tname]
    r = P.eval_v2(xi, m, v0, kc=t["kc"], zeta_c=t["zeta_c"], npass=npass,
                  legacy_kc=lk, legacy_segmass=lm, legacy_cc=lc)
    if r is None or r.get("fail"):
        return dict(fail=(r or {}).get("fail", "none"))
    return dict(peak_a=r["peak_a"], stroke=r["stroke"], F_peak=r["F_peak"],
                mass_frac=r["mass_frac"], struct_over=r["struct_over"],
                mass_over=r["mass_over"], governs=r["governs"],
                D_mm=r["D_mm"], D_max_mm=r["D_max_mm"],
                dpeak_pass_pct=r["dpeak_pass_pct"])


def rel_spread(v, need=3):
    """有效质量点 ≥ need 就算,不要求 5 个全成功。

    软地形上重机体会因塌陷/深陷而无解——那本身就是质量效应的一部分,
    若因此把整条设计丢掉,反而看不到最强的质量依赖(v1 版判据的漏洞)。"""
    v = np.asarray([x for x in v if x is not None and np.isfinite(x)], float)
    if v.size < need:
        return np.nan
    return float((v.max() - v.min()) / max(abs(np.median(v)), 1e-12))


def _job_kc(a):
    xi, m, v0, kc, zc, npass = a
    r = P.eval_v2(xi, m, v0, kc=kc, zeta_c=zc, npass=npass)
    if r is None or r.get("fail"):
        return dict(fail=(r or {}).get("fail", "none"))
    return dict(peak_a=r["peak_a"], stroke=r["stroke"], leg_stroke_mm=r["leg_stroke_mm"],
                sink_mm=r["sink_mm"], struct_over=r["struct_over"],
                mass_over=r["mass_over"], F_peak=r["F_peak"])


def capability_map(args):
    """kc × m 能力图:峰值、地面下陷、可行率。方案 §五 要的两张主图之一。

    读法:硬地那一行几乎平(质量无关的残迹);越往软走,质量效应越强,
    且方向相反——重机体被软地垫得更软(峰值↓)却陷得更深(行程↑),
    两个通道打架 ⇒ 可行域在质量方向上出现**内部最优**。
    """
    lo, hi = np.array(M.LO_BIRD7), np.array(M.HI_BIRD7)
    rng = np.random.default_rng(args.seed)
    X = lo + (hi - lo) * rng.random((args.ndes, len(lo)))
    zc = {2.0e6: 0.05, 8.0e5: 0.08, 3.0e5: 0.12, 1.2e5: 0.20, 5.0e4: 0.30, 2.0e4: 0.35}
    jobs, tags = [], []
    for kc in KC_GRID:
        for i, xi in enumerate(X):
            for m in MASSES:
                jobs.append((tuple(xi), m, args.v0, kc, zc.get(kc, 0.15), args.npass))
                tags.append((kc, i, m))
    print(f"[map] 地面刚度 {len(KC_GRID)} × 设计 {args.ndes} × 质量 {len(MASSES)} "
          f"= {len(jobs)} 次评价")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_job_kc, jobs, chunksize=1))
    print(f"[map] 完成 ({time.time() - t0:.0f}s)\n")
    R = {tg: r for tg, r in zip(tags, rows)}

    def cell(kc, m, fn):
        vs = [fn(R[(kc, i, m)]) for i in range(args.ndes)
              if R[(kc, i, m)] and not R[(kc, i, m)].get("fail")]
        return float(np.median(vs)) if vs else float("nan")

    grid = {}
    for name, fn, unit, scale in [
            ("peak_a", lambda r: r["peak_a"] / 9.81, "g", 1),
            ("sink", lambda r: r["sink_mm"], "mm", 1),
            ("leg_stroke", lambda r: r["leg_stroke_mm"], "mm", 1)]:
        print(f"--- {name} 中位 ({unit}) ---")
        print(f"{'kc (N/m)':>10}" + "".join(f"{m:>9g}" for m in MASSES) + "   沿 m 相对极差")
        grid[name] = {}
        for kc in KC_GRID:
            row = [cell(kc, m, fn) for m in MASSES]
            grid[name][f"{kc:g}"] = row
            sp = rel_spread(row, need=3)
            print(f"{kc:>10.0f}" + "".join(f"{v:>9.2f}" for v in row)
                  + f"{sp*100:>16.1f}%")
        print()

    print(f"--- 可行率 % (g_cap={args.gcap/9.81:.0f}g, s_max={args.smax*1e3:.0f}mm) ---")
    print(f"{'kc (N/m)':>10}" + "".join(f"{m:>9g}" for m in MASSES) + "   峰值位置")
    feas = {}
    for kc in KC_GRID:
        row = []
        for m in MASSES:
            ok = sum(int(P.feasible_v2(R[(kc, i, m)], args.gcap, args.smax)[0])
                     for i in range(args.ndes))
            row.append(100.0 * ok / args.ndes)
        feas[f"{kc:g}"] = row
        am = int(np.argmax(row))
        tag = ("单调↓" if all(row[i] >= row[i+1] - 1e-9 for i in range(len(row)-1))
               else ("单调↑" if all(row[i] <= row[i+1] + 1e-9 for i in range(len(row)-1))
                     else f"内部最优 m={MASSES[am]:g}kg"))
        print(f"{kc:>10.0f}" + "".join(f"{v:>9.1f}" for v in row) + f"{tag:>16}")
    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "e15_map.json")
    json.dump(dict(kc_grid=KC_GRID, masses=MASSES, ndes=args.ndes,
                   gcap=args.gcap, smax=args.smax, grid=grid, feas=feas),
              open(fp, "w"), indent=2, ensure_ascii=False)
    print(f"\n[map] → {fp}")


def channels(args):
    """通道归因:把 v2 的两处改动分别退回 v1 写法,看不变性回来多少。

      A 全 v1 口径        kc=4000mg, 杆件=5%m       → 应回到 ~0(E14 已证)
      B 只改接触          kc 绝对,  杆件=5%m
      C 只改杆件质量      kc=4000mg, 杆件由定尺导出
      D 全 v2             两处都改                   → e15 主表那一栏
    """
    lo, hi = np.array(M.LO_BIRD7), np.array(M.HI_BIRD7)
    rng = np.random.default_rng(args.seed)
    X = lo + (hi - lo) * rng.random((args.ndes, len(lo)))
    #            名称                      legacy_kc  legacy_segmass  legacy_cc
    combos = [("A 全 v1 口径",              True,  True,  False),
              ("B 接触(刚度+阻尼)",         False, True,  False),
              ("B2 只绝对化刚度",           False, True,  True),
              ("C 只改杆件质量",            True,  False, False),
              ("D 全 v2",                   False, False, False)]
    tn = args.terrain
    jobs, tags = [], []
    for cn, lk, lm, lc in combos:
        for i, xi in enumerate(X):
            for m in MASSES:
                jobs.append((tuple(xi), m, args.v0, tn, args.npass, lk, lm, lc))
                tags.append((cn, i, m))
    print(f"[通道] {len(combos)} 组合 × 设计 {args.ndes} × 质量 {len(MASSES)} "
          f"= {len(jobs)} 次评价,地形={tn}")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=1))
    print(f"[通道] 完成 ({time.time() - t0:.0f}s)\n")
    R = {tg: r for tg, r in zip(tags, rows)}
    print(f"{'组合':<22}{'peak_a 沿 m 相对极差':>22}{'stroke':>10}{'有效设计':>10}")
    out = {}
    for cn, _, _, _ in combos:
        sp, ss_, n = [], [], 0
        for i in range(args.ndes):
            rs = [R[(cn, i, m)] for m in MASSES]
            gv = lambda k: [(r[k] if (r and not r.get("fail")) else None) for r in rs]
            p = rel_spread(gv("peak_a"))
            if not np.isfinite(p):
                continue
            n += 1
            sp.append(p)
            ss_.append(rel_spread(gv("stroke")))
        v = float(np.median(sp)) if sp else float("nan")
        out[cn] = dict(peak=v, stroke=float(np.median(ss_)) if ss_ else float("nan"), n=n)
        print(f"{cn:<22}{v*100:>21.2f}%{out[cn]['stroke']*100:>9.1f}%{n:>10d}")
    os.makedirs(args.out, exist_ok=True)
    json.dump(dict(terrain=tn, ndes=args.ndes, masses=MASSES, channels=out),
              open(os.path.join(args.out, "e15_channels.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[通道] → {os.path.join(args.out, 'e15_channels.json')}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ndes", type=int, default=12)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--v0", type=float, default=1.2)
    ap.add_argument("--gcap", type=float, default=10 * 9.81)
    ap.add_argument("--smax", type=float, default=0.024)
    ap.add_argument("--npass", type=int, default=2)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="outputs/gen_v2_g1")
    ap.add_argument("--channels", action="store_true", help="只跑通道归因")
    ap.add_argument("--map", action="store_true", help="只跑 kc×m 能力图")
    ap.add_argument("--terrain", default="turf", help="通道归因用哪种地形")
    ap.add_argument("--terrains", default=None,
                    help="逗号分隔,覆盖默认地形表,如 concrete,turf")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    global TERRAINS
    if args.terrains:
        TERRAINS = [t.strip() for t in args.terrains.split(",") if t.strip()]
    if args.map:
        return capability_map(args)
    if args.channels:
        return channels(args)

    lo, hi = np.array(M.LO_BIRD7), np.array(M.HI_BIRD7)
    rng = np.random.default_rng(args.seed)
    X = lo + (hi - lo) * rng.random((args.ndes, len(lo)))

    jobs, tags = [], []
    for i, xi in enumerate(X):
        for tn in TERRAINS:
            for m in MASSES:
                jobs.append((tuple(xi), m, args.v0, tn, args.npass))
                tags.append((i, tn, m))
    print(f"[G1] 设计 {args.ndes} × 地形 {len(TERRAINS)} × 质量 {len(MASSES)} "
          f"= {len(jobs)} 次评价(每次 {args.npass} 遍仿真)")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = list(ex.map(_job, jobs, chunksize=1))
    print(f"[G1] 完成 ({time.time() - t0:.0f}s)")

    R = {tg: r for tg, r in zip(tags, rows)}

    # ---------------------------------------------------------- G1-a 质量效应
    print("\n=== G1-a 同一设计沿 m 的相对极差(中位;判据 ≥ 10%)===")
    print(f"{'地形':<10}{'kc (N/m)':>12}{'peak_a':>10}{'stroke':>10}{'F_peak':>10}{'有效设计':>10}")
    spreads = {}
    for tn in TERRAINS:
        sp_p, sp_s, sp_f, nok = [], [], [], 0
        for i in range(args.ndes):
            rs = [R[(i, tn, m)] for m in MASSES]
            if any((r is None or r.get("fail")) for r in rs):
                continue
            nok += 1
            sp_p.append(rel_spread([r["peak_a"] for r in rs]))
            sp_s.append(rel_spread([r["stroke"] for r in rs]))
            sp_f.append(rel_spread([r["F_peak"] for r in rs]))
        spreads[tn] = dict(peak=float(np.median(sp_p)) if sp_p else np.nan,
                           stroke=float(np.median(sp_s)) if sp_s else np.nan,
                           F=float(np.median(sp_f)) if sp_f else np.nan, n=nok)
        s = spreads[tn]
        print(f"{tn:<10}{P.TERRAIN[tn]['kc']:>12.0f}{s['peak']*100:>9.1f}%"
              f"{s['stroke']*100:>9.1f}%{s['F']*100:>9.1f}%{nok:>10d}")
    best = max((v["peak"] for v in spreads.values() if np.isfinite(v["peak"])),
               default=float("nan"))

    # ---------------------------------------------------------- G1-b 可行率
    print(f"\n=== G1-b 可行率随 m 的走向(g_cap={args.gcap/9.81:.0f}g, "
          f"s_max={args.smax*1e3:.0f}mm;判据:单调下降)===")
    hdr = "".join(f"{m:>9g}" for m in MASSES)
    print(f"{'地形':<10}{hdr}   (可行率 %)")
    feas_tab, why = {}, {}
    for tn in TERRAINS:
        row = []
        for m in MASSES:
            ok = 0; tot = 0
            for i in range(args.ndes):
                r = R[(i, tn, m)]
                tot += 1
                f, ws = P.feasible_v2(r, args.gcap, args.smax)
                ok += int(f)
                for w in ws:                      # 记全部违反项,不带检查顺序伪影
                    why[(tn, m, w)] = why.get((tn, m, w), 0) + 1
            row.append(100.0 * ok / max(tot, 1))
        feas_tab[tn] = row
        mono = all(row[i] >= row[i + 1] - 1e-9 for i in range(len(row) - 1))
        print(f"{tn:<10}" + "".join(f"{v:>9.0f}" for v in row)
              + ("   单调↓" if mono else "   非单调"))

    print("\n卡人的是哪条约束(次数):")
    for tn in TERRAINS:
        for m in MASSES:
            ws = {w: c for (t_, m_, w), c in why.items() if t_ == tn and m_ == m and w != "ok"}
            if ws:
                print(f"  {tn:<9} m={m:>4g}kg  " +
                      "  ".join(f"{w}×{c}" for w, c in sorted(ws.items(), key=lambda kv: -kv[1])))

    # ---------------------------------------------------------- 附:定点收敛
    ok_rows = [r for r in rows if r and not r.get("fail")]
    dp = [r["dpeak_pass_pct"] for r in ok_rows]
    if dp:
        print(f"\n两遍质量回代对峰值的影响:中位 {np.median(dp):.2f}%  "
              f"P90 {np.percentile(dp, 90):.2f}%  最大 {max(dp):.2f}%")
    mf = [r["mass_frac"] for r in ok_rows]
    print(f"腿质量分数:中位 {np.median(mf)*100:.2f}%  范围 "
          f"[{min(mf)*100:.2f}%, {max(mf)*100:.2f}%]")

    res = dict(masses=MASSES, terrains=TERRAINS, ndes=args.ndes, v0=args.v0,
               spreads=spreads, feas=feas_tab,
               why={f"{k[0]}|{k[1]:g}|{k[2]}": v for k, v in why.items()},
               dpeak_pass_median=float(np.median(dp)) if dp else None,
               mass_frac_median=float(np.median(mf)) if mf else None)
    fp = os.path.join(args.out, "e15_g1.json")
    json.dump(res, open(fp, "w"), indent=2, ensure_ascii=False)

    # ---------------------------------------------------------- 判决
    mono_any = any(all(r[i] >= r[i + 1] - 1e-9 for i in range(len(r) - 1))
                   for r in feas_tab.values())
    drop_any = any(r[0] - r[-1] > 1e-9 for r in feas_tab.values())
    print(f"\n[G1-a] 最强地形下 peak_a 沿 m 的相对极差 = {best*100:.1f}%  "
          + ("通过 ✅" if best >= 0.10 else "未通过 ❌(判据 ≥10%)"))
    print(f"[G1-b] 可行率随 m {'单调下降 ✅' if (mono_any and drop_any) else '未见单调下降 ❌'}")
    print(f"[G1] {'两条都过,可进 G2 小试' if (best >= 0.10 and mono_any and drop_any) else '未过,回头调物理,别进 G2'}")
    print(f"[G1] → {fp}")


if __name__ == "__main__":
    main()

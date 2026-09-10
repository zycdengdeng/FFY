# -*- coding: utf-8 -*-
"""P9 · 足端摩擦与水平速度的准入体检 + 崩溃阶梯（C 组的门）。

v2.5 改了两处物理，都要先确认数值上站得住，再花十几个小时重跑工厂：

  ① 机体从「竖直滑轨」放开成「x-z 面内自由平动」。
     v2.3 里 `CreatePrismaticJoint(axis=[0,0,1])` 把机体锁成只能上下走，
     髋-足水平偏移产生的力全被约束反力免费吃掉 —— 所以倾斜姿态不花任何代价。
     E26 就是被这一点误导的：它算出 28° 的峰值只有 50° 的 42%，
     但那个姿态需要 μ ≈ 0.70，打印尼龙对混凝土只有 0.30–0.40，真实地面上会打滑。
  ② 初速度加水平分量 v_x = Fr·√(g·L_ref(m))。

三种模式：

  --mode reg    **回归测试**。从 v2.3 的 factory.jsonl 里取原样的设计与工况，
                用 planar=False / v_x=0 重跑，和当年存的 peak_a 逐条比。
                这一条不过说明重构动了不该动的地方，直接回滚，别看后面。
  --mode a      **准入体检**。planar=True，Fr ∈ [0,2]，扫三档 q1_0。
                五条判据全过才允许开工厂。其中第 5 条是 E26 那张 μ 表的实测替代。
  --mode b      **崩溃阶梯**。Fr 一路推到 15，不设通过与否，
                产出就是「现有接触模型在哪一档崩」本身。

用法：
    python src/stage10_v2/p9_friction.py --mode reg --n 200 --workers 128
    python src/stage10_v2/p9_friction.py --mode a --workers 128
    python src/stage10_v2/p9_friction.py --mode b --workers 128
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from agroup_io import load_blocks, to_Y, setup_font, CRIM, STEEL, GREEN, ORANGE  # noqa: E402
import froude as FR                                                             # noqa: E402

MASSES = (5.0, 8.0, 12.0, 20.0, 30.0)
Q1_PROBE = (28.0, 38.0, 50.0)          # E26 的三档：会滑 / 临界 / 安全
FR_A = (0.0, 0.5, 1.0, 1.5, 2.0)
FR_B = FR.FR_LADDER                    # (2, 4, 6, 9, 12, 15)
KC_PROBE = (1.0e6, 2.0e5)              # 硬地 / 草地
GCAP_G, SMAX = 10.0, 0.024


def zeta_of_kc(kc):
    return float(np.clip(10 ** (1.771 - 0.4875 * np.log10(kc)), 0.05, 0.35))


def _eval(a):
    """子进程：连 import 一起包在 try 里，无人值守时不许打断整批。"""
    try:
        import physics_v2 as P
        x, m, v0, kc, q1, v_x, planar = a
        base = {**P.SCEN_BIRD_X, "hip_damp_unified": True, "foot_mode": "bearing",
                "mu_from_ground": bool(planar)}
        xx = list(x)
        if q1 is not None:
            xx = (xx + [0.0] * 10)[:10]
            xx[9] = float(q1)          # 第 10 维就是 q1_0（度）
        r = P.eval_v2(tuple(xx), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2,
                      base=base, v_x=v_x, planar=planar)
    except Exception as e:
        return dict(fail=type(e).__name__)
    if r is None or r.get("fail"):
        return dict(fail=(r or {}).get("fail", "none"))
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    return dict(peak_g=r["peak_a"] / 9.81,
                a_res_g=float(r.get("a_res", np.nan)) / 9.81,
                leg_stroke_mm=float(r.get("leg_stroke", r["stroke"])) * 1e3,
                sink_mm=float(r.get("sink", 0.0)) * 1e3,
                mu_demand=float(r.get("mu_demand", np.nan)),
                mu_ground=float(r.get("mu_ground", np.nan)),
                slip_mm=float(r.get("slip", 0.0)) * 1e3,
                x_drift_mm=float(r.get("x_drift", 0.0)) * 1e3,
                e_gain=float(r.get("e_gain", np.nan)),
                leg_mass_g=float(r["leg_mass_kg"]) * 1e3,
                ok=bool(ok), why=list(why))


def run(jobs, workers, tag):
    from concurrent.futures import ProcessPoolExecutor
    print("[p9/%s] %d 个点，多进程 %d" % (tag, len(jobs), workers), flush=True)
    out, t0, nbad = [], time.time(), 0
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_eval, [j[0] for j in jobs], chunksize=2), 1):
            out.append({**jobs[i - 1][1], **r})
            nbad += ("fail" in r)
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                print("      %d/%d 失败 %d (%.0fs, 预计剩 %.0fs)"
                      % (i, len(jobs), nbad, el, el / i * (len(jobs) - i)), flush=True)
            if i == 100 and nbad == i:
                print("\n[p9] 前 100 个全失败 —— exudyn 或 physics_v2 有问题，提前收工。")
                return out
    return out


# ---------------------------------------------------------------- reg
def mode_reg(args):
    """拿 v2.3 工厂里存的真实结果做基准，逐条比 peak_a。"""
    blocks = load_blocks(args.factory)
    if not blocks:
        raise SystemExit("[p9] 找不到 v2.3 工厂：%s" % args.factory)
    iP = 0                                  # KEYS_V2[0] = peak_a
    rng = np.random.default_rng(9)
    jobs = []
    for _ in range(args.n):
        b = blocks[rng.integers(len(blocks))]
        j = int(rng.integers(len(b["X"])))
        y = to_Y(b)[j]
        if not np.isfinite(y[iP]):
            continue
        jobs.append(((b["X"][j], b["m"], b["v0"], b["kc"], None, 0.0, False),
                     dict(ref_peak_g=float(y[iP]) / 9.81, cid=b["cid"], j=j)))
    rows = run(jobs, args.workers, "reg")
    good = [r for r in rows if "fail" not in r]
    if len(good) < 10:
        raise SystemExit("[p9/reg] 有效结果太少（%d），先查 exudyn" % len(good))
    d = np.array([100.0 * (r["peak_g"] - r["ref_peak_g"]) / max(r["ref_peak_g"], 1e-9)
                  for r in good])
    print("=" * 74)
    print("回归测试：planar=False, v_x=0，与 v2.3 工厂存档逐条比 peak_a")
    print("   n = %d（失败 %d）" % (len(good), len(rows) - len(good)))
    print("   相对偏差：中位 %+.4f%%   P95 %.4f%%   最大 %.4f%%"
          % (np.median(d), np.percentile(np.abs(d), 95), np.max(np.abs(d))))
    ok = np.max(np.abs(d)) < 0.5
    print("   → %s" % ("通过（最大偏差 < 0.5%），重构没动物理。" if ok else
                       "**不通过**。重构动到了不该动的地方，回滚后再查。"))
    print("=" * 74)
    json.dump(dict(n=len(good), med=float(np.median(d)),
                   p95=float(np.percentile(np.abs(d), 95)),
                   max=float(np.max(np.abs(d))), passed=bool(ok)),
              open(os.path.join(args.out, "p9_reg.json"), "w"), indent=1)
    return ok


# ---------------------------------------------------------------- 探针设计
def probe_designs(n, seed):
    from bioprior import BioPrior
    pr = BioPrior("bio", v25=True)
    rng = np.random.default_rng(seed)
    U = rng.random((n, pr.ndim))
    return pr, U


def mode_ab(args, frs, tag):
    pr, U = probe_designs(args.nprobe, args.seed)
    jobs = []
    for m in MASSES:
        X = pr.expand(U, m)
        for kc in KC_PROBE:
            for q1 in Q1_PROBE:
                for fr in frs:
                    vx = float(FR.fr_to_vx(fr, m))
                    for x in X:
                        jobs.append(((x, m, 1.2, kc, q1, vx, True),
                                     dict(m=m, kc=kc, q1=q1, Fr=float(fr), v_x=vx)))
    rows = run(jobs, args.workers, tag)
    fp = os.path.join(args.out, "p9_%s.jsonl" % tag)
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("→ %s" % fp)
    return rows


def report_a(rows, args):
    good = [r for r in rows if "fail" not in r]
    n, ng = len(rows), len(good)
    fail_rate = 100.0 * (n - ng) / max(n, 1)
    print("=" * 74)
    print("准入体检（planar=True, Fr ∈ [0,2]）")
    print("   总点数 %d，solver 失败 %d（%.1f%%）" % (n, n - ng, fail_rate))
    if not good:
        return False
    sink = np.array([r["sink_mm"] for r in good])
    eg = np.array([r["e_gain"] for r in good])
    slip = np.array([r["slip_mm"] for r in good])
    frs = np.array([r["Fr"] for r in good])
    c2 = fail_rate <= args.fail_cap * 100
    c3 = np.nanpercentile(np.abs(eg), 95) < 0.02
    slip_mono = all(np.nanmedian(slip[frs == a]) <= np.nanmedian(slip[frs == b]) + 1e-6
                    for a, b in zip(sorted(set(frs)), sorted(set(frs))[1:]))
    print("   2 失败率 ≤ %.0f%%          : %s" % (args.fail_cap * 100, "过" if c2 else "**不过**"))
    print("   3 |e_gain| P95 = %.4f < 0.02: %s" % (np.nanpercentile(np.abs(eg), 95),
                                                   "过" if c3 else "**不过**"))
    print("   4 slip 随 Fr 单调           : %s" % ("过" if slip_mono else "**不过**"))
    print("     地面下陷中位 %.2f mm，P95 %.2f mm" % (np.median(sink), np.percentile(sink, 95)))

    print("-" * 74)
    print("   5 ★ mu_demand 实测 vs E26 的 tan(前倾角) 估计")
    print("      %6s %12s %12s %10s" % ("q1_0", "mu_demand中位", "E26估计", "通过验收%"))
    EST = {28.0: 0.70, 38.0: 0.47, 50.0: 0.23}
    c5, dev = True, []
    for q1 in Q1_PROBE:
        sel = [r for r in good if r["q1"] == q1 and r["Fr"] == 0.0]
        if not sel:
            continue
        md = float(np.nanmedian([r["mu_demand"] for r in sel]))
        okp = 100.0 * np.mean([r["ok"] for r in sel])
        e = EST[q1]
        dev.append(abs(md - e) / max(e, 1e-9))
        print("      %5.0f° %12.3f %12.2f %9.0f%%" % (q1, md, e, okp))
    if dev:
        c5 = max(dev) < 0.30
        print("      最大相对偏差 %.0f%% —— %s" % (100 * max(dev),
              "过，那张 μ 表可以直接用来定 q1_0 下界" if c5 else
              "**不过，以实测为准重画那张表再定盒子**"))
    print("-" * 74)
    allok = c2 and c3 and slip_mono and c5
    print("→ %s" % ("五条全过，可以开工厂。" if allok else "**有判据不过，先别开工厂。**"))
    print("=" * 74)
    json.dump(dict(n=n, fail_rate=fail_rate, e_gain_p95=float(np.nanpercentile(np.abs(eg), 95)),
                   slip_monotonic=bool(slip_mono), mu_ok=bool(c5), passed=bool(allok)),
              open(os.path.join(args.out, "p9a_report.json"), "w"), indent=1)
    return allok


def report_b(rows, args):
    good = [r for r in rows if "fail" not in r]
    print("=" * 74)
    print("崩溃阶梯（planar=True，Fr 2→15）—— 不设通过与否，产出就是「在哪一档崩」")
    print("   %6s %10s %10s %12s %12s %12s"
          % ("Fr", "失败率%", "下陷P95mm", "|e_gain|P95", "滑移中位mm", "mu_demand中位"))
    tab = []
    for fr in FR_B:
        sel_all = [r for r in rows if r.get("Fr") == fr]
        sel = [r for r in good if r["Fr"] == fr]
        if not sel_all:
            continue
        fr_fail = 100.0 * (len(sel_all) - len(sel)) / len(sel_all)
        row = dict(Fr=float(fr), fail=fr_fail)
        if sel:
            row.update(sink_p95=float(np.percentile([r["sink_mm"] for r in sel], 95)),
                       e_p95=float(np.nanpercentile(np.abs([r["e_gain"] for r in sel]), 95)),
                       slip=float(np.nanmedian([r["slip_mm"] for r in sel])),
                       mu=float(np.nanmedian([r["mu_demand"] for r in sel])))
        tab.append(row)
        print("   %6.1f %10.1f %10.2f %12.4f %12.1f %12.3f"
              % (fr, fr_fail, row.get("sink_p95", np.nan), row.get("e_p95", np.nan),
                 row.get("slip", np.nan), row.get("mu", np.nan)))
    bad = [t for t in tab if t["fail"] > 20 or t.get("e_p95", 0) > 0.02]
    print("-" * 74)
    if bad:
        fr0 = bad[0]["Fr"]
        print("   → 现有接触模型在 **Fr ≈ %.0f** 开始失效（≈ %.0f–%.0f m/s，看质量）。"
              % (fr0, FR.fr_to_vx(fr0, 5.0), FR.fr_to_vx(fr0, 30.0)))
        print("     这就是论文里那句方法边界：本文的点接触 + 库仑摩擦模型只在 Fr < %.0f 有效，" % fr0)
        print("     滑跑着陆需要另一套接触模型和轮系拓扑 —— 单独立项，不在 v2.5 范围内。")
    else:
        print("   → 一路到 Fr = %.0f 都数值稳。**v2.6 可以直接把区间拉宽重跑，" % FR_B[-1])
        print("     参数化一行不用改** —— 这就是当初用 Fr 而不是 v_x 的价值。")
    print("=" * 74)
    json.dump(dict(ladder=tab), open(os.path.join(args.out, "p9b_ladder.json"), "w"),
              ensure_ascii=False, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["reg", "a", "b"], required=True)
    ap.add_argument("--factory", default=os.path.join(ROOT, "outputs/v23_data_bio/factory.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/v25_p9"))
    ap.add_argument("--n", type=int, default=200, help="reg 模式抽多少条")
    ap.add_argument("--nprobe", type=int, default=64, help="a/b 模式的探针设计数")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--fail-cap", type=float, default=0.10, help="准入允许的 solver 失败率")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    if args.mode == "reg":
        sys.exit(0 if mode_reg(args) else 1)
    elif args.mode == "a":
        sys.exit(0 if report_a(mode_ab(args, FR_A, "a"), args) else 1)
    else:
        report_b(mode_ab(args, FR_B, "b"), args)


if __name__ == "__main__":
    main()

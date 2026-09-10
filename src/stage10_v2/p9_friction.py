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
from e26_q1_sweep import lean_of                                                # noqa: E402
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
        x, m, v0, kc, q1, v_x, planar, mu = a
        base = {**P.SCEN_BIRD_X, "hip_damp_unified": True, "foot_mode": "bearing",
                "mu_from_ground": bool(planar)}
        xx = list(x)
        if q1 is not None:
            xx = (xx + [0.0] * 10)[:10]
            xx[9] = float(q1)          # 第 10 维就是 q1_0（度）
        r = P.eval_v2(tuple(xx), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2,
                      base=base, v_x=v_x, planar=planar, mu=mu)
    except Exception as e:
        return dict(fail=type(e).__name__)
    if r is None or r.get("fail"):
        return dict(fail=(r or {}).get("fail", "none"))
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    L1mm = float(xx[0])
    return dict(peak_g=r["peak_a"] / 9.81, L1_mm=L1mm,
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
        jobs.append(((b["X"][j], b["m"], b["v0"], b["kc"], None, 0.0, False, None),
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
                        jobs.append(((x, m, 1.2, kc, q1, vx, args.planar, None),
                                     dict(m=m, kc=kc, q1=q1, Fr=float(fr), v_x=vx)))
    rows = run(jobs, args.workers, tag)
    fp = os.path.join(args.out, "p9_%s.jsonl" % tag)
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("→ %s" % fp)
    return rows


MU_GRID = (0.30, 0.40, 0.50, 0.70, 1.00, 1.50, 2.50)
SLIP_STICK_MM = 2.0        # 滑移小于此值就算"站住了"


def mode_mu(args):
    """扫 μ 求**临界摩擦系数** —— 回答 E26 那张 tan(前倾角) 表到底对不对。

    为什么不能直接看 mu_demand：P9a 首跑发现三档姿态的 mu_demand 全是 0.502/0.502/0.505，
    而且恰好等于 μ 本身。因为足端一旦开始滑，滑动摩擦就饱和了，F_x/F_z 被 μ 钳死 ——
    **测到的是 μ，不是"需要多少 μ"**。

    正确的测法是反过来：给足够大的 μ 让足端粘住，再一档一档降，
    看降到多少开始滑。那个临界值才是"需求"。这里用 Fr=0（没有水平初速，
    纯粹由姿态引起的水平力）扫 μ，滑移 < 2 mm 记为站住。
    """
    pr, U = probe_designs(args.nprobe, args.seed)
    q1s = np.arange(28.0, 55.1, 3.0)          # 比 P9a 密，为了画出 μ_crit(q1_0) 曲线
    jobs = []
    for m in (5.0, 12.0, 30.0):
        X = pr.expand(U, m)
        for q1 in q1s:
            for mu in MU_GRID:
                for x in X:
                    jobs.append(((x, m, 1.2, 1.0e6, q1, 0.0, True, float(mu)),
                                 dict(m=m, q1=float(q1), mu=float(mu))))
    rows = run(jobs, args.workers, "mu")
    fp = os.path.join(args.out, "p9_mu.jsonl")
    with open(fp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")

    good = [r for r in rows if "fail" not in r]
    print("=" * 74)
    print("μ 扫描：多大的摩擦系数才能让足端站住（Fr=0，硬地 k_c=1e6，滑移<%.0fmm 算站住）"
          % SLIP_STICK_MM)
    print("   %6s %10s %10s   %s" % ("q1_0", "μ_crit", "tan(前倾)", "各 μ 下的站住率"))
    tab = []
    for q1 in q1s:
        sel = [r for r in good if abs(r["q1"] - q1) < 1e-9]
        if not sel:
            continue
        stick = {}
        for mu in MU_GRID:
            s2 = [r for r in sel if abs(r["mu"] - mu) < 1e-9]
            stick[mu] = float(np.mean([r["slip_mm"] < SLIP_STICK_MM for r in s2])) if s2 else 0.0
        crit = next((mu for mu in MU_GRID if stick[mu] >= 0.5), float("nan"))
        # E26 的估计：把腿当成沿足-髋连线受力的二力杆，需要的 μ ≈ tan(前倾角)
        lean = float(np.median([lean_of(x, q1)[0] for x in pr.expand(U, 12.0)]))
        est = float(np.tan(np.radians(lean)))
        tab.append(dict(q1=float(q1), mu_crit=crit, lean=lean, tan_lean=est,
                        stick={str(k): v for k, v in stick.items()}))
        print("   %5.0f° %10s %10s   %s"
              % (q1, ("%.2f" % crit) if np.isfinite(crit) else ">2.5", "%.2f" % est,
                 " ".join("%.0f%%" % (100 * stick[mu]) for mu in MU_GRID)))
    print("   （最后一列依次对应 μ = %s）" % ", ".join("%.2f" % mu for mu in MU_GRID))
    print("-" * 74)
    fin = [t for t in tab if np.isfinite(t["mu_crit"])]
    if fin:
        lo, hi = min(t["mu_crit"] for t in fin), max(t["mu_crit"] for t in fin)
        d = [abs(t["mu_crit"] - t["tan_lean"]) / max(t["tan_lean"], 1e-9) for t in fin]
        print("   μ_crit 跨 28–55° 的范围：%.2f – %.2f" % (lo, hi))
        print("   与 E26 的 tan(前倾角) 估计：中位相对偏差 %.0f%% —— %s"
              % (100 * np.median(d),
                 "估计站得住，E26 那张表可以用" if np.median(d) < 0.30 else
                 "**估计不准，q1_0 的盒子下界按实测的 μ_crit 定**"))
        print("   打印尼龙对混凝土 0.30–0.40 —— %s"
              % ("够用" if hi <= 0.40 else
                 "**不够，仿鸟构型在硬地上需要抓地结构（齿/爪/高摩擦垫）**"))
    else:
        print("   **所有姿态在 μ ≤ 2.5 下都站不住** —— 这不是摩擦不够，是模型里有别的问题，先查。")
    json.dump(dict(mu_grid=list(MU_GRID), slip_stick_mm=SLIP_STICK_MM, rows=tab),
              open(os.path.join(args.out, "p9_mu.json"), "w"), ensure_ascii=False, indent=1)
    print("=" * 74)


def report_a(rows, args):
    """准入体检。**判据在 P9a 首跑之后改过一次**，两处：

    ① e_gain 原来判 |e_gain| 的 P95 < 0.02 —— 写反了。实测中位 −1.10、100% 为负，
       那是对的物理：机体在 1 秒里掉了约 81 mm，重力势能降幅本来就远超初动能
       (½v₀² 只有 0.72 J/kg)。数值健康只该查**能量被造出来**，所以改判 max(e_gain) < +0.02。
    ② 可行率不再当成通过/不通过的闸，而是**用来给 Fr 上界定标** ——
       仿鸟构型（两腿同姿态并排、水平力不抵消）到底能扛多大的 Fr，是要量出来的，
       不是先拍一个 [0,2] 再去验。
    """
    good = [r for r in rows if "fail" not in r]
    n, ng = len(rows), len(good)
    fail_rate = 100.0 * (n - ng) / max(n, 1)
    print("=" * 74)
    print("准入体检（planar=%s, Fr ∈ [0,2]）" % args.planar)
    print("   总点数 %d，solver 失败 %d（%.1f%%）" % (n, n - ng, fail_rate))
    from collections import Counter
    if n - ng:
        print("   失败原因：%s" % Counter(r["fail"] for r in rows if "fail" in r).most_common(5))
    if not good:
        return False

    eg = np.array([r["e_gain"] for r in good])
    slip = np.array([r["slip_mm"] for r in good])
    frs = np.array([r["Fr"] for r in good])
    c2 = fail_rate <= args.fail_cap * 100
    eg_max = float(np.nanmax(eg))
    c3 = eg_max < 0.02
    slip_mono = all(np.nanmedian(slip[frs == a]) <= np.nanmedian(slip[frs == b]) + 1e-6
                    for a, b in zip(sorted(set(frs)), sorted(set(frs))[1:]))
    print("   1 失败率 ≤ %.0f%%              : %s (%.1f%%)"
          % (args.fail_cap * 100, "过" if c2 else "**不过**", fail_rate))
    print("   2 max(e_gain) < 0.02（不造能）: %s (最大 %+.4f，中位 %+.2f)"
          % ("过" if c3 else "**不过**", eg_max, float(np.nanmedian(eg))))
    print("   3 slip 随 Fr 单调             : %s" % ("过" if slip_mono else "**不过**"))

    # ---- 用可行率给 Fr 上界定标 ----
    print("-" * 74)
    print("   4 ★ 可行率随 Fr 怎么掉（据此定主工厂的 Fr 上界）")
    print("      %6s %10s %12s %12s %12s"
          % ("Fr", "可行率%", "滑移中位mm", "滑移/L1", "a_res中位g"))
    curve = []
    for fr in sorted(set(frs)):
        sel = [r for r in good if r["Fr"] == fr]
        ok = 100.0 * np.mean([r["ok"] for r in sel])
        sl = float(np.nanmedian([r["slip_mm"] for r in sel]))
        rel = float(np.nanmedian([r["slip_mm"] / max(r["L1_mm"], 1e-9) for r in sel]))
        ar = float(np.nanmedian([r["a_res_g"] for r in sel]))
        curve.append(dict(Fr=float(fr), feas=ok, slip_mm=sl, slip_over_L1=rel, a_res_g=ar))
        print("      %6.1f %10.1f %12.1f %12.2f %12.2f" % (fr, ok, sl, rel, ar))
    okfr = [c["Fr"] for c in curve if c["feas"] >= args.feas_floor * 100]
    fr_rec = max(okfr) if okfr else 0.0
    print("      → 可行率 ≥ %.0f%% 的最大 Fr = **%.1f**" % (args.feas_floor * 100, fr_rec))
    if fr_rec < 2.0:
        print("        比原定的 2.0 低。主工厂的 --fr-max 应改成这个数，")
        print("        否则一大半样本落在物理上站不住的区间里，训练目标会被污染。")

    print("-" * 74)
    allok = c2 and c3 and slip_mono
    print("→ %s" % ("三条数值判据全过。Fr 上界按上面第 4 条取。"
                    if allok else "**有数值判据不过，先别开工厂。**"))
    print("=" * 74)
    json.dump(dict(planar=bool(args.planar), n=n, fail_rate=fail_rate,
                   e_gain_max=eg_max, e_gain_med=float(np.nanmedian(eg)),
                   slip_monotonic=bool(slip_mono), curve=curve,
                   fr_recommended=float(fr_rec), passed=bool(allok)),
              open(os.path.join(args.out, "p9a_report.json"), "w"),
              ensure_ascii=False, indent=1)
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
    ap.add_argument("--mode", choices=["reg", "a", "b", "mu"], required=True)
    ap.add_argument("--factory", default=os.path.join(ROOT, "outputs/v23_data_bio/factory.jsonl"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/v25_p9"))
    ap.add_argument("--n", type=int, default=200, help="reg 模式抽多少条")
    ap.add_argument("--nprobe", type=int, default=64, help="a/b 模式的探针设计数")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seed", type=int, default=9)
    ap.add_argument("--fail-cap", type=float, default=0.10, help="准入允许的 solver 失败率")
    ap.add_argument("--planar", type=int, default=1,
                    help="1=仿鸟(两腿同姿态并排,水平力不抵消,机体面内自由平动);"
                         "0=对置构型(姿态引起的水平力在机体内部抵消,保留竖直约束)")
    ap.add_argument("--feas-floor", type=float, default=0.20,
                    help="给 Fr 上界定标时要求的最低可行率")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    args.planar = bool(args.planar)
    if args.mode == "mu":
        mode_mu(args); return
    if args.mode == "reg":
        sys.exit(0 if mode_reg(args) else 1)
    elif args.mode == "a":
        sys.exit(0 if report_a(mode_ab(args, FR_A, "a"), args) else 1)
    else:
        report_b(mode_ab(args, FR_B, "b"), args)


if __name__ == "__main__":
    main()

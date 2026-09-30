# -*- coding: utf-8 -*-
"""E27 · 「髋在足正上方」作为**结构**假设（B 组，零新增仿真，复用 E26 的缓存）。

**这条的性质变过一次。** 原方案是"由垂线约束反解 q1_0，给 bioprior 降一维"，
理由是生物学。E28 把这个理由否掉了：真鸟触地时前倾 30.5° ± 6.4°，
11 次落水没有一次接近 0，而且压缩过程中还继续变斜。**所以那条方案撤销。**

但 Bigoni 那句话还剩一个**纯力学**的内核，值得单独查一次：

    腿越竖直 → 节段越接近轴压而非弯曲 → size_segment 里由弯矩定的 D 越小 → 腿越轻。

这是个和生物无关的结构命题，而且我们已经有数据能回答它 —— E26 的扫描里
每一档 q1_0 都记了 D_mm / leg_mass / M_Nm / F_peak，直接算就行。

**判据**
  · 竖直化显著减重（> 15%）且峰值代价可接受 → 写进讨论：**这是一个性能-仿生的显式取舍**，
    并在论文里明说我们选了仿生一侧；
  · 减重不显著 → 这条彻底关掉，以后不用再提。

**一个必须写在结论旁边的边界**：现在的模型里机体被 `CreatePrismaticJoint(axis=[0,0,1])`
锁成只能竖直平动，髋-足水平偏移产生的力矩由这个约束反力承担。所以本实验测到的
"倾斜的弯矩代价"是真的；但等 v2.5 加了水平速度 v_x 之后，受力图会变，
这条结论要重新测一次。

用法：
    python src/stage10_v2/e27_vertical_structural.py --fig
"""
import argparse, collections, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from agroup_io import setup_font, CRIM, STEEL, GREEN, ORANGE          # noqa: E402

# 与 e8_struct.size_segment 一致的薄壁圆管系数（壁厚 = 0.10·D）
C_BEND, C_AREA, C_I = 0.05796, 0.2827, 0.02898
NLEGS, Q_BASE = 2, 50.0
SEGS = ["tarso (L1)", "tibio (L2)", "femur (L3)"]


def stress_split(r):
    """返回每段的 (弯曲应力, 轴向应力, 弯曲占比)。M 的取法与 size_structure 一致。"""
    Ma, Mk, Mh = [v / NLEGS for v in r["M_Nm"]]
    Mb = [Ma, max(Ma, Mk), max(Mk, Mh)]
    F = r["F_peak"] / NLEGS
    out = []
    for s in range(3):
        D = r["D_mm"][s] * 1e-3
        sb = Mb[s] / (C_BEND * D ** 3)
        sa = F / (C_AREA * D ** 2)
        out.append((sb, sa, sb / max(sb + sa, 1e-30)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.join(ROOT, "outputs/v23_e26/e26_sweep_cache.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--mass-trigger", type=float, default=15.0, help="减重显著性阈值 %%")
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()
    out_dir = args.out or os.path.dirname(args.cache)
    if not os.path.exists(args.cache):
        raise SystemExit("[e27] 找不到 E26 的缓存 %s —— 先跑 e26_q1_sweep.py" % args.cache)

    rows = [json.loads(ln) for ln in open(args.cache, encoding="utf-8") if ln.strip()]
    good = [r for r in rows if "fail" not in r and np.isfinite(r.get("peak_g", np.nan))]
    print("[e27] 有效点 %d / %d" % (len(good), len(rows)))
    by = collections.defaultdict(dict)
    for r in good:
        by[(r["ci"], r["di"])][round(r["q1"], 4)] = r

    # ---------------- ① 弯曲 vs 轴压：倾斜到底带来多少弯矩 ----------------
    print("=" * 78)
    print("① 弯曲应力占比随 q1_0 的变化（腿越竖直，理论上弯曲占比越低）")
    qs = sorted({round(r["q1"], 4) for r in good if abs(r["q1"] - r["q1_vert"]) > .05})
    tab = collections.defaultdict(lambda: [[] for _ in range(3)])
    for r in good:
        for s, (_, _, fb) in enumerate(stress_split(r)):
            tab[round(r["q1"], 4)][s].append(fb)
    print("   %8s | %s" % ("q1_0", "  ".join("%-14s" % n for n in SEGS)))
    for q in qs:
        print("   %7.1f° | %s" % (q, "  ".join("%13.1f%%" % (100 * np.median(tab[q][s]))
                                               for s in range(3))))
    print("   （弯曲占比 = σ_bend / (σ_bend + σ_axial)，与 size_segment 的组合应力同式）")

    # ---------------- ② 竖直化的净收益 ----------------
    print("=" * 78)
    print("② 从 q1_0 = %.0f° 走到「髋在足正上方」的 q1_0*，各项变化" % Q_BASE)
    d = collections.defaultdict(list)
    for k, dd in by.items():
        r0 = dd.get(round(Q_BASE, 4))
        qv = dd[list(dd)[0]]["q1_vert"]
        rv = dd.get(round(qv, 4))
        if r0 is None or rv is None:
            continue
        rel = lambda a, b: 100.0 * (b - a) / max(abs(a), 1e-12)
        d["q1_vert"].append(qv)
        d["mass"].append(rel(r0["leg_mass_g"], rv["leg_mass_g"]))
        d["peak"].append(rel(r0["peak_g"], rv["peak_g"]))
        d["stroke"].append(rel(r0["leg_stroke_mm"], rv["leg_stroke_mm"]))
        d["Dmax"].append(rel(max(r0["D_mm"]), max(rv["D_mm"])))
        d["lean0"].append(r0["lean"]); d["leanv"].append(rv["lean"])
        d["fb0"].append(np.mean([f for _, _, f in stress_split(r0)]))
        d["fbv"].append(np.mean([f for _, _, f in stress_split(rv)]))
        d["ok0"].append(bool(r0["ok"])); d["okv"].append(bool(rv["ok"]))
    if not d["mass"]:
        raise SystemExit("[e27] 配不上对（基准档或垂线档缺失），检查 E26 的 grid 里有没有 50°")
    n = len(d["mass"])
    print("   配对设计 %d 个，q1_0* 中位 %.1f°（前倾 %.1f° → %.1f°）"
          % (n, np.median(d["q1_vert"]), np.median(d["lean0"]), np.median(d["leanv"])))
    print("-" * 78)
    for key, lab, good_dir in (("mass", "腿质量", -1), ("Dmax", "最大管径", -1),
                               ("peak", "峰值过载", -1), ("stroke", "可用行程", +1)):
        v = np.array(d[key])
        arrow = "↓好" if good_dir < 0 else "↑好"
        print("   %-8s %+7.1f%%   (四分位 %+.1f%% ~ %+.1f%%)   %s"
              % (lab, np.median(v), np.percentile(v, 25), np.percentile(v, 75), arrow))
    print("   弯曲占比  %.1f%% → %.1f%%" % (100 * np.median(d["fb0"]), 100 * np.median(d["fbv"])))
    print("   通过验收  %.0f%% → %.0f%%"
          % (100 * np.mean(d["ok0"]), 100 * np.mean(d["okv"])))

    # ---------------- 判定 ----------------
    dm, dp = float(np.median(d["mass"])), float(np.median(d["peak"]))
    print("=" * 78)
    if dm <= -args.mass_trigger:
        print("判定：竖直化确实显著减重（%+.1f%%，超过 %.0f%% 阈值），代价是峰值 %+.1f%%。"
              % (dm, args.mass_trigger, dp))
        print()
        print("     写法：**这是一个显式的性能-仿生取舍**。")
        print("     竖直化在纯力学上更省材料（弯矩臂变短），但 E28 已经证明真鸟不这么落地。")
        print("     我们选仿生一侧，并把这个代价量化写出来 —— 这比「我们就是仿生」有说服力得多。")
        print("     顺带回应 Bigoni：他的直觉在力学上是对的，只是不是生物学上的事实。")
    else:
        print("判定：竖直化**没有**带来显著减重（%+.1f%%，阈值 %.0f%%）。"
              % (dm, args.mass_trigger))
        print()
        print("     这条彻底关掉，以后不用再提。原因也说得清：")
        print("     `size_segment` 里的 D 主要由**关节峰值力矩**定，而关节力矩来自弹簧刚度×转角，")
        print("     不是来自髋-足偏移的静力臂。所以把腿摆竖直并不会显著卸掉弯矩。")
        if dp > 0:
            print("     而且峰值过载还涨了 %+.1f%% —— 竖直化在这个模型里是纯亏。" % dp)

    print("=" * 78)
    print("⚠ 边界：现在机体被 CreatePrismaticJoint(axis=[0,0,1]) 锁成只能竖直平动，")
    print("        髋-足水平偏移的力矩由约束反力承担。v2.5 加了水平速度 v_x 之后受力图会变，")
    print("        这条结论要重新测一次。")

    blob = dict(n_pairs=n, q1_vert_median=float(np.median(d["q1_vert"])),
                lean_base=float(np.median(d["lean0"])), lean_vert=float(np.median(d["leanv"])),
                d_mass_pct=dm, d_peak_pct=dp,
                d_stroke_pct=float(np.median(d["stroke"])),
                d_Dmax_pct=float(np.median(d["Dmax"])),
                bend_frac_base=float(np.median(d["fb0"])),
                bend_frac_vert=float(np.median(d["fbv"])),
                feas_base=float(np.mean(d["ok0"])), feas_vert=float(np.mean(d["okv"])),
                mass_trigger=args.mass_trigger,
                verdict="tradeoff" if dm <= -args.mass_trigger else "closed",
                bend_frac_by_q={str(q): [float(np.median(tab[q][s])) for s in range(3)]
                                for q in qs})
    fp = os.path.join(out_dir, "e27_vertical_structural.json")
    json.dump(blob, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % fp)
    if args.fig:
        _fig(by, qs, tab, d, out_dir, dm, dp)


def _fig(by, qs, tab, d, out_dir, dm, dp):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    COLS = [STEEL, GREEN, ORANGE]
    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.4), gridspec_kw=dict(wspace=.28))

    a = ax[0]
    for s in range(3):
        a.plot(qs, [100 * np.median(tab[q][s]) for q in qs], "-o", ms=4,
               color=COLS[s], lw=2, label=SEGS[s])
    a.axvline(Q_BASE, color=CRIM, ls="--", lw=1.6)
    a.axvline(np.median(d["q1_vert"]), color="k", ls=":", lw=1.6)
    a.text(Q_BASE - .5, a.get_ylim()[0], "50°", color=CRIM, ha="right", fontsize=9)
    a.text(np.median(d["q1_vert"]) + .5, a.get_ylim()[0], "垂线 %.0f°" % np.median(d["q1_vert"]),
           fontsize=9)
    a.set_xlabel("q1_0 / °"); a.set_ylabel("弯曲应力占比 / %")
    a.set_title("(a) 腿摆竖直，弯矩卸掉了多少", fontsize=11)
    a.legend(fontsize=9, frameon=False); a.grid(alpha=.22)

    b = ax[1]
    for k, dd in by.items():
        q = np.array(sorted(dd)); mm = np.array([dd[x]["leg_mass_g"] for x in q])
        i0 = np.argmin(abs(q - Q_BASE))
        b.plot(q, 100 * (mm / mm[i0] - 1), "-", lw=.9, alpha=.3, color=STEEL)
    b.axhline(0, color="k", lw=.9)
    b.axhline(-15, color=GREEN, ls="--", lw=1.6)
    b.text(qs[0], -15, " 减重 15% 阈值", color=GREEN, fontsize=9, va="bottom")
    b.axvline(Q_BASE, color=CRIM, ls="--", lw=1.6)
    b.set_xlabel("q1_0 / °"); b.set_ylabel("腿质量相对 50° 的变化 / %")
    b.set_title("(b) 竖直化能不能减重\n垂线档中位 %+.1f%%" % dm, fontsize=11)
    b.grid(alpha=.22)

    c = ax[2]
    labs = ["腿质量", "最大管径", "峰值过载", "可用行程"]
    keys = ["mass", "Dmax", "peak", "stroke"]
    med = [np.median(d[k]) for k in keys]
    c.barh(range(4), med, color=[CRIM if v > 0 else GREEN for v in med], alpha=.85)
    for i, v in enumerate(med):
        c.text(v + (1 if v >= 0 else -1), i, "%+.1f%%" % v, va="center",
               ha="left" if v >= 0 else "right", fontsize=10, fontweight="bold")
    c.set_yticks(range(4)); c.set_yticklabels(labs, fontsize=10); c.invert_yaxis()
    c.axvline(0, color="k", lw=.9)
    c.set_xlabel("50° → 垂线档的相对变化 / %")
    c.set_title("(c) 竖直化的净账", fontsize=11)
    c.grid(alpha=.22, axis="x")
    fig.tight_layout()
    fp = os.path.join(out_dir, "e27_vertical_structural.png")
    fig.savefig(fp, dpi=170); print("→ %s" % fp)


if __name__ == "__main__":
    main()

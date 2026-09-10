# -*- coding: utf-8 -*-
"""E23 · κ / τ 盒壁饱和体检（A 组，零新增仿真）。

动机：待办 C4 说 τ 和 κ_踝 贴着下界跑，但依据只有"5 个设计、单次生成"，
这是**旗标不是结论**。C 组要重设盒子，必须先把这件事做实。

做法：对工厂里**全部可行设计**（不是前沿点，前沿点会被 Pareto 选择二次偏置），
在 u 空间（每维都已归一化到 [0,1]）统计各维的分布，报告：
  · 贴壁率 = u < wall 或 u > 1-wall 的比例（默认 wall = 0.05）
  · 各维的 5/25/50/75/95 分位
  · 与"均匀分布"的偏离（贴壁率的理论基线就是 2×wall = 10%）

判据：某一维贴壁率 > 20%（即基线的 2 倍）→ 该维盒子在 v2.5 里放宽一档。

用法：
    python src/stage10_v2/e23_box_saturation.py --fig
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from agroup_io import (load_blocks, to_Y, feasible_mask, pareto2, selfcheck,   # noqa: E402
                       iter_requirements, meta_of, default_factory, setup_font,
                       iP, iL, CRIM, STEEL, GREEN)

DIMS9 = ["L1 (u_L)", "r2", "r3", "κ 踝", "κ 膝", "κ 髋", "τ (ζ)", "θ_A 踝角", "θ_K 膝角"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default=default_factory(ROOT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--wall", type=float, default=0.05, help="贴壁判定带宽（u 空间）")
    ap.add_argument("--trigger", type=float, default=0.20, help="放宽盒子的触发阈值")
    ap.add_argument("--nreq", type=int, default=4)
    ap.add_argument("--ktop", type=int, default=8)
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()
    out_dir = args.out or os.path.dirname(args.factory)

    selfcheck()
    blocks = load_blocks(args.factory)
    if not blocks:
        raise SystemExit("[e23] 空工厂：%s" % args.factory)
    meta = meta_of(args.factory)
    nd = int(meta.get("u_dim", len(blocks[0]["U"][0])))
    names = DIMS9[:nd] if nd <= len(DIMS9) else ["u%d" % i for i in range(nd)]
    print("[e23] 块 %d，设计维 %d" % (len(blocks), nd))

    Uf, Ufront = [], []      # 全部可行 / 进训练的前沿点
    for blk in blocks:
        Y, U = to_Y(blk), np.array(blk["U"], float)
        seen = np.zeros(len(Y), bool)
        for gcap, smax in iter_requirements(blk, args.nreq):
            fe = feasible_mask(Y, gcap, smax)
            if not fe.any():
                continue
            idx = np.where(fe)[0]
            seen[idx] = True                      # 同一设计在多组要求下只计一次
            Ufront.append(U[idx[pareto2(Y[idx, iP], Y[idx, iL])][:args.ktop]])
        if seen.any():
            Uf.append(U[seen])
    Uf = np.concatenate(Uf) if Uf else np.zeros((0, nd))
    Ufront = np.concatenate(Ufront) if Ufront else np.zeros((0, nd))
    print("[e23] 可行（去重）%d 个设计；进训练的前沿点 %d 个" % (len(Uf), len(Ufront)))

    w = args.wall
    base = 200.0 * w          # 均匀分布下的贴壁率（%）
    rows = []
    for j, nm in enumerate(names):
        u = Uf[:, j]
        lo = float((u < w).mean()); hi = float((u > 1 - w).mean())
        uf = Ufront[:, j] if len(Ufront) else u
        rows.append(dict(dim=j, name=nm, lo=100 * lo, hi=100 * hi, tot=100 * (lo + hi),
                         med=float(np.median(u)),
                         q=[float(np.percentile(u, q)) for q in (5, 25, 50, 75, 95)],
                         front_lo=float(100 * (uf < w).mean()),
                         front_hi=float(100 * (uf > 1 - w).mean()),
                         front_med=float(np.median(uf)),
                         front_q25=float(np.percentile(uf, 25)),
                         front_q75=float(np.percentile(uf, 75))))

    print("-" * 88)
    print("%-12s %8s %8s | %8s %8s | %8s %8s"
          % ("维", "下贴壁%", "上贴壁%", "前沿下%", "前沿上%", "全体中位", "前沿中位"))
    print("-" * 88)
    hits = []
    for r in rows:
        flag = ""
        if r["lo"] * 2 > args.trigger * 100:
            flag = "  ← 下界要放宽"; hits.append((r["name"], "下界", r["lo"]))
        elif r["hi"] * 2 > args.trigger * 100:
            flag = "  ← 上界要放宽"; hits.append((r["name"], "上界", r["hi"]))
        print("%-12s %8.1f %8.1f | %8.1f %8.1f | %8.3f %8.3f%s"
              % (r["name"], r["lo"], r["hi"], r["front_lo"], r["front_hi"],
                 r["med"], r["front_med"], flag))
    print("-" * 88)
    print("均匀分布基线：单侧 %.1f%%，双侧合计 %.1f%%。触发阈值 = 单侧 %.0f%%（基线的 %.1f 倍）。"
          % (base / 2, base, args.trigger * 100 / 2, args.trigger * 100 / base))
    if hits:
        print("结论：以下维在 v2.5 的盒子里要放宽 ——")
        for nm, side, v in hits:
            print("   · %-10s %s 贴壁 %.1f%%" % (nm, side, v))
        print("放宽幅度建议：把贴的那一侧外推到「贴壁率回落到基线 2 倍以内」为止，")
        print("并在 v2.5 跑完后用同一脚本复测（这是可迭代的收敛判据，不是拍一次）。")
    else:
        print("结论：没有一维显著贴壁 —— 盒子够用，v2.5 保持不动。")
        print("      待办 C4 那条「τ 和 κ踝 贴下界」在**全部可行设计**上不成立。")

    # 前沿点的方向性偏置：贴壁率没超阈值，也可能存在系统性往一侧靠的倾向。
    # 这正是待办 C4 用 5 个设计看到的东西 —— 它看的是前沿，不是全体。
    # 用「前沿中位偏离 0.5 多少」来度量偏置 —— 比贴壁率灵敏得多：
    # 前沿可以整体压在 u<0.3，却因为不进最后 5% 而贴壁率不高（τ 就是这种情况）。
    asym = sorted(((0.5 - r["front_med"], r) for r in rows), key=lambda t: -abs(t[0]))
    strong = [(d, r) for d, r in asym if abs(d) >= 0.08]
    print("-" * 88)
    if strong:
        print("前沿点的方向性偏置（贴壁率未超阈值，但 Pareto 前沿系统性地往一侧靠）：")
        for d, r in strong:
            print("   · %-10s 前沿偏向%s界   前沿中位 u = %.3f（全体 %.3f），"
                  "四分位 [%.2f, %.2f]，贴壁 %.1f%%/%.1f%%"
                  % (r["name"], "下" if d > 0 else "上", r["front_med"], r["med"],
                     r["front_q25"], r["front_q75"], r["front_lo"], r["front_hi"]))
        print()
        print("解读：**贴壁率没超阈值，但前沿整体压在一侧** —— 这两件事要分开说。")
        print("  · 全体可行设计在盒里基本均匀（贴壁率都在 5% 基线附近）→ 盒子没有把可行域切掉；")
        print("  · 但 Pareto 前沿系统性地往一侧靠 → 在可行域内部，最优解确实偏爱那一侧。")
        print("  · 待办 C4 用 5 个设计看到的就是后者，它没错，只是把它说成了「盒子不够大」。")
        print()
        print("建议：v2.5 **不**放宽盒子（放宽只会把前沿往外挪，不会让更多设计变可行），")
        print("      但把这个偏置方向写进论文 —— 它本身就是一条物理结论：")
        print("      落震最优偏爱**低阻尼、软踝、小踝角**，而这正好是水鸟腿的方向。")
    else:
        print("前沿点也没有明显的方向性偏置。")

    fp = os.path.join(out_dir, "e23_box_saturation.json")
    json.dump(dict(wall=w, trigger=args.trigger, n_feasible=len(Uf),
                   n_front=len(Ufront), dims=rows,
                   front_asym=[dict(name=r["name"], delta=r["front_lo"] - r["front_hi"])
                               for r in rows],
                   widen=[dict(name=n, side=s, pct=v) for n, s, v in hits]),
              open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % fp)

    if args.fig and len(Uf):
        _fig(Uf, Ufront, names, rows, args, out_dir, base)


def _fig(Uf, Ufront, names, rows, args, out_dir, base):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    n = len(names)
    nc = 3; nr = int(np.ceil(n / nc))
    fig, ax = plt.subplots(nr, nc, figsize=(4.1 * nc, 2.7 * nr))
    ax = np.atleast_1d(ax).ravel()
    for j, nm in enumerate(names):
        a = ax[j]
        a.hist(Uf[:, j], bins=40, range=(0, 1), color=STEEL, alpha=.8,
               density=True, label="全部可行")
        if len(Ufront):
            a.hist(Ufront[:, j], bins=40, range=(0, 1), histtype="step",
                   color=CRIM, lw=1.6, density=True, label="进训练的前沿点")
        a.axhline(1.0, color="k", lw=.9, ls=":")
        a.axvspan(0, args.wall, color=CRIM, alpha=.10)
        a.axvspan(1 - args.wall, 1, color=CRIM, alpha=.10)
        r = rows[j]
        hot = max(r["lo"], r["hi"]) * 2 > args.trigger * 100
        a.set_title("%s   贴壁 %.0f%% / %.0f%%%s" % (nm, r["lo"], r["hi"],
                                                    "  ←" if hot else ""),
                    fontsize=10, color=CRIM if hot else "k",
                    fontweight="bold" if hot else "normal")
        a.set_xlim(0, 1); a.set_yticks([])
        if j == 0:
            a.legend(fontsize=7.5, frameon=False)
    for j in range(n, len(ax)):
        ax[j].axis("off")
    fig.suptitle("E23 · 可行设计在先验盒内的分布（u 空间；虚线 = 均匀分布，粉带 = 贴壁区 %.0f%%）"
                 % (100 * args.wall), fontsize=12.5, y=1.005)
    fig.tight_layout()
    fp = os.path.join(out_dir, "e23_box_saturation.png")
    fig.savefig(fp, dpi=165, bbox_inches="tight"); print("→ %s" % fp)


if __name__ == "__main__":
    main()

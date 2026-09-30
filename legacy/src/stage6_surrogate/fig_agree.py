# -*- coding: utf-8 -*-
"""机器和鸟一致在哪里 —— 只用**非循环**的证据。

先说清楚什么是循环的：设计盒子本身就写成
    log10 L1 = 0.479 + 0.391·log10 m + 0.0784·u ,  u ∈ [-2.5, +2.5]
所以「腿长随体重按幂律走」是参数化给的，不是机器发现的。网络选的是无量纲的 u。

左图：机器在这条走廊里往哪边靠。u 是它相对生物中线的偏离；
      若机器对体重没有意见，u 应当是一条水平线（= 完全照搬生物指数 0.391）。
右图：分段比 r2/r3 与触地姿态的盒子是「生物实测全距」，中位不占便宜，
      机器落在哪里是完全独立的选择。

用法: python src/stage6_surrogate/fig_agree.py [--lang cn|en]"""
import os, json, argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Rectangle
U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="cn", choices=["cn", "en"])
LANG = ap.parse_args().lang
OUT = f"{U}/outputs/ppt_{LANG}"; os.makedirs(OUT, exist_ok=True)
_have = {f.name for f in fm.fontManager.ttflist}
_cjk = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if f in _have), None)
plt.rcParams.update({"font.sans-serif": ([_cjk] if (_cjk and LANG == "cn") else []) + ["DejaVu Sans"],
                     "font.family": "sans-serif", "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
BIRD, HARD, SOFT, SAND, GRY = "#8E2A34", "#1b6ca8", "#2E7D5B", "#7a4bbf", "#666666"
A0, B0 = 0.47899383301187465, 0.39112926377807683
SIG = (0.030063514185437547 / 1.96) * 0.35 * np.sqrt(213)      # = 0.0784
UMAX = 2.5
T = {"cn": dict(
        sup="机器在哪些地方和鸟一致",
        Lt="① 我们给的是一条以生物中线为轴的走廊，机器在里面往哪边靠",
        xl="机身质量 (kg)", yl="偏离生物中线  u  （单位：σ = 0.079 dex）",
        flat="若机器对体重没有意见，这里应是一条水平线（照搬生物指数 0.391）",
        wall="走廊边界 ±2.5σ", hard="硬地 k_c=1e6", turf="草地 k_c=1e5", sand="湿沙 k_c=5e4",
        Rt="② 这几个量的盒子是「生物实测全距」，中位不占便宜 —— 机器自己落在哪里",
        box="生物实测全距（= 搜索范围）", bird="鸟的中位", mach="机器的中位（3 工况 × 2 种子）",
        names=["分段比 $r_2=L_2/L_1$", "分段比 $r_3=L_3/L_1$", "触地踝角 $\\theta_A$ (°)", "触地膝角 $\\theta_K$ (°)"],
        cap="分段比上机器几乎复刻了鸟（差 <1% 盒宽）；姿态上它比鸟蹲得更低。"),
     "en": dict(
        sup="Where the machine agrees with the bird",
        Lt="(1) Inside the biological corridor - where does it settle?",
        xl="airframe mass (kg)", yl="offset from the biological line  u  (in σ = 0.079 dex)",
        flat="no opinion about mass = a flat line (the bird's 0.391)",
        wall="corridor wall ±2.5σ", hard="rigid, $k_c$=1e6", turf="turf, $k_c$=1e5", sand="wet sand, $k_c$=5e4",
        Rt="(2) Boxes = full biological range; the centre is not privileged",
        box="measured biological range (= search box)", bird="bird median", mach="machine median (3 terrains, 2 seeds)",
        names=["$r_2 = L_2/L_1$", "$r_3 = L_3/L_1$",
               "ankle $\\theta_A$ (deg)", "knee $\\theta_K$ (deg)"],
        cap="On segment ratios the machine matches the bird to within 1% of the box width; "
            "on posture it crouches lower.")}[LANG]

CS = [("concrete1.2", HARD, T["hard"]), ("turf1.2", SOFT, T["turf"]), ("wetsand1.2", SAND, T["sand"])]
def grab(c, key_idx):
    v = []
    for d in ("outputs/v23_e20_r12", "outputs/v23_e20_r12_s1"):
        Z = np.load(f"{U}/{d}/e20_bio_raw.npz", allow_pickle=True)
        if c + "__x7" not in Z: continue
        X = Z[c + "__x7"][..., key_idx]; ok = Z[c + "__ok"]; m = Z["m_all"]
        v.append((m, X, ok))
    return v

fig = plt.figure(figsize=(15.2, 6.4))
gs = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], wspace=.20)

# ---------------- 左：走廊里的位置 ----------------
axL = fig.add_subplot(gs[0, 0])
axL.axhspan(-UMAX, UMAX, color="#8E2A34", alpha=.055, lw=0)
for yv in (-UMAX, UMAX):
    axL.axhline(yv, color=BIRD, lw=1.6, ls="--")
axL.axhline(0, color=BIRD, lw=2.6)
for c, col, lab in CS:
    xs, ys = [], []
    for m, X, ok in grab(c, 0):
        for i in range(len(m)):
            s = X[i][ok[i]]
            if len(s) >= 5:
                u = (np.log10(s) - A0 - B0 * np.log10(m[i] * 1000)) / SIG
                xs.append(m[i]); ys.append(np.median(u))
    xs, ys = np.array(xs), np.array(ys)
    o = np.argsort(xs); xs, ys = xs[o], ys[o]
    sl, ic = np.polyfit(np.log10(xs), ys, 1)
    beff = B0 + SIG * sl
    axL.plot(xs, ys, "o", color=col, ms=6.5, mec="k", mew=.5, alpha=.85, zorder=4)
    xx = np.linspace(np.log10(xs.min()), np.log10(xs.max()), 40)
    axL.plot(10 ** xx, ic + sl * xx, color=col, lw=2.6, zorder=3,
             label=f"{lab}   →  b = {beff:.3f}")
axL.set_xscale("log"); axL.set_xticks([2, 5, 10, 20, 40]); axL.set_xticklabels(["2", "5", "10", "20", "40"])
axL.set_ylim(-3.0, 3.0); axL.set_xlabel(T["xl"], fontsize=12.5); axL.set_ylabel(T["yl"], fontsize=12.5)
axL.set_title(T["Lt"], fontsize=13.5, fontweight="bold", loc="left", pad=10)
axL.text(2.05, -1.55, T["flat"], fontsize=11, color=BIRD, fontweight="bold")
axL.text(2.05, UMAX + .12, T["wall"], fontsize=10.5, color=BIRD, va="bottom", ha="left")
axL.legend(loc="lower right", fontsize=11, framealpha=.95)
axL.grid(alpha=.22)
for s in ("top", "right"): axL.spines[s].set_visible(False)

# ---------------- 右：盒子里机器 vs 鸟 ----------------
axR = fig.add_subplot(gs[0, 1])
BOX = [("r2", 1, 1.49, 2.09, 1.80), ("r3", 2, 0.84, 1.28, 1.06),
       ("thA", 7, 113., 160., 144.), ("thK", 8, 118., 157., 133.)]
axR.set_xlim(-.03, 1.03); axR.set_ylim(-.7, len(BOX) - .3)
for i, (k, idx, lo, hi, bird) in enumerate(BOX):
    y = len(BOX) - 1 - i
    axR.add_patch(Rectangle((0, y - .17), 1, .34, fc="#eceff3", ec="#b9c0c9", lw=1.2, zorder=1))
    mv = []
    for c, col, _ in CS:
        vals = []
        for m, X, ok in grab(c, idx): vals += list(X[ok])
        if vals: mv.append(np.median(vals))
    mm = float(np.mean(mv))
    fb, fm_ = (bird - lo) / (hi - lo), (mm - lo) / (hi - lo)
    axR.plot([fb], [y], "D", color=BIRD, ms=13, mec="k", mew=.8, zorder=5)
    axR.plot([fm_], [y], "o", color=HARD, ms=13, mec="k", mew=.8, zorder=5)
    if abs(fm_ - fb) > .02:
        axR.annotate("", xy=(fm_, y), xytext=(fb, y),
                     arrowprops=dict(arrowstyle="->", lw=2, color="#444"), zorder=4)
    axR.text(-.02, y + .30, T["names"][i], fontsize=12.5, fontweight="bold", ha="left", va="bottom")
    axR.text(0, y - .30, f"{lo:g}", fontsize=10.5, color=GRY, ha="left", va="top")
    axR.text(1, y - .30, f"{hi:g}", fontsize=10.5, color=GRY, ha="right", va="top")
    d = 100 * abs(fm_ - fb)
    axR.text(1.0, y + .30, (f"鸟 {bird:g} → 机器 {mm:.2f}   差 {d:.0f}% 盒宽" if LANG == "cn"
                            else f"bird {bird:g} \u2192 {mm:.2f}   ({d:.0f}% of box)"),
             fontsize=11.5, ha="right", va="bottom",
             color=(SOFT if d < 5 else "#c0392b"), fontweight="bold")
axR.set_yticks([]); axR.set_xticks([])
for s in ("top", "right", "left", "bottom"): axR.spines[s].set_visible(False)
axR.set_title(T["Rt"], fontsize=13.5, fontweight="bold", loc="left", pad=10)
axR.plot([], [], "D", color=BIRD, ms=11, label=T["bird"])
axR.plot([], [], "o", color=HARD, ms=11, label=T["mach"])
axR.add_patch(Rectangle((0, 0), 0, 0, fc="#eceff3", ec="#b9c0c9", label=T["box"]))
axR.legend(loc="lower center", fontsize=11, ncol=3, frameon=False, bbox_to_anchor=(.5, -.16))
fig.suptitle(T["sup"], fontsize=17, fontweight="bold", y=1.01)
fig.text(.5, -.045, T["cap"], ha="center", fontsize=13, fontweight="bold", color=BIRD)
out = f"{OUT}/{LANG}_F_agree.png"
fig.savefig(out, dpi=165, bbox_inches="tight"); print("→", out)

# -*- coding: utf-8 -*-
"""机器和鸟一致在哪里：生成器从未被告知「腿长要随体重按幂律走」，
它对每个体重独立求解；但把它选的腿长画到双对数上，自己collapse 成一条直线。
数据：outputs/v23_e20_r12{,_s1}/e20_bio_raw.npz（两个种子）
用法: python src/stage6_surrogate/fig_agree.py [--lang cn|en]"""
import os, json, argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="cn", choices=["cn", "en"])
LANG = ap.parse_args().lang
OUT = f"{U}/outputs/ppt_{LANG}"; os.makedirs(OUT, exist_ok=True)
_have = {f.name for f in fm.fontManager.ttflist}
_cjk = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if f in _have), None)
plt.rcParams.update({"font.sans-serif": ([_cjk] if (_cjk and LANG == "cn") else []) + ["DejaVu Sans"],
                     "font.family": "sans-serif", "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
BIRD, HARD, SOFT, GRY = "#8E2A34", "#1b6ca8", "#2E7D5B", "#666666"
T = {"cn": dict(
        sup="生成器从没被告诉过「腿长要随体重按幂律走」——它自己走出了这条律",
        L="生成器在每个体重上独立选出的腿长",
        xl="机身质量 (kg)", yl="生成器选的跗跖长 $L_1$ (mm)",
        R="标度指数 b：机器 vs 生物 vs 理论",
        xl2="标度指数 b", note="点 = 该体重下所有可行设计的中位；细线 = 拟合",
        bird="水鸟（实测，亲缘校正）", geo="几何相似", ela="弹性相似",
        mh="机器 · 硬地", ms="机器 · 软地", seeds="误差棒 = 两个随机种子",
        cap="在鸟真正着陆的软地面上，机器选的指数和鸟几乎一样；只有到刚性地面才明显变平。"),
     "en": dict(
        sup="The generator was never told that leg length must follow a power law of mass - it found one anyway",
        L="Leg lengths the generator chose, solving each mass independently",
        xl="airframe mass (kg)", yl="leg length $L_1$ chosen by the generator (mm)",
        R="Scaling exponent b: machine vs biology vs theory",
        xl2="scaling exponent b", note="dots = median over all feasible designs at that mass; line = fit",
        bird="waterbirds (measured, phylogenetically corrected)", geo="geometric similarity",
        ela="elastic similarity", mh="machine · rigid ground", ms="machine · soft ground",
        seeds="error bar = two random seeds",
        cap="On the soft ground birds actually land on, the machine picks almost the bird's exponent. "
            "Only on rigid ground does it flatten.")}[LANG]

A = json.load(open(f"{U}/outputs/v23_e20_r12/e20_bio.json"))
def load(d):
    Z = np.load(f"{U}/{d}/e20_bio_raw.npz", allow_pickle=True)
    return Z, Z["m_all"]
def fit(c, d):
    Z, m = load(d)
    if c + "__x7" not in Z: return None
    L1 = Z[c + "__x7"][..., 0]; ok = Z[c + "__ok"]
    xs, ys = [], []
    for i in range(len(m)):
        s = L1[i][ok[i]]
        if len(s) >= 5: xs.append(np.log10(m[i])); ys.append(np.log10(np.median(s)))
    if len(xs) < 4: return None
    b, a = np.polyfit(xs, ys, 1)
    return np.array(xs), np.array(ys), b, a, float(np.corrcoef(xs, ys)[0, 1] ** 2)

CS = [("concrete1.2", HARD, T["mh"], "硬地 k_c=1e6" if LANG == "cn" else "rigid, $k_c$=1e6"),
      ("turf1.2",     SOFT, T["ms"], "草地 k_c=1e5" if LANG == "cn" else "turf, $k_c$=1e5"),
      ("wetsand1.2",  "#7a4bbf", T["ms"], "湿沙 k_c=5e4" if LANG == "cn" else "wet sand, $k_c$=5e4")]
fig = plt.figure(figsize=(15.0, 6.3))
gs = fig.add_gridspec(1, 2, width_ratios=[1.25, 1], wspace=.22)

# ---------- 左：双对数上自己变成直线 ----------
axL = fig.add_subplot(gs[0, 0])
B_A, B_B = 0.6163, 0.3665     # 水鸟 PGLS（a 用 g 为单位，这里换算到 kg）
res = {}
for c, col, _, lab in CS:
    r = fit(c, "outputs/v23_e20_r12")
    if r is None: continue
    xs, ys, b, a, r2 = r; res[c] = (b, r2)
    axL.plot(10 ** xs, 10 ** ys, "o", color=col, ms=7.5, mec="k", mew=.6, alpha=.9, zorder=4)
    xx = np.linspace(xs.min() - .05, xs.max() + .05, 50)
    axL.plot(10 ** xx, 10 ** (a + b * xx), color=col, lw=2.4, zorder=3,
             label=f"{lab}   b = {b:.3f}   R² = {r2:.3f}")
mm = np.linspace(np.log10(1.9), np.log10(42), 50)
axL.plot(10 ** mm, 10 ** (B_A + B_B * (mm + 3)), color=BIRD, lw=3.2, ls="--", zorder=5,
         label=f"{T['bird']}   b = {B_B:.3f}")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xticks([2, 5, 10, 20, 40]); axL.set_xticklabels(["2", "5", "10", "20", "40"])
axL.set_yticks([60, 100, 150, 200, 300]); axL.set_yticklabels(["60", "100", "150", "200", "300"])
axL.set_xlabel(T["xl"], fontsize=13); axL.set_ylabel(T["yl"], fontsize=13)
axL.set_title(T["L"], fontsize=14, fontweight="bold", loc="left", pad=10)
axL.grid(alpha=.25, which="both"); axL.legend(loc="upper left", fontsize=11, framealpha=.95)
axL.text(.985, .03, T["note"], transform=axL.transAxes, ha="right", fontsize=10.5, color=GRY)
for s in ("top", "right"): axL.spines[s].set_visible(False)

# ---------- 右：指数一维对比 ----------
axR = fig.add_subplot(gs[0, 1])
def pair(c):
    a0 = fit(c, "outputs/v23_e20_r12"); a1 = fit(c, "outputs/v23_e20_r12_s1")
    v = [x[2] for x in (a0, a1) if x]
    return float(np.mean(v)), float(abs(v[0] - v[1]) / 2 if len(v) == 2 else 0)
_L = {"cn": dict(hard="机器 · 硬地 (k_c=1e6)", turf="机器 · 草地 (k_c=1e5)",
                 sand="机器 · 湿沙 (k_c=5e4)", bird="水鸟实测（亲缘校正）"),
      "en": dict(hard="machine · rigid ground", turf="machine · turf",
                 sand="machine · wet sand", bird="waterbirds (measured)")}[LANG]
ITEMS = [(T["ela"], 0.250, 0, GRY, "theory"),
         (_L["hard"], *pair("concrete1.2"), HARD, "m"),
         (T["geo"], 0.333, 0, GRY, "theory"),
         (_L["turf"], *pair("turf1.2"), SOFT, "m"),
         (_L["sand"], *pair("wetsand1.2"), "#7a4bbf", "m"),
         (_L["bird"], 0.3665, 0, BIRD, "bio")]
ITEMS = sorted(ITEMS, key=lambda z: z[1])
y = np.arange(len(ITEMS))
for i, (lab, v, e, col, kind) in enumerate(ITEMS):
    axR.barh(i, v, color=col, height=.44, alpha=(.35 if kind == "theory" else .92),
             hatch=("//" if kind == "theory" else None), edgecolor=col, lw=1.4)
    if e > 0: axR.errorbar(v, i, xerr=e, color="k", capsize=4, lw=1.6)
    axR.text(v + e + .005, i, f"{v:.3f}", va="center", fontsize=12.5, fontweight="bold", color=col)
# 标签写在柱子上方（柱子很长，写左边会被压住）
for i, (lab, v, e, col, kind) in enumerate(ITEMS):
    axR.text(0.2035, i + .335, lab, va="bottom", ha="left", fontsize=12,
             fontweight=("normal" if kind == "theory" else "bold"), color=col)
axR.set_yticks([]); axR.set_ylim(-.62, len(ITEMS) - .22)
axR.set_xlim(0.20, 0.42); axR.set_xlabel(T["xl2"], fontsize=13)
axR.set_title(T["R"], fontsize=14, fontweight="bold", loc="left", pad=10)
axR.grid(axis="x", alpha=.25)
axR.text(.985, .015, T["seeds"], transform=axR.transAxes, ha="right", fontsize=10.5, color=GRY)
for s in ("top", "right"): axR.spines[s].set_visible(False)
fig.suptitle(T["sup"], fontsize=16.5, fontweight="bold", y=1.02)
fig.text(.5, -.045, T["cap"], ha="center", fontsize=13, fontweight="bold", color=BIRD)
out = f"{OUT}/{LANG}_F_agree.png"
fig.savefig(out, dpi=165, bbox_inches="tight")
print("→", out, " | " + " · ".join(f"{c}: b={res[c][0]:.3f} R²={res[c][1]:.3f}" for c in res))

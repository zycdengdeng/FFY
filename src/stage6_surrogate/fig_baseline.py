# -*- coding: utf-8 -*-
"""与 DJI Agras T30 式滑橇起落架的对比（30 kg / 1.2 m/s / 硬地 / 仅垂直着陆）。

对照构型：按 DJI Agras T30 的起落架构型重建 —— 2 组拱形支架
（4 根外张 18° 的斜撑，h = 0.30 m）+ 2 根滑橇管（0.90 m）+ 4 个橡胶脚垫（φ40 × 厚 20 mm）。
支架用与我们的腿完全相同的打印材料（碳纤维尼龙）、薄壁圆管（壁厚 0.1D）、安全系数 2.0，
按轴压 + 欧拉屈曲取最小可行管径；竖向行程几乎全部来自橡胶脚垫
（k_pad = 4·E_r·A/t，E_r ≈ 10 MPa 含形状系数），与支架竖向刚度串联。
峰值由线性弹簧能量守恒解出：F = 2mg + m·v0²/s，即 a/g = 2 + v0²/(g·s)。
右图的灰带给出脚垫在 φ30–70 mm × 厚 10–70 mm 内变化时的可达行程范围。
我们的 A0 数据取自 outputs/anim_b/b_compare_hard.json（同工况落震仿真）。

用法: python src/stage6_surrogate/fig_baseline.py [--lang cn|en]
"""
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
OURS, RIG, PALE, GRY, SRC, RED = "#2E7D5B", "#1b6ca8", "#bcd6ea", "#555", "#8a8a8a", "#c0392b"

# ---------------- 工况与材料 ----------------
g, m, v0 = 9.81, 30.0, 1.2
A_C, I_C, SF = 0.2827, 0.02898, 2.0                 # 薄壁管面积/惯矩系数（壁厚 0.1D）
RHO, SY, E = 1180., 70e6, 6e9                       # cfnylon，与我们的腿同材料
H, ALPHA = 0.30, np.radians(18.)                    # 支架高度、外张角
LS = H / np.cos(ALPHA)
NST, NSK, LSK, NPAD = 4, 2, 0.90, 4
E_R, RHO_R = 10e6, 1200.                            # 橡胶脚垫等效模量 / 密度
DPAD, TPAD = 0.040, 0.020                           # 钉死的脚垫尺寸
DGRID = np.arange(0.012, 0.0605, 0.002)

def frame_k(D):                                     # 4 根斜撑的竖向刚度（轴向为主）
    return NST * (E * A_C * D ** 2 / LS) * np.cos(ALPHA) ** 2

def size_frame(F):                                  # 轴压 + 屈曲，取最小可行管径
    P = F / (NST * np.cos(ALPHA))
    for D in DGRID:
        if P / (A_C * D ** 2) <= SY / SF and P <= (np.pi ** 2 * E * I_C * D ** 4 / LS ** 2) / SF:
            return D
    return None

def solve(dp, t):
    A = np.pi / 4 * dp ** 2
    kp = NPAD * E_R * A / t
    D = 0.030
    for _ in range(40):                             # 刚度-受力-选管径 迭代
        k = 1. / (1. / kp + 1. / frame_k(D))
        s = (m * g + np.sqrt((m * g) ** 2 + k * m * v0 ** 2)) / k
        Dn = size_frame(k * s)
        if Dn is None: return None
        if abs(Dn - D) < 1e-9: D = Dn; break
        D = Dn
    if s / t > 0.25: return None                    # 橡胶压缩应变超线性段
    mass = RHO * A_C * D ** 2 * (NST * LS + NSK * LSK) + NPAD * A * t * RHO_R
    return dict(dp=dp * 1e3, t=t * 1e3, D=D * 1e3, s=s * 1e3, a=k * s / (m * g), mass=mass * 1e3)

SK = solve(DPAD, TPAD)
ALT = [r for r in (solve(dp, t) for dp in (0.030, 0.040, 0.050, 0.060, 0.070)
                   for t in (0.010, 0.015, 0.020, 0.030, 0.040, 0.050, 0.060, 0.070))
       if r is not None and r["t"] <= r["dp"]]
SLO, SHI = min(r["s"] for r in ALT), max(r["s"] for r in ALT)

B = json.load(open(f"{U}/outputs/anim_b/b_compare_hard.json"))["meta"]["bio"]
OA, OS, OM = B["peak_g"], B["leg_stroke_mm"], B["leg_mass_g"]
S10 = 1e3 * v0 ** 2 / (g * 8.)                      # a/g = 10 时所需行程

SMAX = 24.0                                         # 验收行程上限（feasible_v2: s_max = 24 mm）

T = {"cn": dict(
    sup="与 DJI Agras T30 式滑橇起落架的对比：30 kg 机身 · 1.2 m/s · 硬地 · 仅垂直着陆",
    lt="① 质量相当，峰值过载相差 4 倍", rt="② 峰值过载由可用行程决定，与构型无关",
    bars=[("峰值过载 (g)", SK["a"], OA, "{:.1f}", (0., 10.), "合格区 ≤ 10 g"),
          ("可用行程 (mm)", SK["s"], OS, "{:.1f}", (S10, SMAX), f"合格区 {S10:.0f}–{SMAX:.0f} mm"),
          ("起落架总重 (g)", SK["mass"], OM, "{:.0f}", None, None)],
    nb=f"滑橇（Agras T30 式，脚垫 φ{SK['dp']:.0f}×{SK['t']:.0f} mm）", nb2="滑橇（T30 式）",
    no="我们的 A0（仿生腿）",
    xl2="可用行程 (mm)", yl2="峰值过载 (g)", lim="10 g 上限",
    curve="线性弹簧界: a/g = 2 + v0²/(g·s)",
    band=f"任何脚垫尺寸\n行程只有 {SLO:.0f}–{SHI:.0f} mm",
    okwin=f"合格窗口\n{S10:.0f}–{SMAX:.0f} mm",
    a0lab=f"A0：{OS:.1f} mm / {OA:.1f} g",
    srch="数据来源",
    src=[("滑橇构型与尺度", "DJI Agras T30 公开规格（整机 2858×2685×790 mm、空机 26.4 kg）；"
                            "据此取支架高 0.30 m、滑橇管长 0.90 m、外张 18°"),
         ("结构材料与校核", "与本工作的腿相同：碳纤维尼龙、薄壁圆管（壁厚 0.1D）、安全系数 2.0；"
                            "斜撑按轴压 + 欧拉屈曲定尺"),
         ("橡胶脚垫",       "φ40 × 厚 20 mm、等效模量 10 MPa —— 橡胶隔振件典型量级（量级估计，非厂家数据）"),
         ("验收判据",       "峰值 ≤ 10 g 且行程 ≤ 24 mm（本工作规格）；10 g 对应的最小行程 18 mm 由能量守恒反解")]),
     "en": dict(
    sup="Comparison with DJI Agras T30-style skid gear: 30 kg airframe · 1.2 m/s · rigid ground · vertical landing only",
    lt="(1) Comparable mass, 4x difference in peak deceleration",
    rt="(2) Peak deceleration is set by available stroke",
    bars=[("peak deceleration (g)", SK["a"], OA, "{:.1f}", (0., 10.), "pass: ≤ 10 g"),
          ("available stroke (mm)", SK["s"], OS, "{:.1f}", (S10, SMAX), f"pass: {S10:.0f}–{SMAX:.0f} mm"),
          ("landing-gear mass (g)", SK["mass"], OM, "{:.0f}", None, None)],
    nb=f"skid gear (T30-style, pads φ{SK['dp']:.0f}×{SK['t']:.0f} mm)", nb2="T30-style skid",
    no="our A0 (bio-inspired leg)",
    xl2="available stroke (mm)", yl2="peak deceleration (g)", lim="10 g limit",
    curve="linear-spring bound: a/g = 2 + v$_0^2$/(g·s)",
    band=f"any pad size:\nonly {SLO:.0f}–{SHI:.0f} mm",
    okwin=f"pass window\n{S10:.0f}–{SMAX:.0f} mm",
    a0lab=f"A0: {OS:.1f} mm / {OA:.1f} g",
    srch="Sources",
    src=[("geometry", "DJI Agras T30 published specs (2858x2685x790 mm unfolded, 26.4 kg empty);\n"
                      "struts 0.30 m high, skid tubes 0.90 m, splayed 18 degrees"),
         ("material", "same as our leg: CF-nylon, thin-wall tube (t = 0.1D), safety factor 2.0;\n"
                      "struts sized against axial stress and Euler buckling"),
         ("rubber pads", "phi 40 x 20 mm thick, effective modulus 10 MPa - typical for rubber isolators\n"
                         "(order-of-magnitude estimate, not manufacturer data)"),
         ("acceptance", "peak <= 10 g and stroke <= 24 mm (this work); the 18 mm minimum for 10 g\n"
                        "follows from energy conservation")])}[LANG]

# ---------------- 布局（显式定位，避免自动布局互相压字）----------------
fig = plt.figure(figsize=(15.8, 6.5))
BW, BX = .52, (0., .85)
axb = [fig.add_axes([x, .40, .132, .385]) for x in (.048, .228, .408)]
axs = fig.add_axes([.048, .030, .495, .300]); axs.axis("off")
Bx  = fig.add_axes([.620, .155, .360, .620])

# ---------------- ① 三个指标的直接对照（绿色 = 验收合格区）----------------
for A, (lab, vb, vo, fmt, win, wtxt) in zip(axb, T["bars"]):
    top = max(vb, vo, (win[1] if win else 0)) * 1.30
    if win:
        A.axhspan(win[0], win[1], color=OURS, alpha=.11, lw=0, zorder=1)
        for yv in win:
            if 0 < yv < top:
                A.axhline(yv, color=OURS, lw=1.5, ls="--", zorder=2)
        A.text(1.22, (win[0] + min(win[1], top)) / 2, wtxt, color="#1d5a41", fontsize=10.5,
               ha="left", va="center", fontweight="bold", zorder=5)
    A.bar([BX[0]], [vb], width=BW, color=PALE, ec=RIG, lw=1.6, zorder=3)
    A.bar([BX[1]], [vo], width=BW, color=OURS, ec="#1d5a41", lw=1.6, alpha=.92, zorder=3)
    A.set_ylim(0, top); A.set_xlim(-.52, 2.45)
    for x, v, c in ((BX[0], vb, RIG), (BX[1], vo, "#1d5a41")):
        mk = "" if not win else ("  ✓" if win[0] <= v <= win[1] else "  ×")
        yt = v + top * .028                                   # 贴柱顶
        if win and win[1] < top and abs(v - win[1]) < top * .075:
            yt = win[1] + top * .035                          # 太贴合格线时抬到线上方
        A.text(x, yt, fmt.format(v) + mk, ha="center", va="bottom",
               fontsize=14.5, fontweight="bold", color=c, zorder=5)
    A.set_title(lab, fontsize=13, fontweight="bold", pad=9)
    A.set_xticks([]); A.grid(axis="y", alpha=.22)
    for sp in ("top", "right", "bottom"): A.spines[sp].set_visible(False)

h = [plt.Rectangle((0, 0), 1, 1, fc=PALE, ec=RIG, lw=1.6),
     plt.Rectangle((0, 0), 1, 1, fc=OURS, ec="#1d5a41", lw=1.6)]
fig.legend(h, [T["nb"], T["no"]], loc="center left", bbox_to_anchor=(.048, .845),
           ncol=2, fontsize=12, frameon=False)
fig.text(.048, .905, T["lt"], fontsize=14.5, fontweight="bold", ha="left", va="center")

# ---------------- 数据来源 ----------------
axs.set_xlim(0, 1); axs.set_ylim(0, 1)
axs.text(0, 1.06, T["srch"], fontsize=11.5, fontweight="bold", color=GRY, va="top")
y = .86
for k, v in T["src"]:
    axs.text(.0, y, k, fontsize=10, color=GRY, va="top", fontweight="bold")
    axs.text(.185, y, v, fontsize=8.9, color=SRC, va="top", style="italic", linespacing=1.55)
    y -= .25

# ---------------- ② 行程 – 峰值过载 ----------------
ss = np.linspace(2.0, 45, 400)
Bx.axvspan(S10, SMAX, color=OURS, alpha=.13, lw=0, zorder=1)
Bx.axvspan(SLO, SHI, color=RIG, alpha=.15, lw=0, zorder=1)
Bx.plot(ss, 2. + v0 ** 2 / (g * ss * 1e-3), color="#333", lw=3, zorder=4, label=T["curve"])
Bx.axhline(10, color=RED, lw=2, ls="--", zorder=3)
Bx.text(44.5, 10.5, T["lim"], color=RED, fontsize=11.5, va="bottom", ha="right")
Bx.scatter([SK["s"]], [SK["a"]], s=230, marker="o", c=PALE, ec=RIG, lw=2.0, zorder=6)
Bx.annotate(T["nb2"], xy=(SK["s"], SK["a"]), xytext=(9.5, 37.5),
            fontsize=11.5, fontweight="bold", color=RIG,
            arrowprops=dict(arrowstyle="->", lw=1.6, color=RIG), zorder=7)
Bx.text(9.5, 32.5, T["band"], color=RIG, fontsize=11, va="top")
Bx.text(SMAX + .8, 26.5, T["okwin"], color="#1d5a41", fontsize=11.5, fontweight="bold", va="top")
Bx.scatter([OS], [OA], s=460, marker="*", c=OURS, ec="k", lw=1.3, zorder=6)
Bx.annotate(T["a0lab"], xy=(OS, OA), xytext=(28.5, 16.0),
            fontsize=11.5, fontweight="bold", color="#1d5a41",
            arrowprops=dict(arrowstyle="->", lw=1.6, color=OURS), zorder=7)
Bx.set_xlim(0, 45); Bx.set_ylim(0, 42)
Bx.set_xlabel(T["xl2"], fontsize=13); Bx.set_ylabel(T["yl2"], fontsize=13)
Bx.set_title(T["rt"], fontsize=14.5, fontweight="bold", loc="left", pad=10)
Bx.legend(loc="upper right", fontsize=10.8, framealpha=.95)
Bx.grid(alpha=.22)
for sp in ("top", "right"): Bx.spines[sp].set_visible(False)

fig.suptitle(T["sup"], fontsize=(16.5 if LANG == "cn" else 15.2), fontweight="bold", y=.985)
out = f"{OUT}/{LANG}_F_baseline.png"
fig.savefig(out, dpi=165)
print("->", out)
print(f"  T30 式滑橇: 行程 {SK['s']:.2f} mm, 峰值 {SK['a']:.1f} g, 总重 {SK['mass']:.0f} g")
print(f"  我们的 A0 : 行程 {OS:.1f} mm, 峰值 {OA:.2f} g, 总重 {OM:.0f} g")
print(f"  合格窗口 {S10:.1f}-{SMAX:.1f} mm; 脚垫全范围可达 {SLO:.1f}-{SHI:.1f} mm")

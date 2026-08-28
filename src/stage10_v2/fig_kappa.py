# -*- coding: utf-8 -*-
"""κ 的定义图:为什么关节刚度要写成无量纲的,以及它到底在量什么。

(a) 定义:弹簧扭矩 / 重力扭矩,两者都是 N·m,相除无量纲
(b) 三个关节的设计盒与网络实际选点
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
import matplotlib.font_manager  # noqa: F401
import sys
sys.path.insert(0, "/tmp/pipeline_code/src/stage10_v2")
try:
    from cjkfont import setup as _c
    _c()
except Exception:
    for _n in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        if _n in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [_n] + plt.rcParams["font.sans-serif"]
            break
plt.rcParams["axes.unicode_minus"] = False

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8b8a85"; CRIM = "#8E2A34"
CA, CK, CH = "#1baf7a", "#eb6834", "#2a78d6"          # 踝 / 膝 / 髋

L1, R2, R3 = 100.0, 1.80, 1.06
L2, L3 = R2 * L1, R3 * L1
a1 = np.radians(50.0); a2 = a1 + np.radians(60); a3 = a2 - np.radians(90)
d = lambda t: np.array([np.cos(t), np.sin(t)])
F = np.array([0.0, 0.0]); A = F + L1*d(a1); K = A + L2*d(a2); H = K + L3*d(a3)
LLEG = L1 + L2 + L3                                    # 全腿长(三段之和)

fig = plt.figure(figsize=(13.6, 6.6), dpi=250)
fig.patch.set_facecolor(SURF)
gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.16], height_ratios=[1.0, .34],
                      left=.045, right=.975, top=.87, bottom=.175,
                      wspace=.14, hspace=.06)

# ---------------------------------------------------------------- (a) 定义
ax = fig.add_subplot(gs[0, 0]); ax.set_facecolor(SURF)
ax.set_aspect("equal"); ax.axis("off")
SH = -40.0                                             # 整条腿左移,右侧留给标注
F = np.array([SH, 0.0]); A = F + L1*d(a1); K = A + L2*d(a2); H = K + L3*d(a3)
ax.set_xlim(-185, 250); ax.set_ylim(-52, 340)
ax.fill_between([-185, 250], -52, 0, color="#e8e4dc", lw=0)
ax.plot([-185, 250], [0, 0], color="#9a948a", lw=1.5)

for p0, p1 in ((F, A), (A, K), (K, H)):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color="#b9b3a8", lw=7,
            solid_capstyle="round", zorder=3)
ax.add_patch(Circle(F, 17, color="#b9b3a8", zorder=3))
ax.add_patch(FancyBboxPatch((H[0]-56, H[1]+16), 112, 30,
                            boxstyle="round,pad=3,rounding_size=8",
                            fc="#c8c3b8", ec="none", zorder=2))
ax.text(H[0], H[1]+31, "机体 m", fontsize=10, color="#4a463f", ha="center", va="center")

th = np.linspace(0, 4.6*np.pi, 300); rr = np.linspace(3, 24, 300)
ax.plot(H[0] + rr*np.cos(th), H[1] + rr*np.sin(th), color=CH, lw=1.9, zorder=5)
ax.add_patch(Circle(H, 6.5, fc="#ffffff", ec=CH, lw=1.6, zorder=6))
ax.plot([H[0]+22, H[0]+58], [H[1]-14, H[1]-42], color=CH, lw=.8, alpha=.6)
ax.text(H[0]+62, H[1]-48, "关节扭簧\n弹簧扭矩 $k\\cdot\\theta$", fontsize=11,
        color=CH, ha="left", va="center", linespacing=1.5, fontweight="bold")

ax.add_patch(FancyArrowPatch((H[0]+112, H[1]+44), (H[0]+112, H[1]-20),
                             color=CRIM, lw=2.2,
                             arrowstyle="-|>,head_width=4.5,head_length=8"))
ax.text(H[0]+122, H[1]+12, "$m\\,g$", fontsize=13, color=CRIM, va="center")
ax.annotate("", xy=(-148, H[1]), xytext=(-148, 0),
            arrowprops=dict(arrowstyle="<->", color=CRIM, lw=1.6))
ax.plot([-148, F[0]-17], [0, 0], color=CRIM, lw=.7, alpha=.5)
ax.plot([-148, H[0]-30], [H[1], H[1]], color=CRIM, lw=.7, alpha=.5)
ax.text(-158, H[1]/2 + 26, "力臂 = 全腿长\n$L_{leg}=L_1{+}L_2{+}L_3$", fontsize=10.5,
        color=CRIM, ha="right", va="center", linespacing=1.6, fontweight="bold")
ax.text(-158, H[1]/2 - 48, "重力扭矩\n$m\\,g\\,L_{leg}$", fontsize=11.5, color=CRIM,
        ha="right", va="center", linespacing=1.5, fontweight="bold")
ax.set_title("(a) κ = 弹簧扭矩 ÷ 重力扭矩", fontsize=13, color=INK,
             loc="left", pad=8, fontweight="bold")

axt = fig.add_subplot(gs[1, 0]); axt.axis("off")
axt.text(.5, .96,
         r"$\kappa=\dfrac{k}{m\,g\,L_{leg}}"
         r"=\dfrac{[\mathrm{N\cdot m/rad}]}{[\mathrm{N\cdot m}]}"
         r"\;\Rightarrow\;$无量纲",
         fontsize=14, color=INK, ha="center", va="top")
axt.text(.5, .40,
         "θ = 1 rad 时弹簧给出的力矩,是整机重量挂在一个全腿长上的几倍。\n"
         "静态下垂角 ≈ 1/κ 弧度:κ = 2 → 约 29°(软)　κ = 17 → 约 3.4°(硬)",
         fontsize=10, color=INK2, ha="center", va="top", linespacing=1.8)

# ---------------------------------------------------------------- (b) 盒子与落点
ax = fig.add_subplot(gs[:, 1]); ax.set_facecolor(SURF)
ROWS = [("κ 髋", CH, (6.0, 32.0), (10.71, 17.70), 16.0),
        ("κ 膝", CK, (1.5, 8.0), (3.61, 5.64), 4.0),
        ("κ 踝", CA, (1.5, 8.0), (1.87, 3.35), 4.0)]
for i, (nm, col, box, ai, v1) in enumerate(ROWS):
    ax.barh(i, box[1]-box[0], left=box[0], height=.42, color=col, alpha=.16,
            lw=0, zorder=2)
    ax.plot([box[0], box[0]], [i-.21, i+.21], color=col, lw=2.0, zorder=3)
    ax.plot([box[1], box[1]], [i-.21, i+.21], color=col, lw=2.0, zorder=3)
    ax.barh(i, ai[1]-ai[0], left=ai[0], height=.42, color=col, alpha=.95,
            lw=0, zorder=4)
    ax.plot([v1], [i], "|", color="#333333", ms=17, mew=2.0, zorder=6)
    ax.text(box[0]*.90, i, nm, fontsize=12, color=col, ha="right", va="center",
            fontweight="bold")
    ax.text(ai[1]*1.08, i+.005, f"{ai[0]:.1f}–{ai[1]:.1f}", fontsize=9,
            color=col, va="center", fontweight="bold")
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
HND = [Patch(fc="#f0eeea", ec="#6b665e", lw=1.6, label="设计盒(量纲规则 4/4/16 向两侧展开约 2×)"),
       Patch(fc="#6b665e", ec="none", label="网络在 0.5–120 kg 上实际选的范围"),
       Line2D([], [], color="#111111", marker="|", ls="none", ms=15, mew=2.2,
              label="量纲规则的默认值(踝/膝 κ=4,髋 4κ=16)")]
ax.set_xscale("log"); ax.set_xlim(1.05, 46)
ax.set_xticks([1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32])
ax.set_xticklabels(["1.5", "2", "3", "4", "6", "8", "12", "16", "24", "32"])
from matplotlib.ticker import NullLocator, NullFormatter
ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.set_ylim(-.60, len(ROWS)+.62); ax.set_yticks([])
ax.set_xlabel("κ  (对数轴)　　←  软 · 下垂大 · 行程长　　　硬 · 下垂小 · 行程短  →",
              fontsize=10.5, color=INK2)
ax.set_title("(b) 三个关节的设计盒与网络实际落点", fontsize=13, color=INK,
             loc="left", pad=10, fontweight="bold")
ax.grid(axis="x", alpha=.18, lw=.6, zorder=0)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.tick_params(colors=INK2, labelsize=9)
ax.legend(handles=HND, fontsize=9, frameon=False, loc="upper right",
          bbox_to_anchor=(1.005, 1.02), handlelength=1.9, handleheight=1.15,
          labelspacing=.85, borderpad=.2)
ax.text(1.10, -.46, "κ踝 贴着下界跑(盒内 6%–28%)→ 最优可能在盒外,待做边界扫描",
        fontsize=8.8, color=CRIM, va="center")

fig.suptitle("关节刚度为什么写成无量纲的 κ:让 1 kg 与 12 kg 用同一个设计盒",
             fontsize=14.5, color=CRIM, x=.045, ha="left", y=.962)
fig.text(.045, .098,
         "κ 相同 = 相对刚度相同。若直接用绝对刚度 k(N·m/rad),1 kg 合适的取值\n"
         "到 12 kg 会差十几倍,一个盒子盖不住整个体重区间。\n\n"
         "髋取 4κ 的依据:课题组前期实测(触水后髋大幅弯曲、膝踝稳定),且本工作\n"
         "的结构定尺独立算出髋关节峰值力矩为膝/踝的 3–4 倍。",
         fontsize=9.2, color=INK2, va="top", linespacing=1.6)
fig.text(.50, .052,
         "注:静态下垂角 ≈ 1/κ 为量级估计(真实力臂非整条腿长);\n"
         "且因阻尼按 c = τ·k 设定,κ 同时也在设定阻尼 —— 冲击时段实际以阻尼项为主。",
         fontsize=8.4, color=MUTED, va="top", linespacing=1.65)
fig.savefig("/tmp/seg/fig_kappa.png", facecolor=SURF, bbox_inches="tight")
print("→ /tmp/seg/fig_kappa.png")

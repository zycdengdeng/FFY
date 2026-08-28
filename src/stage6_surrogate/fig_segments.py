# -*- coding: utf-8 -*-
"""三段腿骨的定义图:给非鸟类背景的听众解释 L1 / r2 / r3 到底指什么。

姿态用的是仿真里真实的触地构型(跗跖 50°、踝 120°、膝 90°),不是随手画的示意,
所以这张图同时也交代了"我们是按什么姿态摔的"。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Circle
import matplotlib.font_manager  # noqa: F401
import os, sys
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

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8b8a85"
CRIM = "#8E2A34"
C1, C2, C3 = "#1baf7a", "#eb6834", "#2a78d6"      # 跗跖 / 胫跗 / 股骨

L1, R2, R3 = 100.0, 1.80, 1.06                    # 代表性取值(比例取雁鸭中位)
L2, L3 = R2 * L1, R3 * L1
a1 = np.radians(50.0)                             # 跗跖与地面夹角
a2 = a1 + np.radians(180 - 120)                   # 踝角 120°
a3 = a2 - np.radians(180 - 90)                    # 膝角 90°
d = lambda t: np.array([np.cos(t), np.sin(t)])
F = np.array([0.0, 0.0]); A = F + L1 * d(a1); K = A + L2 * d(a2); H = K + L3 * d(a3)

fig, ax = plt.subplots(figsize=(6.6, 5.4), dpi=250)
fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
ax.set_aspect("equal"); ax.axis("off")
ax.set_xlim(-130, 212); ax.set_ylim(-42, 352)

ax.fill_between([-130, 205], -42, 0, color="#e8e4dc", lw=0)
ax.plot([-130, 205], [0, 0], color="#9a948a", lw=1.6)

for p0, p1, col, lw in ((F, A, C1, 8.5), (A, K, C2, 7.5), (K, H, C3, 9.5)):
    ax.plot([p0[0], p1[0]], [p0[1], p1[1]], color=col, lw=lw,
            solid_capstyle="round", zorder=3)
ax.add_patch(Circle(F, 20, color=C1, zorder=4))                     # 蹼足
ax.add_patch(FancyBboxPatch((H[0] - 62, H[1] - 6), 124, 34,
                            boxstyle="round,pad=3,rounding_size=8",
                            fc="#c8c3b8", ec="none", zorder=2))
ax.text(H[0], H[1] + 12, "机体", fontsize=10.5, color="#4a463f",
        ha="center", va="center", zorder=5)
for p, nm in ((A, "踝"), (K, "膝"), (H, "髋")):
    ax.add_patch(Circle(p, 9.5, fc="#ffffff", ec="#555555", lw=1.4, zorder=5))
    ax.text(p[0] + 15, p[1] - 12, nm, fontsize=10, color=INK2, zorder=6)

def seg(p0, p1, col, tag, name, en, extra):
    mid = 0.5 * (p0 + p1)
    ax.annotate("", xy=(-96, p1[1]), xytext=(-96, p0[1]),
                arrowprops=dict(arrowstyle="<->", color=col, lw=1.5))
    ax.plot([-96, min(p0[0], p1[0])], [p0[1], p0[1]], color=col, lw=.7, alpha=.5)
    ax.plot([-96, min(p0[0], p1[0])], [p1[1], p1[1]], color=col, lw=.7, alpha=.5)
    ax.text(-104, 0.5 * (p0[1] + p1[1]),
            f"{tag}\n{name}\n{en}\n{extra}", fontsize=9, color=col,
            ha="right", va="center", linespacing=1.55, fontweight="bold")

seg(F, A, C1, "L1", "跗跖骨", "tarsometatarsus", "随体重走(见上式)")
seg(A, K, C2, "L2", "胫跗骨", "tibiotarsus", "r2 = L2/L1 ∈ [1.49, 2.09]")
seg(K, H, C3, "L3", "股骨", "femur", "r3 = L3/L1 ∈ [0.84, 1.28]")

ax.text(-126, 336, "几何先验 = 一个长度 + 两个比例", fontsize=12, color=CRIM,
        ha="left", va="center", fontweight="bold")
ax.text(208, 205,
        "L1 由异速律随体重平移\n"
        "r2 / r3 为固定窄带,不随体重变\n"
        "窄带取自 Watanabe 2017 (The Auk)\n"
        "91 种会飞雁鸭科的实测全距",
        fontsize=9, color=INK2, ha="right", va="top", linespacing=1.75)
ax.text(196, -30, "姿态为仿真实际触地构型:跗跖 50° · 踝 120° · 膝 90°",
        fontsize=8, color=MUTED, ha="right")
fig.tight_layout(pad=0.3)
fig.savefig("/tmp/seg/fig_leg_segments.png", facecolor=SURF, bbox_inches="tight")
print("→ /tmp/seg/fig_leg_segments.png")

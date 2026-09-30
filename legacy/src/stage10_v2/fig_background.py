# -*- coding: utf-8 -*-
"""课题背景页的三张示意图（非数据图，纯结构/现象示意）。
用法: python src/stage10_v2/fig_background.py [--lang cn|en]
产出: outputs/ppt_{lang}/{lang}_BG1_skid.png / BG2_load.png / BG3_design.png"""
import os, argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Rectangle, Circle, FancyArrowPatch, Arc

U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="cn", choices=["cn", "en"])
LANG = ap.parse_args().lang
OUT = f"{U}/outputs/ppt_{LANG}"; os.makedirs(OUT, exist_ok=True)
_have = {f.name for f in fm.fontManager.ttflist}
_cjk = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if f in _have), None)
plt.rcParams.update({"font.sans-serif": ([_cjk] if (_cjk and LANG == "cn") else []) + ["DejaVu Sans"],
                     "font.family": "sans-serif", "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})

CRIM, BLUE, GRY, INK = "#7F2D32", "#1b6ca8", "#8a8a8a", "#222222"
RUB, GND, STEEL = "#3d3d3d", "#b3ab9c", "#5c7f9e"
T = {"cn": dict(
        f1a="机身", f1b="斜撑", f1c="滑橇管", f1d="橡胶脚垫",
        f1n="全部缓冲行程只来自脚垫的压缩",
        f2a="冲击", f2b="机身 · 载荷 · 云台",
        f2n="着陆冲击几乎无衰减地传至机身与载荷",
        f3a="多长？", f3b="分几段？", f3c="关节多硬？",
        f3n="腿的尺度与刚度无规则可依，只能试凑"),
     "en": dict(
        f1a="airframe", f1b="strut", f1c="skid tube", f1d="rubber pad",
        f1n="all of the stroke comes from squashing the pad",
        f2a="impact", f2b="airframe · payload · gimbal",
        f2n="the shock reaches the payload almost undamped",
        f3a="how long?", f3b="how many links?", f3c="how stiff?",
        f3n="no rule for the scale or the stiffness of the leg")}[LANG]

def frame(nm):
    fig, ax = plt.subplots(figsize=(5.4, 3.05))
    ax.set_xlim(0, 100); ax.set_ylim(0, 56); ax.axis("off")
    ax.set_aspect("equal", adjustable="box")
    fig.subplots_adjust(left=.01, right=.99, top=.99, bottom=.10)
    return fig, ax

def note(fig, s):
    fig.text(.5, .035, s, ha="center", va="bottom", fontsize=12.5 if LANG == "cn" else 11.8,
             fontweight="bold", color=CRIM)

def ground(ax, y=8, x0=6, x1=94):
    ax.plot([x0, x1], [y, y], color=GND, lw=4, solid_capstyle="butt", zorder=2)
    for x in np.arange(x0 + 2, x1, 5.0):
        ax.plot([x, x - 3], [y, y - 4], color=GND, lw=1.4, alpha=.75, zorder=1)

# ---------------- ① 滑橇 + 橡胶脚垫 ----------------
fig, ax = frame("skid"); ground(ax, 8)
ax.add_patch(FancyBboxPatch((33, 38), 34, 12, boxstyle="round,pad=.6",
                            fc="#e8eef4", ec=STEEL, lw=2.0, zorder=4))
ax.text(50, 44, T["f1a"], ha="center", va="center", fontsize=12.5, color=INK, zorder=5)
for sx, ex in ((36, 22), (64, 78)):                       # 外张斜撑
    ax.plot([sx, ex], [38, 14], color=STEEL, lw=6, solid_capstyle="round", zorder=3)
ax.plot([16, 84], [14, 14], color=STEEL, lw=7, solid_capstyle="round", zorder=3)  # 滑橇管
for cx in (22, 78):                                        # 橡胶脚垫
    ax.add_patch(FancyBboxPatch((cx - 7, 8.4), 14, 5.4, boxstyle="round,pad=.25",
                                fc=RUB, ec="#1d1d1d", lw=1.2, zorder=5))
ax.annotate(T["f1b"], xy=(29, 26), xytext=(6, 34), fontsize=11.5, color=STEEL,
            fontweight="bold", ha="left", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.5, color=STEEL))
ax.annotate(T["f1c"], xy=(50, 14), xytext=(50, 25), fontsize=11.5, color=STEEL,
            fontweight="bold", ha="center", va="bottom",
            arrowprops=dict(arrowstyle="->", lw=1.5, color=STEEL))
ax.annotate(T["f1d"], xy=(78, 11), xytext=(95, 30), fontsize=12, color=CRIM,
            fontweight="bold", ha="right", va="center",
            arrowprops=dict(arrowstyle="->", lw=1.8, color=CRIM),
            bbox=dict(boxstyle="round,pad=.3", fc="#fbf0f1", ec=CRIM, lw=1.4))
note(fig, T["f1n"])
fig.savefig(f"{OUT}/{LANG}_BG1_skid.png", dpi=200); plt.close(fig)

# ---------------- ② 冲击传到机身与载荷 ----------------
fig, ax = frame("load"); ground(ax, 8)
ax.add_patch(FancyBboxPatch((30, 36), 40, 14, boxstyle="round,pad=.6",
                            fc="#e8eef4", ec=STEEL, lw=2.0, zorder=4))
ax.add_patch(Rectangle((44, 39.5), 12, 7, fc="#cfd8e2", ec=STEEL, lw=1.4, zorder=5))
ax.add_patch(Circle((50, 34), 3.4, fc="#cfd8e2", ec=STEEL, lw=1.4, zorder=5))
for sx, ex in ((36, 26), (64, 74)):
    ax.plot([sx, ex], [36, 14], color=STEEL, lw=6, solid_capstyle="round", zorder=3)
ax.plot([20, 80], [14, 14], color=STEEL, lw=7, solid_capstyle="round", zorder=3)
for cx in (26, 74):
    ax.add_patch(FancyBboxPatch((cx - 6, 8.4), 12, 5.4, boxstyle="round,pad=.25",
                                fc=RUB, ec="#1d1d1d", lw=1.2, zorder=5))
for cx in (26, 74):                                        # 冲击箭头往上贯穿
    ax.add_patch(FancyArrowPatch((cx, 15), (cx, 34), lw=0, zorder=6,
                                 arrowstyle="simple,head_width=9,head_length=8,tail_width=4",
                                 color=CRIM, alpha=.92))
ax.text(50, 22, T["f2a"], ha="center", va="center", fontsize=13, fontweight="bold",
        color=CRIM, zorder=7,
        bbox=dict(boxstyle="round,pad=.3", fc="white", ec=CRIM, lw=1.5))
xx = np.linspace(32, 68, 220)
ax.plot(xx, 52.5 + 1.5 * np.sin((xx - 32) / 2.6), color=CRIM, lw=2.0, alpha=.85, zorder=5)
note(fig, T["f2n"])
fig.savefig(f"{OUT}/{LANG}_BG2_load.png", dpi=200); plt.close(fig)

# ---------------- ③ 没有设计规则 ----------------
fig, ax = frame("design"); ground(ax, 8)
LEGS = [(20, [(0, 0), (-3.5, 12), (3.0, 21), (-1.5, 30)]),
        (50, [(0, 0), (5.0, 10), (-2.5, 20), (2.0, 27)]),
        (80, [(0, 0), (-2.0, 15), (4.5, 26), (0.0, 36)])]
for cx, pts in LEGS:
    P = np.array([(cx + dx, 8 + dy) for dx, dy in pts])
    ax.plot(P[:, 0], P[:, 1], "-o", color="#9aa6b2", lw=5, ms=7,
            mec="#6b7683", mew=1.2, solid_capstyle="round", zorder=3)
    ax.add_patch(Circle((P[0, 0], P[0, 1]), 2.2, fc="#eceff3", ec="#6b7683", lw=1.2, zorder=4))
for x, s in ((20, T["f3a"]), (50, T["f3b"]), (80, T["f3c"])):
    ax.text(x, 50, s, ha="center", va="center", fontsize=12, fontweight="bold",
            color=CRIM, zorder=6,
            bbox=dict(boxstyle="round,pad=.28", fc="#fbf0f1", ec=CRIM, lw=1.2))
note(fig, T["f3n"])
fig.savefig(f"{OUT}/{LANG}_BG3_design.png", dpi=200); plt.close(fig)
print("→", OUT, "BG1/BG2/BG3")

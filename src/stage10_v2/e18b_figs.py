# -*- coding: utf-8 -*-
"""E18b 出图:可行率随体重的走廊曲线 + 失效判据分解。

图 1  fig_e18b_corridor.png   四臂 × 多工况小倍数,可行率 vs 体重
图 2  fig_e18b_criteria.png   bio 臂:各判据违反率 vs 体重(轻端过载卡/重端结构卡)
图 3  fig_e18b_validity.png   模型有效性:deep_sink 占失效的比例 —— 重端结论能信多少

用法:  python src/stage10_v2/e18b_figs.py --dir outputs/v2_e18b
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager  # noqa: F401



try:
    from cjkfont import setup as _cjk_setup
    _CJK = _cjk_setup()
except Exception:                       # 独立运行、没有 cjkfont.py 时的退路
    _CJK = None
    for _c in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Microsoft YaHei", "SimHei"):
        if _c in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [_c] + plt.rcParams["font.sans-serif"]
            _CJK = _c
            break
plt.rcParams["axes.unicode_minus"] = False

SURF = "#fcfcfb"
INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8b8a85"
# 分类色板(已过 CVD 校验:最差相邻对 ΔE 9.1 protan / 22.9 normal)
ARMC = {"bio": "#2a78d6", "geo": "#eb6834", "elastic": "#1baf7a", "none": "#eda100"}
ARMN = {"bio": "bio  b=0.391", "geo": "geo  b=1/3",
        "elastic": "elastic  b=1/4", "none": "none  b=0"}
CRITC = {"gcap": "#2a78d6", "smax": "#eb6834", "slenderness": "#1baf7a",
         "massbudget": "#eda100", "deep_sink": "#e87ba4"}
CRITN = {"gcap": "过载超限", "smax": "行程超限", "slenderness": "细长比超限",
         "massbudget": "质量预算超限", "deep_sink": "模型失效(足端侵入超界)"}

M_VALID = 12.0      # 物理模型标定上限(kc≥5e4 在 m∈[1,12] 全程有效)
M_BIODATA = 10.7    # AVONET 水鸟样本最重个体(疣鼻天鹅)


def shade_extrap(ax, ms, note=True):
    """把外推区涂灰:>12kg 的一切结论都是外推,图上必须自己说出来。"""
    ax.axvspan(M_VALID, ms[-1] * 1.15, color="#000000", alpha=0.045, lw=0, zorder=0)
    ax.axvline(M_VALID, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)
    if note:
        ax.text(M_VALID * 1.12, 0.965, "外推区\n(物理模型标定上限 12 kg)",
                fontsize=7.4, color=MUTED, va="top", ha="left", linespacing=1.35)


def load(d):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "e18b_*.json"))):
        j = json.load(open(fp, encoding="utf-8"))
        out[j["cond"]] = j
    assert out, f"{d} 里没有 e18b_*.json"
    return out


def fig_corridor(J, outfp):
    conds = list(J)
    nc = len(conds)
    ncol = min(3, nc); nrow = int(np.ceil(nc / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.6 * ncol, 3.5 * nrow),
                             dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURF)
    for k, cname in enumerate(conds):
        j = J[cname]; ax = axes[k // ncol][k % ncol]
        ms = np.array(j["m_grid"])
        ax.set_facecolor(SURF)
        shade_extrap(ax, ms, note=(k == 0))
        # 直接标注:对比度校验给了 WARN,不能只靠图例辨认。四条线常在同一处
        # 达峰,所以标注点沿 x 错开固定分位,而不是各自取 argmax(必撞)。
        FRAC = {"bio": 0.80, "geo": 0.62, "elastic": 0.44, "none": 0.26}
        for arm, v in j["arms"].items():
            y = np.array(v["pooled"])
            ax.plot(ms, y, color=ARMC[arm], lw=2.0, zorder=3,
                    solid_capstyle="round")
            ax.plot(ms, y, "o", color=ARMC[arm], ms=3.2, zorder=3)
            i = int(round(FRAC.get(arm, 0.5) * (len(ms) - 1)))
            ax.annotate(ARMN[arm].split()[0], (ms[i], y[i]),
                        textcoords="offset points", xytext=(0, 8),
                        fontsize=8, color=ARMC[arm], ha="center",
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURF,
                                  ec="none", alpha=0.85))
        ax.set_xscale("log")
        ax.set_xlim(ms[0] * 0.85, ms[-1] * 1.18)
        ax.set_ylim(-0.03, 1.06)
        ax.set_title(j["label"], fontsize=10.5, color=INK, loc="left", pad=7)
        ax.grid(alpha=0.18, lw=0.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(MUTED); ax.spines[sp].set_linewidth(0.8)
        ax.tick_params(colors=INK2, labelsize=8.5)
        if k % ncol == 0:
            ax.set_ylabel("可行率(设计盒内合并 u)", fontsize=9.5, color=INK2)
        if k // ncol == nrow - 1:
            ax.set_xlabel("机体质量 m / kg", fontsize=9.5, color=INK2)
    for k in range(nc, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")

    h = [plt.Line2D([], [], color=ARMC[a], lw=2.4, label=ARMN[a]) for a in ARMC]
    fig.legend(handles=h, ncol=4, loc="lower center", frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 0.002))
    fig.suptitle(f"可行体重走廊:四条标度律 × {nc} 种工况    "
                 "(g_cap = 10 g · s_max = 24 mm · 纯物理,不加载网络)",
                 fontsize=12.5, color=INK, x=0.012, ha="left", y=0.985)
    fig.tight_layout(rect=[0, 0.045, 1, 0.955])
    fig.savefig(outfp, facecolor=SURF)
    print("→", outfp)


def fig_criteria(J, outfp, arm="bio", cond=None):
    cond = cond or ("turf1.2" if "turf1.2" in J else list(J)[0])
    j = J[cond]; ms = np.array(j["m_grid"])
    nu = len(j["u_grid"])
    rows = j["arms"][arm]["rows"]
    fig, ax = plt.subplots(figsize=(8.6, 4.8), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    shade_extrap(ax, ms)
    ax.plot(ms, j["arms"][arm]["pooled"], color=INK, lw=2.6, zorder=5,
            label="可行率")
    ax.annotate("可行率", (ms[int(np.argmax(j["arms"][arm]["pooled"]))],
                        max(j["arms"][arm]["pooled"])),
                textcoords="offset points", xytext=(0, 8), fontsize=9,
                color=INK, ha="center", fontweight="bold")
    for c, col in CRITC.items():
        y = np.mean([r[c] for r in rows], axis=0)      # 池合并 u
        ls = (0, (3, 2)) if c == "deep_sink" else "-"  # 模型失效用虚线:非物理失效
        ax.plot(ms, y, color=col, lw=2.0, ls=ls, zorder=4)
        i = int(np.argmax(y))
        if y[i] > 0.05:
            ax.annotate(CRITN[c].split("(")[0], (ms[i], y[i]),
                        textcoords="offset points", xytext=(0, 7), fontsize=8,
                        color=col, ha="center", fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(ms[0] * 0.85, ms[-1] * 1.18)
    ax.set_ylim(-0.03, 1.06)
    ax.set_xlabel("机体质量 m / kg", fontsize=10, color=INK2)
    ax.set_ylabel("违反率 / 可行率", fontsize=10, color=INK2)
    ax.grid(alpha=0.18, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    h = [plt.Line2D([], [], color=INK, lw=2.6, label="可行率")] + \
        [plt.Line2D([], [], color=CRITC[c], lw=2.0,
                    ls=((0, (3, 2)) if c == "deep_sink" else "-"),
                    label=CRITN[c]) for c in CRITC]
    ax.legend(handles=h, ncol=3, fontsize=8.6, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.13))
    ax.set_title(f"是什么卡住了它:各判据违反率沿体重的演化    "
                 f"({ARMN[arm]} · {j['label']})",
                 fontsize=12, color=INK, loc="left", pad=9)
    fig.text(0.012, 0.012,
             "每次评价可能同时违反多条,故各条不可叠加成 100%。"
             "虚线的 deep_sink 是罚接触模型侵入超界(非物理失效),"
             "它在重端的抬升标出了结论可信度的边界。",
             fontsize=7.8, color=MUTED)
    fig.tight_layout(rect=[0, 0.045, 1, 1])
    fig.savefig(outfp, facecolor=SURF)
    print("→", outfp)


def fig_validity(J, outfp, arm="bio"):
    """重端的'塌方'里有多少其实是模型失效?这张图决定 40kg 那个数能不能说。"""
    conds = list(J)
    fig, ax = plt.subplots(figsize=(8.6, 4.4), dpi=200)
    fig.patch.set_facecolor(SURF); ax.set_facecolor(SURF)
    ms = np.array(J[conds[0]]["m_grid"])
    shade_extrap(ax, ms)
    cols = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]
    for k, cname in enumerate(conds):
        j = J[cname]; rows = j["arms"][arm]["rows"]
        infeas = 1.0 - np.array(j["arms"][arm]["pooled"])
        ds = np.mean([r["deep_sink"] for r in rows], axis=0)
        share = np.where(infeas > 1e-6, ds / np.maximum(infeas, 1e-6), np.nan)
        ax.plot(np.array(j["m_grid"]), share, color=cols[k % len(cols)], lw=2.0)
        ok = ~np.isnan(share)
        if ok.any():
            i = int(np.nanargmax(share))
            ax.annotate(j["label"].split(" · ")[0], (j["m_grid"][i], share[i]),
                        textcoords="offset points", xytext=(0, 7), fontsize=8,
                        color=cols[k % len(cols)], ha="center", fontweight="bold")
    ax.set_xscale("log"); ax.set_xlim(ms[0] * 0.85, ms[-1] * 1.18)
    ax.set_ylim(-0.03, 1.06)
    ax.axhline(0.5, color=MUTED, lw=1.0, ls=":")
    ax.text(ms[0] * 0.9, 0.52, "过半失效是模型失效 → 该体重的结论不能用",
            fontsize=8, color=MUTED)
    ax.set_xlabel("机体质量 m / kg", fontsize=10, color=INK2)
    ax.set_ylabel("deep_sink 占全部失效的比例", fontsize=10, color=INK2)
    ax.grid(alpha=0.18, lw=0.6, zorder=0)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.set_title("模型有效性自检:重端的'塌方'有多少是模型失效而非物理失效",
                 fontsize=12, color=INK, loc="left", pad=9)
    fig.tight_layout()
    fig.savefig(outfp, facecolor=SURF)
    print("→", outfp)


def summary(J, arm="bio"):
    """打一份可以直接抄进汇报的数字。"""
    print("\n" + "=" * 78)
    print(f"{arm} 臂 · 池合并可行率(%) 随体重")
    ms = J[list(J)[0]]["m_grid"]
    print(f"{'工况':<22}" + "".join(f"{m:>6.1f}" for m in ms))
    for cname, j in J.items():
        print(f"{j['label']:<22}" +
              "".join(f"{v*100:>6.0f}" for v in j["arms"][arm]["pooled"]))
    print("\n各工况下 可行率≥50% 的最重体重(仅取 ≤12kg 有效区与外推区分别报):")
    for cname, j in J.items():
        p = np.array(j["arms"][arm]["pooled"]); mm = np.array(j["m_grid"])
        i = np.where(p >= 0.5)[0]
        s = f"{mm[i[-1]]:.1f} kg" if len(i) else "无"
        iv = np.where((p >= 0.5) & (mm <= M_VALID))[0]
        sv = f"{mm[iv[-1]]:.1f} kg" if len(iv) else "无"
        print(f"  {j['label']:<22} 全程 {s:>9}   限有效区(≤12kg) {sv:>9}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs/v2_e18b")
    ap.add_argument("--arm", default="bio")
    a = ap.parse_args()
    J = load(a.dir)
    fig_corridor(J, os.path.join(a.dir, "fig_e18b_corridor.png"))
    fig_criteria(J, os.path.join(a.dir, "fig_e18b_criteria.png"), arm=a.arm)
    fig_validity(J, os.path.join(a.dir, "fig_e18b_validity.png"), arm=a.arm)
    summary(J, arm=a.arm)

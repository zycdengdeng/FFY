# -*- coding: utf-8 -*-
"""E20 出图。

图 A  fig_e20_vs_random.png  盒子 vs 模型:E18b 随机抓 与 E20 条件生成 的可行率对照
图 B  fig_e20_metrics.png    四项力学指标随体重,附板簧/油气基准线
图 C  fig_e20_anchors.png    真实机型锚点上的成绩单(表格式,给汇报直接抄)

用法:  python src/stage10_v2/e20_figs.py --gen outputs/v2_e20 --rand outputs/v2_e18b
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

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8b8a85"
# 地形用颜色(已过 CVD 校验:最差相邻 ΔE 9.2 deutan / 27.6 normal),速度用线型
TERR = {"concrete": ("#2a78d6", "硬地"), "turf": ("#eb6834", "草地"),
        "wetsand": ("#1baf7a", "湿沙")}
LS = {"1.2": "-", "2.0": (0, (4, 2.5))}
M_TRAIN = 12.0        # cVAE 训练质量上限,也是物理模型标定上限
MET_SPEC = [
    ("peak_g", "峰值加速度 / g", [(10.0, "g_cap = 10 g", "#8E2A34")], (0, 14)),
    ("eta", "缓冲效率 η", [(0.50, "线性弹簧(碳纤板簧)= 0.50", "#8b8a85"),
                        (0.85, "油气支柱 0.8–0.9", "#2E8B57")], (0.3, 1.02)),
    ("cfe", "压溃力效率 cfe", [(0.50, "线性弹簧 = 0.50", "#8b8a85")], (0.3, 1.02)),
    ("leg_stroke_mm", "起落架行程 / mm", [(24.0, "s_max = 24 mm", "#8E2A34")], (0, 30)),
]


def split(cname):
    for t in TERR:
        if cname.startswith(t):
            return t, cname[len(t):]
    raise ValueError(cname)


def shade(ax, xhi):
    ax.axvspan(M_TRAIN, xhi, color="#000000", alpha=0.045, lw=0, zorder=0)
    ax.axvline(M_TRAIN, color=MUTED, lw=1.0, ls=(0, (4, 3)), zorder=1)


def load_gen(d):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "e20_*.json"))):
        j = json.load(open(fp, encoding="utf-8")); out[j["arm"]] = j
    assert out, f"{d} 里没有 e20_*.json"
    return out


def load_rand(d):
    out = {}
    for fp in sorted(glob.glob(os.path.join(d, "e18b_*.json"))):
        j = json.load(open(fp, encoding="utf-8")); out[j["cond"]] = j
    return out


# --------------------------------------------------------------- 图 A
def fig_vs_random(G, R, arm, outfp):
    g = G[arm]; ms = np.array(g["m_grid"]); conds = list(g["conds"])
    nc = len(conds); ncol = min(3, nc); nrow = int(np.ceil(nc / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4.7 * ncol, 3.6 * nrow),
                             dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURF)
    for k, cn in enumerate(conds):
        ax = axes[k // ncol][k % ncol]; c = g["conds"][cn]
        ax.set_facecolor(SURF); shade(ax, ms[-1] * 1.2)
        y = np.array(c["feas"])
        ax.plot(ms, y, color="#2a78d6", lw=2.4, zorder=4)
        ax.plot(ms, y, "o", color="#2a78d6", ms=3.4, zorder=4)
        i = int(round(0.72 * (len(ms) - 1)))
        ax.annotate("cVAE 生成", (ms[i], y[i]), textcoords="offset points",
                    xytext=(0, 9), fontsize=8.6, color="#2a78d6", ha="center",
                    fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=.85))
        if cn in R:                       # 同格的随机基线(bio 臂,池合并 u)
            yr = np.array(R[cn]["arms"].get(arm, R[cn]["arms"]["bio"])["pooled"])
            ax.plot(ms, yr, color=MUTED, lw=2.0, ls=(0, (4, 2.5)), zorder=3)
            j = int(round(0.34 * (len(ms) - 1)))
            ax.annotate("盒内随机抓", (ms[j], yr[j]), textcoords="offset points",
                        xytext=(0, -16), fontsize=8.6, color=MUTED, ha="center",
                        fontweight="bold", zorder=6,
                        bbox=dict(boxstyle="round,pad=0.15", fc=SURF, ec="none", alpha=.85))
            ax.fill_between(ms, yr, y, where=(y >= yr), color="#2a78d6",
                            alpha=0.10, lw=0, zorder=2)
        ax.set_xscale("log"); ax.set_xlim(ms[0] * .85, ms[-1] * 1.2)
        ax.set_ylim(-.03, 1.06)
        ax.set_title(c["label"], fontsize=10.5, color=INK, loc="left", pad=6)
        ax.grid(alpha=.18, lw=.6, zorder=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=8.5)
        if k % ncol == 0:
            ax.set_ylabel("可行率", fontsize=9.5, color=INK2)
        if k // ncol == nrow - 1:
            ax.set_xlabel("机体质量 m / kg", fontsize=9.5, color=INK2)
        if k == 0:
            ax.text(M_TRAIN * 1.15, 1.0, "外推区\n(训练只到 12 kg)", fontsize=7.4,
                    color=MUTED, va="top", linespacing=1.35)
    for k in range(nc, nrow * ncol):
        axes[k // ncol][k % ncol].axis("off")
    fig.suptitle(f"盒子 vs 模型:同工况、同体重、同判官,只差设计从哪来  "
                 f"（{arm} 臂 · {g['ckpt']} · 每格 {g['nz']} 个样本）",
                 fontsize=12.5, color=INK, x=.012, ha="left", y=.985)
    fig.text(.012, .012, "灰虚线=在生物先验盒里闭眼随机抓(E18b);蓝实线=cVAE 条件生成。"
             "两者共用同一个盒子,差值即模型本身的贡献。",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, .035, 1, .95])
    fig.savefig(outfp, facecolor=SURF); print("→", outfp)


# --------------------------------------------------------------- 图 B
def fig_metrics(G, arm, gendir, outfp, feasible_only=True):
    g = G[arm]; ms = np.array(g["m_grid"])
    z = np.load(os.path.join(gendir, f"e20_{arm}_raw.npz"))
    ng = len(ms)
    nmin = max(5, g["nz"] // 20)          # 中位数至少要这么多可行样本才画
    MI = {k: i for i, k in enumerate(g["met"])}
    fig, axes = plt.subplots(2, 2, figsize=(12.6, 8.0), dpi=200)
    fig.patch.set_facecolor(SURF)
    for ax, (key, ttl, refs, ylim) in zip(axes.ravel(), MET_SPEC):
        ax.set_facecolor(SURF); shade(ax, ms[-1] * 1.2)
        for v, lbl, col in refs:
            ax.axhline(v, color=col, lw=1.3, ls=":", zorder=2)
            ax.text(ms[0] * .9, v, " " + lbl, fontsize=8, color=col, va="bottom")
        for cn in g["conds"]:
            t, sp = split(cn); col, tn = TERR[t]
            ok = z[f"{cn}__ok"][:ng]; mv = z[f"{cn}__met"][:ng, :, MI[key]]
            mask = ok if feasible_only else np.ones_like(ok, bool)
            med = np.array([np.nanmedian(r[k]) if k.sum() >= nmin else np.nan
                            for r, k in zip(mv, mask)])
            ax.plot(ms, med, color=col, lw=2.0, ls=LS[sp], zorder=4)
        ax.set_xscale("log"); ax.set_xlim(ms[0] * .85, ms[-1] * 1.2)
        ax.set_ylim(*ylim)
        ax.set_title(ttl, fontsize=12, color=INK, loc="left", pad=7)
        ax.set_xlabel("机体质量 m / kg", fontsize=9.5, color=INK2)
        ax.grid(alpha=.18, lw=.6, zorder=0)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)
        ax.tick_params(colors=INK2, labelsize=9)
    h = [plt.Line2D([], [], color=TERR[t][0], lw=2.2, label=TERR[t][1]) for t in TERR]
    h += [plt.Line2D([], [], color=MUTED, lw=2.2, ls=LS[s], label=f"v0 = {s} m/s")
          for s in LS]
    fig.legend(handles=h, ncol=5, loc="lower center", frameon=False,
               fontsize=9.5, bbox_to_anchor=(.5, .002))
    fig.suptitle("生成设计的力学成绩单:四项指标随体重（"
                 + ("仅可行设计的中位数" if feasible_only else "全部样本中位数")
                 + f"，可行样本 < {nmin} 处断开）", fontsize=13, color=INK, x=.012,
                 ha="left", y=.985)
    fig.tight_layout(rect=[0, .045, 1, .955])
    fig.savefig(outfp, facecolor=SURF); print("→", outfp)


# --------------------------------------------------------------- 图 C
def fig_anchors(G, arm, gendir, outfp):
    g = G[arm]
    z = np.load(os.path.join(gendir, f"e20_{arm}_raw.npz"))
    ng = len(g["m_grid"]); anc = sorted(float(k) for k in g["anchors"])
    MI = {k: i for i, k in enumerate(g["met"])}
    rows, conds = [], [c for c in g["conds"] if c.endswith("1.2")]
    for cn in conds:
        for i, m in enumerate(anc):
            ok = z[f"{cn}__ok"][ng + i]; mv = z[f"{cn}__met"][ng + i]
            sel = ok if ok.sum() >= 5 else np.ones_like(ok, bool)
            rows.append([g["conds"][cn]["label"].split(" ")[0], f"{m:.0f}",
                         f"{ok.mean()*100:.0f}%",
                         f"{np.nanmedian(mv[sel, MI['peak_g']]):.1f}",
                         f"{np.nanmedian(mv[sel, MI['eta']]):.2f}",
                         f"{np.nanmedian(mv[sel, MI['leg_stroke_mm']]):.1f}",
                         f"{np.nanmedian(mv[sel, MI['leg_mass_g']]):.0f}",
                         f"{np.nanmedian(mv[sel, MI['mass_frac']])*100:.2f}%"])
    hdr = ["地形", "m/kg", "可行率", "峰值/g", "η", "行程/mm", "腿重/g", "占比"]
    fig, ax = plt.subplots(figsize=(11.0, 0.42 * (len(rows) + 3)), dpi=200)
    fig.patch.set_facecolor(SURF); ax.axis("off")
    t = ax.table(cellText=rows, colLabels=hdr, loc="center", cellLoc="center")
    t.auto_set_font_size(False); t.set_fontsize(9.5); t.scale(1, 1.5)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor("#e6e4df")
        if r == 0:
            cell.set_facecolor("#8E2A34"); cell.set_text_props(color="w", weight="bold")
        else:
            cell.set_facecolor("#faf8f6" if r % 2 else "#ffffff")
    ax.set_title("真实机型量级上的成绩单（v0 = 1.2 m/s · 全部在外推区，仅供体量参考）\n"
                 "锚点:30 kg 货运级 · 65 kg FlyCart 30 空机 · 85 kg 半载 · 95 kg 最大起飞重量",
                 fontsize=11.5, color=INK, loc="left", pad=14)
    fig.text(.012, .01, "多旋翼货运机为垂直可控降落,与固定翼着陆不同;"
             "此表只用于给出体量感,不代表可直接用于该机型。", fontsize=8, color=MUTED)
    fig.tight_layout(); fig.savefig(outfp, facecolor=SURF); print("→", outfp)


def summary(G, R, arm):
    g = G[arm]; ms = np.array(g["m_grid"])
    print("\n" + "=" * 86)
    print(f"{arm} 臂 · 可行率(%):模型 vs 随机")
    print(f"{'工况':<20}{'来源':<8}" + "".join(f"{m:>6.1f}" for m in ms))
    for cn, c in g["conds"].items():
        print(f"{c['label']:<20}{'模型':<7}" +
              "".join(f"{v*100:>6.0f}" for v in c["feas"]))
        if cn in R:
            yr = R[cn]["arms"].get(arm, R[cn]["arms"]["bio"])["pooled"]
            print(f"{'':<20}{'随机':<7}" + "".join(f"{v*100:>6.0f}" for v in yr))
    print("\n真实机型锚点可行率(%)")
    for cn, c in g["conds"].items():
        print(f"  {c['label']:<20}" +
              "  ".join(f"{k}kg:{v*100:.0f}" for k, v in c["anchor_feas"].items()))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--gen", default="outputs/v2_e20")
    ap.add_argument("--rand", default="outputs/v2_e18b")
    ap.add_argument("--arm", default="bio")
    a = ap.parse_args()
    G = load_gen(a.gen); R = load_rand(a.rand)
    fig_vs_random(G, R, a.arm, os.path.join(a.gen, "fig_e20_vs_random.png"))
    fig_metrics(G, a.arm, a.gen, os.path.join(a.gen, "fig_e20_metrics.png"))
    fig_anchors(G, a.arm, a.gen, os.path.join(a.gen, "fig_e20_anchors.png"))
    summary(G, R, a.arm)

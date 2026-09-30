# -*- coding: utf-8 -*-
"""E21 出图:真鸟几何 vs 生成几何(同一套刚度),配对斜线图。

配对斜线图是"同一对象两种处理"的标准形式:每条线连接同一物种的两个结果,
线的方向就是结论,不需要读数字。
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager  # noqa: F401
try:
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from cjkfont import setup as _cjk_setup
    _cjk_setup()
except Exception:
    for _c in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        if _c in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [_c] + plt.rcParams["font.sans-serif"]
            break
plt.rcParams["axes.unicode_minus"] = False

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8b8a85"
BIRD = "#eb6834"   # 真鸟几何
GEN = "#2a78d6"    # 生成几何
CN = {"Tachybaptus ruficollis": "小䴙䴘", "Aythya fuligula": "凤头潜鸭",
      "Anas platyrhynchos": "绿头鸭", "Phalacrocorax carbo": "普通鸬鹚",
      "Gavia immer": "普通潜鸟", "Pelecanus onocrotalus": "白鹈鹕",
      "Cygnus olor": "疣鼻天鹅"}
M_TRAIN_LO = 1.0   # cVAE 训练质量下限,低于此为外推


def main(fp, outfp):
    D = json.load(open(fp, encoding="utf-8"))
    conds = sorted({d["cond"] for d in D}, key=lambda c: -[d for d in D if d["cond"] == c][0]["x7"][0])
    conds = ["concrete1.2", "turf1.2", "wetsand1.2"]
    conds = [c for c in conds if any(d["cond"] == c for d in D)]
    sps = sorted({d["species"] for d in D}, key=lambda s: [d for d in D if d["species"] == s][0]["m_kg"])

    fig, axes = plt.subplots(1, len(conds), figsize=(5.0 * len(conds), 6.2),
                             dpi=200, squeeze=False)
    fig.patch.set_facecolor(SURF)
    for k, cn in enumerate(conds):
        ax = axes[0][k]; ax.set_facecolor(SURF)
        lab = [d for d in D if d["cond"] == cn][0]["label"]
        pairs = []
        for sp in sps:
            b = next(d for d in D if d["cond"] == cn and d["species"] == sp and d["who"] == "bird")
            g = next(d for d in D if d["cond"] == cn and d["species"] == sp and d["who"] == "gen")
            pairs.append((sp, b, g))
        good = [(sp, b, g) for sp, b, g in pairs
                if np.isfinite(b["peak_g"]) and np.isfinite(g["peak_g"])]
        if not good:
            ax.axis("off"); continue
        ys = [v for _, b, g in good for v in (b["peak_g"], g["peak_g"])]
        lo, hi = min(ys), max(ys); pad = 0.10 * (hi - lo + 1e-9)
        lo, hi = lo - pad, hi + pad
        # 左侧物种名放在固定行位(按体重排),用引导线连到真实数据点,彻底避免压字
        n = len(good)
        slot = {sp: hi - (i + 0.5) * (hi - lo) / n for i, (sp, _, _) in enumerate(good)}
        for sp, b, g in good:
            yb, yg = b["peak_g"], g["peak_g"]
            win = yg < yb
            ax.plot([-0.30, 0], [slot[sp], yb], color=MUTED, lw=.7, alpha=.55, zorder=1)
            ax.plot([0, 1], [yb, yg], color=(GEN if win else BIRD), lw=2.0,
                    alpha=.9, zorder=3)
            ax.plot([0], [yb], "o", color=BIRD, ms=7.5, zorder=4, mec=SURF, mew=1.3)
            ax.plot([1], [yg], "o", color=GEN, ms=7.5, zorder=4, mec=SURF, mew=1.3)
            mb = "" if b["ok"] else " ×"
            mg = "" if g["ok"] else " ×"
            ax.text(-0.34, slot[sp],
                    f"{CN.get(sp, sp)} {b['m_kg']:.2f}kg  {yb:.2f}{mb}",
                    fontsize=8.4, color=INK2, ha="right", va="center")
            ax.annotate(f"{yg:.2f}{mg}", (1, yg), textcoords="offset points",
                        xytext=(10, 0), fontsize=8.4, va="center",
                        color=(GEN if win else BIRD), fontweight="bold")
        for sp, b, g in pairs:
            if (sp, b, g) not in good:
                ax.text(0.5, lo + 0.04 * (hi - lo), f"{CN.get(sp, sp)}:求解器失败",
                        fontsize=7.5, color=MUTED, ha="center")
        ax.set_ylim(lo, hi)
        ax.set_xlim(-1.62, 1.42); ax.set_xticks([0, 1])
        ax.set_xticklabels(["真鸟几何", "生成几何"], fontsize=10.5)
        ax.tick_params(axis="x", length=0)
        ax.get_xticklabels()[0].set_color(BIRD)
        ax.get_xticklabels()[1].set_color(GEN)
        ax.set_title(lab, fontsize=11, color=INK, loc="left", pad=8)
        ax.grid(axis="y", alpha=.15, lw=.6, zorder=0)
        for sname in ("top", "right", "bottom"):
            ax.spines[sname].set_visible(False)
        ax.tick_params(axis="y", colors=INK2, labelsize=9)
        if k == 0:
            ax.set_ylabel("峰值加速度 / g   (越低越好)", fontsize=10, color=INK2)

    fig.suptitle("水鸟体重区间内:真鸟骨长 vs 生成骨长，只换几何、刚度完全相同",
                 fontsize=13.5, color="#8E2A34", x=.010, ha="left", y=.982)
    fig.text(.010, .938,
             "蓝线 = 生成几何峰值更低;橙线 = 真鸟几何更低。× 表示未通过工程判据(多为行程超限)。",
             fontsize=9.5, color=INK2)
    fig.text(.012, .022,
             "三条限制必须同时读:① 只比几何——真鸟没有关节刚度实测,两侧共用 cVAE 在该体重选的 κ 与 τ;"
             "② r2/r3 统一取雁鸭科中位(其余科无同源骨骼表);"
             "③ 小䴙䴘 0.17 kg 与两种鸭 <1 kg 低于 cVAE 训练下限,属外推。",
             fontsize=8, color=MUTED)
    fig.text(.012, .004, "✕ 判的是本课题的工程判据(g_cap=10g / s_max=24mm / 结构),"
             "不是「这种鸟落不了地」。", fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, .055, 1, .915])
    fig.savefig(outfp, facecolor=SURF); print("→", outfp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/v2_e21/e21_bird_vs_gen.json")
    ap.add_argument("--out", default="outputs/v2_e21/fig_e21_bird_vs_gen.png")
    a = ap.parse_args()
    main(a.json, a.out)

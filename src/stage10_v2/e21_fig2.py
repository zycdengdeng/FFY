# -*- coding: utf-8 -*-
"""E21 出图 v2:拆成两问,每块只回答一个问题。

左:模型要的腿,比真鸟长还是短?(腿长 vs 体重)
右:这个长短差别,换来多少缓冲?(峰值加速度的改善量)

上一版用配对斜线图,读者要自己记住"往下=更好",且引导线互相穿插 —— 弃用。
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager  # noqa: F401
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from cjkfont import setup as _cjk
    _cjk()
except Exception:
    for _c in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        if _c in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [_c] + plt.rcParams["font.sans-serif"]
            break
plt.rcParams["axes.unicode_minus"] = False

SURF = "#fcfcfb"; INK = "#0b0b0b"; INK2 = "#52514e"; MUTED = "#8b8a85"
BIRD = "#eb6834"; GEN = "#2a78d6"
TERR = {"concrete1.2": ("#2a78d6", "硬地"), "turf1.2": ("#eb6834", "草地"),
        "wetsand1.2": ("#1baf7a", "湿沙")}
CN = {"Tachybaptus ruficollis": "小䴙䴘", "Aythya fuligula": "凤头潜鸭",
      "Anas platyrhynchos": "绿头鸭", "Phalacrocorax carbo": "普通鸬鹚",
      "Gavia immer": "普通潜鸟", "Pelecanus onocrotalus": "白鹈鹕",
      "Cygnus olor": "疣鼻天鹅"}
A, B, SIG = 0.4790, 0.39113, 0.078          # 生物异速律


def main(fp, outfp, base="turf1.2"):
    D = json.load(open(fp, encoding="utf-8"))
    sps = sorted({d["species"] for d in D},
                 key=lambda s: [d for d in D if d["species"] == s][0]["m_kg"])
    get = lambda c, s, w: next(d for d in D if d["cond"] == c
                               and d["species"] == s and d["who"] == w)

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14.4, 6.4), dpi=200,
                                   gridspec_kw=dict(width_ratios=[1.0, 1.12]))
    fig.patch.set_facecolor(SURF)

    # ---------- 左:模型要的腿比真鸟长还是短 ----------
    axA.set_facecolor(SURF)
    mm = np.logspace(np.log10(0.13), np.log10(14), 60)
    axA.plot(mm, 10 ** (A + B * np.log10(mm * 1000)), color=MUTED, lw=1.4,
             ls=(0, (5, 3)), zorder=2)
    axA.text(16.5, 25.5, "灰虚线 = 生物异速律中心线\nlog L1 = 0.479 + 0.391 log m",
             fontsize=8.2, color=MUTED, ha="right", va="bottom", linespacing=1.4)
    for si, s in enumerate(sps):
        b, g = get(base, s, "bird"), get(base, s, "gen")
        m = b["m_kg"]; yb, yg = b["L_mm"][0], g["L_mm"][0]
        axA.plot([m, m], [yb, yg], color=(GEN if yg > yb else BIRD),
                 lw=1.6, alpha=.55, zorder=3)
        axA.plot([m], [yb], "o", color=BIRD, ms=8, zorder=4, mec=SURF, mew=1.3)
        axA.plot([m], [yg], "o", color=GEN, ms=8, zorder=4, mec=SURF, mew=1.3)
        pct = (yg - yb) / yb * 100
        # 体重相近的物种在 x 上几乎重合,标签上下交错才不会压在一起
        up = (si % 2 == 0)
        axA.annotate(f"{CN[s]}\n{pct:+.0f}%",
                     (m, max(yb, yg) if up else min(yb, yg)),
                     textcoords="offset points",
                     xytext=((14, 10) if up else (-14, -26)),
                     fontsize=8, color=INK2, linespacing=1.3,
                     ha=("left" if up else "right"),
                     va=("bottom" if up else "top"))
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlim(.115, 18); axA.set_ylim(21, 200)
    axA.set_xlabel("体重 / kg", fontsize=10.5, color=INK2)
    axA.set_ylabel("跗跖长 L1 / mm", fontsize=10.5, color=INK2)
    axA.set_title("① 模型要的腿,比真鸟长还是短?", fontsize=13, color=INK,
                  loc="left", pad=10)
    axA.grid(alpha=.16, lw=.6, which="both", zorder=0)
    for sp_ in ("top", "right"):
        axA.spines[sp_].set_visible(False)
    axA.tick_params(colors=INK2, labelsize=9)
    axA.plot([], [], "o", color=BIRD, ms=8, label="真鸟实测骨长")
    axA.plot([], [], "o", color=GEN, ms=8, label="模型设计骨长")
    axA.legend(fontsize=9.5, frameon=False, loc="upper left")

    # ---------- 右:换来多少缓冲 ----------
    axB.set_facecolor(SURF)
    axB.axvline(0, color="#333333", lw=1.4, zorder=3)
    for i, s in enumerate(sps):
        y0 = len(sps) - 1 - i
        for j, (cn, (col, tn)) in enumerate(TERR.items()):
            b, g = get(cn, s, "bird"), get(cn, s, "gen")
            if not (np.isfinite(b["peak_g"]) and np.isfinite(g["peak_g"])):
                continue
            d = b["peak_g"] - g["peak_g"]        # >0 = 模型几何峰值更低 = 更好
            yy = y0 + (1 - j) * 0.22
            axB.plot([0, d], [yy, yy], color=col, lw=2.2, alpha=.75, zorder=4,
                     solid_capstyle="round")
            axB.plot([d], [yy], "o", color=col, ms=6.5, zorder=5,
                     mec=SURF, mew=1.2)
        axB.text(-3.35, y0, f"{CN[s]}  {get(base, s, 'bird')['m_kg']:.2f} kg",
                 fontsize=9.5, color=INK2, ha="left", va="center")
    axB.set_ylim(-.65, len(sps) - .25)
    axB.set_xlim(-3.4, 3.4)
    axB.set_yticks([])
    axB.set_xlabel("峰值加速度的改善量  Δg = 真鸟 − 模型   (向右 = 模型几何更能缓冲)",
                   fontsize=10, color=INK2)
    axB.set_title("② 这个长短差别,换来多少缓冲?", fontsize=13, color=INK,
                  loc="left", pad=10)
    axB.grid(axis="x", alpha=.16, lw=.6, zorder=0)
    for sp_ in ("top", "right", "left"):
        axB.spines[sp_].set_visible(False)
    axB.tick_params(colors=INK2, labelsize=9)
    axB.text(1.75, -.52, "→ 模型几何更好", fontsize=9, color=GEN, fontweight="bold")
    axB.text(-1.75, -.52, "← 真鸟几何更好", fontsize=9, color=BIRD,
             fontweight="bold", ha="center")
    h = [plt.Line2D([], [], color=c, lw=2.4, label=t) for c, t in TERR.values()]
    axB.legend(handles=h, fontsize=9.5, frameon=False, ncol=3,
               loc="upper right")

    fig.suptitle("真鸟骨长 vs 模型骨长:同一套关节刚度,唯一变量是三段骨头的长度",
                 fontsize=14, color="#8E2A34", x=.010, ha="left", y=.975)
    fig.text(.010, .028,
             "读法:左图看到模型在轻端要比真鸟长得多、在重端反而要短 —— 这就是「更平的标度律」;"
             "右图看到这个选择在轻端换来 1.7–3.0 g 的缓冲改善,在重端趋于持平。",
             fontsize=9, color=INK2)
    fig.text(.010, .008,
             "限制:① 只比几何 —— 真鸟无关节刚度实测,两侧共用模型在该体重选的 κ 与 τ;"
             "右图缺的柱是该工况求解器失败(小䴙䴘硬地、凤头潜鸭草地、绿头鸭湿沙);"
             "② r2/r3 统一取雁鸭科中位;③ 小䴙䴘 0.17 kg 与两种鸭 <1 kg 低于模型训练下限,属外推。",
             fontsize=8, color=MUTED)
    fig.tight_layout(rect=[0, .052, 1, .945])
    fig.savefig(outfp, facecolor=SURF); print("→", outfp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/v2_e21/e21_bird_vs_gen.json")
    ap.add_argument("--out", default="outputs/v2_e21/fig_e21_bird_vs_gen.png")
    a = ap.parse_args()
    main(a.json, a.out)

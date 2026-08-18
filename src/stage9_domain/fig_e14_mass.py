"""E14 论文图:质量维失效的三项证据(负对照图)。

(a) 解码器灵敏度:把每一维工况从 lo 推到 hi,生成设计移动了多少(8 个种子)
(b) 跨质量移植:为 m_i 生成的设计在 m_j 上的峰值——每一行水平即为不变
(c) 不变性验尸:归一化响应沿 m 恒定,而绝对力严格正比 m

用法: python src/stage9_domain/fig_e14_mass.py \
        --json outputs/gen_e14/e14_mass_probe.json --out outputs/gen_e14/fig_e14.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LBL = {"m(kg)": r"$m$", "v0(m/s)": r"$v_0$",
       "gcap(m/s2)": r"$g_{\rm cap}$", "smax(m)": r"$s_{\max}$"}
C_DEAD, C_LIVE, C_GREY = "#C0392B", "#2C5F8A", "#8A8A8A"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/gen_e14/e14_mass_probe.json")
    ap.add_argument("--out", default="outputs/gen_e14/fig_e14.png")
    args = ap.parse_args()
    d = json.load(open(args.json))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": 0.7,
                         "xtick.major.width": 0.7, "ytick.major.width": 0.7,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(10.2, 2.9))

    # ---------------------------------------------------------------- (a)
    A = d["decoder_sensitivity_all"]
    keys = list(next(iter(A.values())).keys())
    vals = np.array([[A[s][k]["relative"] for k in keys] for s in A])   # seeds × dims
    med = np.median(vals, 0)
    xs = np.arange(len(keys))
    cols = [C_DEAD if k.startswith("m(") else C_LIVE for k in keys]
    ax[0].bar(xs, med, 0.62, color=cols, alpha=.85, zorder=2)
    for j in range(len(keys)):                       # 每个种子一个点
        ax[0].scatter(np.full(vals.shape[0], j) + np.linspace(-.16, .16, vals.shape[0]),
                      vals[:, j], s=7, color="k", zorder=3, linewidths=0)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([LBL[k] for k in keys])
    ax[0].set_ylabel("decoder sensitivity\n(rel. to strongest condition)")
    ax[0].set_ylim(0, 1.12)
    ax[0].set_title("(a) which condition moves the design?", loc="left", fontsize=9)
    ax[0].annotate(f"{med[0]:.2f}", (0, med[0]), textcoords="offset points",
                   xytext=(0, 4), ha="center", color=C_DEAD, fontweight="bold")
    ax[0].text(.03, .93, f"dots = {vals.shape[0]} seeds", transform=ax[0].transAxes,
               ha="left", va="top", color=C_GREY, fontsize=6.8)

    # ---------------------------------------------------------------- (b)
    B = d["transplant"]; ms = B["masses"]
    for md in ms:
        y = [B["table"]["%g->%g" % (md, mt)]["peak_g"] for mt in ms]
        ax[1].plot(ms, y, "o-", ms=3.4, lw=1.2,
                   label=f"designed for {md:g} kg")
    ax[1].set_xscale("log"); ax[1].set_xticks(ms)
    ax[1].set_xticklabels([f"{m:g}" for m in ms])
    ax[1].set_xlabel("mass used at test time (kg)")
    ax[1].set_ylabel("peak acceleration (g)")
    ax[1].set_title("(b) cross-mass transplant: every line is flat",
                    loc="left", fontsize=9)
    ax[1].legend(frameon=False, fontsize=6.6, loc="center right", handlelength=1.2)

    # ---------------------------------------------------------------- (c)
    C = d["invariance"]
    mm = np.array(C["masses"])
    pk = np.array(C["example_peak_g"])
    ax[2].plot(mm, pk / pk[0], "o-", color=C_DEAD, ms=4, lw=1.4,
               label=r"normalised response $a_{\rm peak}/a_{\rm peak}(1\,{\rm kg})$")
    ax[2].plot(mm, C["Fpeak_ratio_mean"], "s-", color=C_LIVE, ms=4, lw=1.4,
               label=r"absolute force $F_{\rm peak}/F_{\rm peak}(1\,{\rm kg})$")
    ax[2].plot(mm, mm / mm[0], "--", color="w", lw=1.1, zorder=4, dashes=(2.6, 2.6))
    ax[2].plot([], [], "--", color=C_GREY, lw=1.1, label=r"$\propto m$ (theory, dashed overlay)")
    ax[2].set_xscale("log"); ax[2].set_yscale("log")
    ax[2].set_xticks(mm); ax[2].set_xticklabels([f"{m:g}" for m in mm])
    ax[2].set_xlabel("body mass $m$ (kg)")
    ax[2].set_ylabel("ratio to $m=1$ kg")
    ax[2].set_title("(c) invariance autopsy", loc="left", fontsize=9)
    ax[2].legend(frameon=False, fontsize=6.6, loc="upper left")
    sp = C["rel_spread"]
    ax[2].text(.97, .06,
               f"rel. spread over $m$:\n$a_{{\\rm peak}}$ {sp['peak_a']:.0e}   "
               f"$s$ {sp['stroke']:.0e}   $\\eta$ {sp['eta']:.0e}",
               transform=ax[2].transAxes, ha="right", va="bottom",
               fontsize=6.4, color=C_GREY)

    fig.tight_layout(w_pad=1.6)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[fig] → {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()

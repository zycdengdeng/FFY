"""v2 主图:地面刚度 × 体重 的能力图。

(a) 峰值加速度:每条线一个地面刚度。绝对地面对更重的机体**相对更软**,
    所以峰值随 m 单调下降——这正是 v1 被人为消掉的那条依赖。
(b) 行程的两个来源:起落架自身行程 vs 地面下陷。硬地上腿行程几乎与 m 无关
    (6% 以内),质量效应全在下陷里;软地上两者都爆。**若不把它们分开,
    就会把"陷进去"误判成"腿塌了"**(v1 判据的 bug)。
(c) 可行率:两个方向相反的质量通道打架,可行域在质量方向出现内部最优。

用法: python src/stage10_v2/fig_v2_map.py --json outputs/gen_v2_g1/e15_map.json \
        --out outputs/gen_v2_g1/fig_v2_map.png
"""
from __future__ import annotations

import argparse
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", default="outputs/gen_v2_g1/e15_map.json")
    ap.add_argument("--out", default="outputs/gen_v2_g1/fig_v2_map.png")
    args = ap.parse_args()
    d = json.load(open(args.json))
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    ms = np.array(d["masses"], float)
    kcs = np.array([float(k) for k in d["grid"]["peak_a"]])
    order = np.argsort(-kcs)
    kcs = kcs[order]
    def mat(name):
        A = np.array([d["grid"][name][f"{k:g}"] for k in kcs], float)
        return A
    PK, SK, LS = mat("peak_a"), mat("sink"), mat("leg_stroke")
    FE = np.array([d["feas"][f"{k:g}"] for k in kcs], float)

    plt.rcParams.update({"font.size": 8.5, "axes.linewidth": .7,
                         "axes.spines.top": False, "axes.spines.right": False})
    fig, ax = plt.subplots(1, 3, figsize=(10.6, 3.0))
    cmap = plt.cm.viridis(np.linspace(.05, .92, len(kcs)))

    # ---------------------------------------------------------------- (a)
    for i, k in enumerate(kcs):
        ax[0].plot(ms, PK[i], "o-", ms=3.2, lw=1.3, color=cmap[i],
                   label=f"{k:.0e}".replace("e+0", "e"))
    ax[0].axhline(10, ls="--", lw=.9, color="#C0392B")
    ax[0].text(1.05, 10.6, r"$g_{\rm cap}=10\,$g", color="#C0392B", fontsize=6.8)
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xticks(ms); ax[0].set_xticklabels([f"{m:g}" for m in ms])
    ax[0].set_yticks([3, 5, 10, 20, 35]); ax[0].set_yticklabels(["3", "5", "10", "20", "35"])
    ax[0].set_xlabel("body mass $m$ (kg)"); ax[0].set_ylabel("peak acceleration (g)")
    ax[0].set_title("(a) absolute ground is relatively\n     softer for heavier craft",
                    loc="left", fontsize=8.6)
    ax[0].legend(title=r"$k_c$ (N/m)", frameon=False, fontsize=6.0,
                 title_fontsize=6.2, loc="lower left", handlelength=1.1, ncol=2)

    # ---------------------------------------------------------------- (b)
    hard, soft = 0, len(kcs) - 1
    ax[1].plot(ms, LS[hard], "o-", color="#2C5F8A", ms=3.4, lw=1.4,
               label=f"leg stroke, $k_c$={kcs[hard]:.0e}".replace("e+0", "e"))
    ax[1].plot(ms, SK[hard], "o--", color="#2C5F8A", ms=3.4, lw=1.2, alpha=.55,
               label="ground sink, same")
    ax[1].plot(ms, LS[soft], "s-", color="#C0392B", ms=3.4, lw=1.4,
               label=f"leg stroke, $k_c$={kcs[soft]:.0e}".replace("e+0", "e"))
    ax[1].plot(ms, SK[soft], "s--", color="#C0392B", ms=3.4, lw=1.2, alpha=.55,
               label="ground sink, same")
    ax[1].set_xscale("log"); ax[1].set_xticks(ms)
    ax[1].set_xticklabels([f"{m:g}" for m in ms])
    ax[1].set_xlabel("body mass $m$ (kg)"); ax[1].set_ylabel("displacement (mm)")
    ax[1].set_title("(b) stroke vs sink must be separated", loc="left", fontsize=8.6)
    ax[1].legend(frameon=False, fontsize=6.0, loc="upper left", handlelength=1.6)

    # ---------------------------------------------------------------- (c)
    im = ax[2].imshow(np.maximum(FE, .1), aspect="auto", origin="upper",
                      cmap="magma", norm=LogNorm(vmin=1, vmax=max(FE.max(), 2)))
    ax[2].set_xticks(range(len(ms))); ax[2].set_xticklabels([f"{m:g}" for m in ms])
    ax[2].set_yticks(range(len(kcs)))
    ax[2].set_yticklabels([f"{k:.0e}".replace("e+0", "e") for k in kcs], fontsize=7)
    for i in range(len(kcs)):
        j = int(np.argmax(FE[i]))
        if 0 < j < len(ms) - 1:          # 只标真正落在内部的最优,边界上的不算
            ax[2].plot(j, i, "*", color="w", ms=8, mec="k", mew=.5)
        for jj in range(len(ms)):
            ax[2].text(jj, i, f"{FE[i, jj]:.0f}", ha="center", va="center",
                       fontsize=6.0, color="w" if FE[i, jj] < 30 else "k")
    ax[2].set_xlabel("body mass $m$ (kg)"); ax[2].set_ylabel("$k_c$ (N/m)")
    ax[2].set_title("(c) feasible fraction (%) — ★ interior optimum",
                    loc="left", fontsize=8.6)
    fig.colorbar(im, ax=ax[2], fraction=.046, pad=.02).set_label("%", fontsize=7)

    fig.tight_layout(w_pad=1.7)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    fig.savefig(args.out.replace(".png", ".pdf"), bbox_inches="tight")
    print(f"[fig] → {args.out} (+ .pdf)")


if __name__ == "__main__":
    main()

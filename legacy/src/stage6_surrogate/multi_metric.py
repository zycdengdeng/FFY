"""多指标落震评估:回应"只看峰值加速度太单一"。

在 m=10kg(天鹅级)条件下,LHS 采样设计 + 真天鹅几何,用 Exudyn 输出
全指标集(peak_a/stroke/η/CFE/jerk/回弹/弹跳/稳定时间),回答:
1) 换指标后,真天鹅还在 Pareto 前沿上吗?(多组两两目标核查)
2) 真天鹅在每个指标上处于设计群体的第几百分位?
3) 天鹅 η 是否落入油气式支柱典型区 0.8-0.9?

用法: python src/stage6_surrogate/multi_metric.py --out outputs/multi_metric [--m 10 --nd 120]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import models as M
from hf_exudyn import exu_eval, SCEN_BIRD_X

SWAN_X = (111.2, 1.764, 0.951)       # 天鹅骨架实测 (Watanabe 2017: TMT/TIB/FEM)——主用
SWAN_BP = (123.0, 1.72, 0.89)        # 蓝本 Table 4 口径——对照
KEYS = ["peak_a", "stroke", "eta", "cfe", "peak_jerk", "rebound", "n_bounce", "t_settle"]
# 每个指标的"好方向":-1=越小越好, +1=越大越好
GOOD = dict(peak_a=-1, stroke=-1, eta=+1, cfe=+1, peak_jerk=-1,
            rebound=-1, n_bounce=-1, t_settle=-1)
LABEL = dict(peak_a="峰值过载", stroke="缓冲行程", eta="缓冲效率η", cfe="CFE",
             peak_jerk="峰值jerk", rebound="回弹高度", n_bounce="弹跳次数",
             t_settle="稳定时间")


def _eval_one(args):
    x, m, v0, kappa = args
    sc = M.bird_size({**SCEN_BIRD_X, "m": m, "v0": v0, "kappa": kappa}, x)
    return exu_eval(tuple(x), sc)


def pareto_mask(P):
    """P:(N,k) 全部按"越小越好"排列;返回非支配掩码。"""
    n = len(P); dom = np.zeros(n, bool)
    for i in range(n):
        for k in range(n):
            if k != i and np.all(P[k] <= P[i]) and np.any(P[k] < P[i]):
                dom[i] = True; break
    return ~dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/multi_metric")
    ap.add_argument("--m", type=float, default=10.0)
    ap.add_argument("--v0", type=float, default=1.2)
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--nd", type=int, default=120)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    lo = np.array(M.LO_BIRD); hi = np.array(M.HI_BIRD)   # 实测边界 v2(见 models.py 注释)
    rng = np.random.default_rng(42)
    X = np.empty((args.nd, 3))
    for j in range(3):
        e = lo[j] + (hi[j] - lo[j]) * (np.arange(args.nd) + rng.random(args.nd)) / args.nd
        X[:, j] = rng.permutation(e)

    with ProcessPoolExecutor(max_workers=8) as ex:
        R = list(ex.map(_eval_one, [(x, args.m, args.v0, args.kappa) for x in X], chunksize=2))
    swan = _eval_one((np.array(SWAN_X), args.m, args.v0, args.kappa))
    swan_bp = _eval_one((np.array(SWAN_BP), args.m, args.v0, args.kappa))
    print("蓝本口径天鹅(对照): " + " ".join(f"{k}={swan_bp[k]:.3g}" for k in KEYS))

    Y = np.array([[r[k] for k in KEYS] for r in R])
    ok = np.isfinite(Y[:, 0])
    Xo, Yo = X[ok], Y[ok]
    sw = np.array([swan[k] for k in KEYS])
    print(f"feasible {ok.sum()}/{args.nd}  |  真天鹅: " +
          " ".join(f"{k}={swan[k]:.3g}" for k in KEYS))

    # --- 1) 百分位:好方向上击败了百分之多少的设计 ---
    pct = {}
    for i, k in enumerate(KEYS):
        col = Yo[:, i]
        beat = (sw[i] <= col) if GOOD[k] < 0 else (sw[i] >= col)
        pct[k] = float(100.0 * np.mean(beat))

    # --- 2) 各目标组合下,真天鹅是否非支配(把天鹅并入群体再判) ---
    pairs = [("peak_a", "stroke"), ("peak_a", "eta"), ("peak_jerk", "stroke"),
             ("peak_a", "t_settle"), ("peak_a", "stroke", "eta")]
    membership = {}
    for combo in pairs:
        cols = [KEYS.index(k) for k in combo]
        P = np.vstack([Yo[:, cols], sw[cols]])
        for j, k in enumerate(combo):               # 统一为最小化
            if GOOD[k] > 0: P[:, j] = -P[:, j]
        pm = pareto_mask(P)
        membership["+".join(combo)] = bool(pm[-1])

    print("百分位:", {k: f"{v:.0f}%" for k, v in pct.items()})
    print("前沿成员资格:", membership)

    # 存档
    import csv
    with open(os.path.join(args.out, "multi_metric.csv"), "w", newline="") as f:
        w = csv.writer(f); w.writerow(["L1", "r2", "r3"] + KEYS + ["is_swan"])
        for x, y in zip(Xo, Yo): w.writerow(list(x) + list(y) + [0])
        w.writerow(list(SWAN_X) + list(sw) + [1])
    json.dump({"m": args.m, "v0": args.v0, "kappa": args.kappa, "keys": KEYS,
               "swan": {k: float(swan[k]) for k in KEYS},
               "swan_percentile_beat": pct, "swan_on_front": membership,
               "n_feasible": int(ok.sum()), "nd": args.nd},
              open(os.path.join(args.out, "multi_metric.json"), "w"),
              indent=2, ensure_ascii=False)

    # ---- 图:2x2 ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p): fm.fontManager.addfont(p)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    MA = "#7F2D32"
    g = 9.81
    fig, ax = plt.subplots(2, 2, figsize=(12.8, 9.6), dpi=115)

    sc0 = ax[0, 0].scatter(Yo[:, 1] * 1e3, Yo[:, 0] / g, c=Yo[:, 2], cmap="viridis",
                           s=26, alpha=.85, vmin=np.nanmin(Yo[:, 2]), vmax=np.nanmax(Yo[:, 2]))
    ax[0, 0].scatter(sw[1] * 1e3, sw[0] / g, marker="D", color="crimson", s=130,
                     edgecolor="k", zorder=5, label="真天鹅几何")
    plt.colorbar(sc0, ax=ax[0, 0], label="缓冲效率 η")
    ax[0, 0].set_xlabel("缓冲行程 (mm) ↓"); ax[0, 0].set_ylabel("峰值过载 (g) ↓")
    ax[0, 0].set_title("老双目标空间,颜色=新指标 η", color=MA, fontweight="bold")
    ax[0, 0].legend(fontsize=9); ax[0, 0].grid(alpha=.3)

    ax[0, 1].scatter(Yo[:, 2], Yo[:, 0] / g, s=26, alpha=.6, color="#3A6EA5")
    ax[0, 1].scatter(sw[2], sw[0] / g, marker="D", color="crimson", s=130,
                     edgecolor="k", zorder=5)
    ax[0, 1].axvspan(0.8, 0.9, color="orange", alpha=.15, label="油气支柱典型区 0.8–0.9")
    ax[0, 1].set_xlabel("缓冲效率 η ↑"); ax[0, 1].set_ylabel("峰值过载 (g) ↓")
    ax[0, 1].set_title("效率-过载空间:天鹅在哪?", color=MA, fontweight="bold")
    ax[0, 1].legend(fontsize=9); ax[0, 1].grid(alpha=.3)

    ax[1, 0].scatter(Yo[:, 4] / g, Yo[:, 0] / g, s=26, alpha=.6, color="#3A6EA5")
    ax[1, 0].scatter(sw[4] / g, sw[0] / g, marker="D", color="crimson", s=130,
                     edgecolor="k", zorder=5)
    ax[1, 0].set_xlabel("峰值 jerk (g/s) ↓"); ax[1, 0].set_ylabel("峰值过载 (g) ↓")
    ax[1, 0].set_title("jerk(冲击突兀感)vs 过载", color=MA, fontweight="bold")
    ax[1, 0].grid(alpha=.3)

    show = [k for k in KEYS if k not in ("n_bounce", "rebound")]
    names = [LABEL[k] for k in show]
    vals = [pct[k] for k in show]
    ax[1, 1].barh(names[::-1], vals[::-1], color=["#3A6EA5" if v >= 50 else "#999" for v in vals[::-1]])
    ax[1, 1].axvline(50, color="k", ls=":", lw=1)
    ax[1, 1].set_xlabel("真天鹅击败的设计比例 (%)")
    ax[1, 1].set_title("天鹅在各指标上的百分位", color=MA, fontweight="bold")
    ax[1, 1].set_xlim(0, 100); ax[1, 1].grid(alpha=.3, axis="x")
    ax[1, 1].text(2, -0.6, "注:回弹/弹跳全体=0(阻尼充分),不区分设计,未列入",
                  fontsize=8, color="#666")

    fig.suptitle(f"多指标落震评估 m={args.m}kg v0={args.v0}m/s (n={ok.sum()} 可行设计)",
                 fontsize=13, fontweight="bold")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(os.path.join(args.out, "multi_metric.png"), bbox_inches="tight")
    print(f"[mm] wrote multi_metric.csv/json/png → {args.out}")


if __name__ == "__main__":
    main()

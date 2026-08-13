"""触地姿态敏感性检验:膝/踝角 2×2(现值 vs Duong 触水帧实测中位)。

背景:固定姿态 (跖仰角50°/踝120°/膝90°) 沿用课题组前期约定;与组内 Duong 12 段
视频 water_contact 帧实测核对后发现:踝 120° 在实测范围 113–160° 内,但膝 90°
低于实测范围 118–158°(中位:踝 144°、膝 133°)。本脚本回答两个问题:
  1) 膝角取错了会不会改变结论?(天鹅画像百分位、设计排名稳定性)
  2) 直接改用实测中位姿态,结论变不变?

方法:与 multi_metric.py 同源——同一 120 设计 LHS(种子 42)+ 真天鹅骨架几何,
m=10kg v0=1.2m/s κ=4;仅改触地姿态角,4 个变体 × 121 次仿真 ≈ 500 次。

用法: python src/stage6_surrogate/theta_sens.py --out outputs/theta_sens --workers 32
产出: theta_sens.json + theta_sens.png + 控制台结论表
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import models as M
from hf_exudyn import exu_eval, SCEN_BIRD_X

SWAN_X = (111.2, 1.764, 0.951)          # Watanabe 2017 骨架实测
KEYS = ["peak_a", "stroke", "eta", "cfe", "peak_jerk"]
GOOD = dict(peak_a=-1, stroke=-1, eta=+1, cfe=+1, peak_jerk=-1)
LABEL = dict(peak_a="峰值过载", stroke="缓冲行程", eta="缓冲效率η",
             cfe="CFE", peak_jerk="峰值jerk")

# 变体:(踝 thetaA, 膝 thetaK)/度。现值 (120,90);Duong 触水帧中位 (144,133)。
VARIANTS = [("现值 120/90", 120.0, 90.0),
            ("仅改膝 120/133", 120.0, 133.0),
            ("仅改踝 144/90", 144.0, 90.0),
            ("Duong中位 144/133", 144.0, 133.0)]


def _eval_one(args):
    x, m, v0, kappa, ta, tk = args
    sc = M.bird_size({**SCEN_BIRD_X, "m": m, "v0": v0, "kappa": kappa,
                      "thetaA": np.radians(ta), "thetaK": np.radians(tk)}, x)
    return exu_eval(tuple(x), sc)


def spearman(a, b):
    """两组数的秩相关(排名一致性,1=完全一致)。"""
    ra = np.argsort(np.argsort(a)).astype(float)
    rb = np.argsort(np.argsort(b)).astype(float)
    return float(np.corrcoef(ra, rb)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/theta_sens")
    ap.add_argument("--m", type=float, default=10.0)
    ap.add_argument("--v0", type=float, default=1.2)
    ap.add_argument("--kappa", type=float, default=4.0)
    ap.add_argument("--nd", type=int, default=120)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    lo = np.array(M.LO_BIRD); hi = np.array(M.HI_BIRD)
    rng = np.random.default_rng(42)                      # 与 multi_metric 同种子
    X = np.empty((args.nd, 3))
    for j in range(3):
        e = lo[j] + (hi[j] - lo[j]) * (np.arange(args.nd) + rng.random(args.nd)) / args.nd
        X[:, j] = rng.permutation(e)

    res = {}
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for name, ta, tk in VARIANTS:
            jobs = [(x, args.m, args.v0, args.kappa, ta, tk) for x in X]
            R = list(ex.map(_eval_one, jobs, chunksize=4))
            swan = _eval_one((np.array(SWAN_X), args.m, args.v0, args.kappa, ta, tk))
            Y = np.array([[r[k] for k in KEYS] for r in R])
            ok = np.isfinite(Y[:, 0])
            Yo = Y[ok]
            sw = np.array([swan[k] for k in KEYS])
            pct = {}
            for i, k in enumerate(KEYS):
                beat = (sw[i] <= Yo[:, i]) if GOOD[k] < 0 else (sw[i] >= Yo[:, i])
                pct[k] = float(100.0 * np.mean(beat))
            res[name] = dict(thetaA=ta, thetaK=tk, n_feas=int(ok.sum()),
                             swan={k: float(swan[k]) for k in KEYS},
                             pct=pct, ok=ok.tolist(),
                             peak_all=Y[:, 0].tolist())
            print(f"[θ] {name}: 可行 {ok.sum()}/{args.nd}  天鹅峰值 {swan['peak_a']/9.81:.2f}g  "
                  f"η {swan['eta']:.3f}  百分位 " +
                  " ".join(f"{LABEL[k]}{v:.0f}%" for k, v in pct.items()))

    # 排名稳定性:各变体 vs 现值,对共同可行设计比峰值排名
    base = res[VARIANTS[0][0]]
    okb = np.array(base["ok"]); pb = np.array(base["peak_all"])
    stab = {}
    for name, _, _ in VARIANTS[1:]:
        okv = np.array(res[name]["ok"]); pv = np.array(res[name]["peak_all"])
        both = okb & okv
        stab[name] = spearman(pb[both], pv[both])
    print("排名稳定性(Spearman vs 现值):",
          {k: f"{v:.3f}" for k, v in stab.items()})

    json.dump(dict(variants={n: {kk: vv for kk, vv in d.items() if kk != "peak_all" and kk != "ok"}
                             for n, d in res.items()},
                   rank_stability=stab,
                   duong_source="组内 12 段视频 water_contact 帧:膝 118–157°(中位 133),踝 113–160°(中位 144)"),
              open(os.path.join(args.out, "theta_sens.json"), "w"),
              indent=2, ensure_ascii=False)

    # 图:天鹅百分位 × 4 变体分组柱
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p): fm.fontManager.addfont(p)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    cols = ["#8E2A34", "#C46A6A", "#6A8EC4", "#3A6EA5"]
    fig, ax = plt.subplots(figsize=(10.5, 5.4), dpi=150)
    w = 0.2
    xs = np.arange(len(KEYS))
    for vi, (name, _, _) in enumerate(VARIANTS):
        vals = [res[name]["pct"][k] for k in KEYS]
        ax.bar(xs + (vi - 1.5) * w, vals, w, label=name, color=cols[vi], alpha=.9)
    ax.set_xticks(xs); ax.set_xticklabels([LABEL[k] for k in KEYS])
    ax.axhline(50, color="k", ls=":", lw=1)
    ax.set_ylabel("天鹅击败的设计比例 (%)"); ax.set_ylim(0, 100)
    ax.set_title("触地姿态 2×2 敏感性:天鹅画像百分位随膝/踝角的变化",
                 fontweight="bold", color="#8E2A34")
    ax.legend(fontsize=9); ax.grid(alpha=.3, axis="y")
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "theta_sens.png"), bbox_inches="tight")
    print(f"[θ] wrote theta_sens.json/png → {args.out}")


if __name__ == "__main__":
    main()

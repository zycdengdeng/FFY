"""水鸟尺度 Pareto 选优 + 真鸟构型对比。

问题:真鸟的腿,在它自己的载荷下,是否已接近工程 Pareto 最优?
做法:对一系列体重 m(鸭→天鹅),各自 LHS 采样几何 (L1,r2,r3) → Exudyn 落震
     → (峰值加速度, 缓冲行程) 双目标 Pareto → 折中最优;
     把最优 L1 随体重的曲线与真实鸟类解剖数据叠加;真天鹅构型放进 m=10kg 的目标空间看位置。
条件固定:v0=1.2 m/s(均值),姿态 50°/120°/90°,关节/接触按载荷配簧(bird_size)。

用法:python src/stage6_surrogate/bird_pareto.py --out outputs/bird_pareto [--nd 80]
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import models as M
from hf_exudyn import exu_eval, SCEN_BIRD_X

MASSES = [1.2, 2.0, 4.0, 7.0, 10.0, 12.0]          # kg, 鸭→天鹅/鹈鹕
V0REF, MREF, V0EXP = 1.2, 5.0, 0.0                  # v0(m)=V0REF*(m/MREF)**V0EXP;EXP=0 即固定

def v0_of(m):
    return V0REF * (m / MREF) ** V0EXP
# 真实鸟类解剖参考(跖骨长,近似文献值)
REAL_BIRDS = {"Duck 鸭": (1.2, 45.0), "Goose 鹅": (4.0, 85.0),
              "Swan 天鹅": (10.0, 123.0), "Pelican 鹈鹕": (10.5, 115.0)}
SWAN_X = (123.0, 1.72, 0.89)                        # 真天鹅几何(蓝本 Table 4)


def _eval_one(args):
    x, m = args
    sc = M.bird_size({**SCEN_BIRD_X, "m": m, "v0": v0_of(m)}, x)
    r = exu_eval(tuple(x), sc)
    return r["peak_a"], r["stroke"]


def pareto2(P):
    """P:(N,2) 双目标最小化;返回非支配掩码。"""
    n = len(P); dom = np.zeros(n, bool)
    for i in range(n):
        for k in range(n):
            if k != i and np.all(P[k] <= P[i]) and np.any(P[k] < P[i]):
                dom[i] = True; break
    return ~dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="outputs/bird_pareto"); ap.add_argument("--nd", type=int, default=80)
    ap.add_argument("--v0exp", type=float, default=0.0)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    global V0EXP; V0EXP = args.v0exp
    from concurrent.futures import ProcessPoolExecutor

    lo = np.array([40.0, 1.3, 0.8]); hi = np.array([130.0, 2.5, 1.9])
    rng = np.random.default_rng(42)
    results = {}
    for m in MASSES:
        X = np.empty((args.nd, 3))
        for j in range(3):
            e = lo[j] + (hi[j] - lo[j]) * (np.arange(args.nd) + rng.random(args.nd)) / args.nd
            X[:, j] = rng.permutation(e)
        with ProcessPoolExecutor(max_workers=8) as ex:
            Y = np.array(list(ex.map(_eval_one, [(x, m) for x in X], chunksize=2)))
        ok = np.isfinite(Y[:, 0])
        Xo, Yo = X[ok], Y[ok]
        pm = pareto2(Yo)
        norm = (Yo - Yo.min(0)) / (np.ptp(Yo, axis=0) + 1e-9)
        sel = int(np.where(pm)[0][np.argmin(np.linalg.norm(norm[pm], axis=1))])
        results[m] = dict(X=Xo.tolist(), Y=Yo.tolist(), pareto=pm.tolist(), sel=sel,
                          n_fail=int((~ok).sum()))
        b = Xo[sel]
        print(f"m={m:5.1f}kg  最优折中: L1={b[0]:.0f}mm r2={b[1]:.2f} r3={b[2]:.2f} "
              f"| peak={Yo[sel,0]/9.81:.1f}g stroke={Yo[sel,1]*1000:.0f}mm | fail {(~ok).sum()}/{args.nd}")

    # 真天鹅构型在 m=10 下的表现
    swan_y = _eval_one((np.array(SWAN_X), 10.0))
    print(f"真天鹅几何 @10kg: peak={swan_y[0]/9.81:.1f}g stroke={swan_y[1]*1000:.0f}mm")

    json.dump({"masses": MASSES, "v0ref": V0REF, "v0exp": V0EXP, "results": {str(k): v for k, v in results.items()},
               "swan_real": {"x": SWAN_X, "peak_a": swan_y[0], "stroke": swan_y[1]},
               "real_birds_L1": REAL_BIRDS},
              open(os.path.join(args.out, "pareto_results.json"), "w"))

    # ---- 图 ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p): fm.fontManager.addfont(p)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    MA = "#7F2D32"

    fig, ax = plt.subplots(1, 2, figsize=(13.2, 5.2), dpi=115)
    # 左:m=10 的目标空间 + 真天鹅
    r10 = results[10.0]; Y = np.array(r10["Y"]); pm = np.array(r10["pareto"]); sel = r10["sel"]
    ax[0].scatter(Y[:, 1] * 1000, Y[:, 0] / 9.81, s=18, alpha=.5, color="#3A6EA5", label="LHS 设计")
    ax[0].scatter(Y[pm, 1] * 1000, Y[pm, 0] / 9.81, edgecolor="crimson", facecolor="none",
                  s=70, lw=1.4, label="Pareto 前沿")
    ax[0].scatter(Y[sel, 1] * 1000, Y[sel, 0] / 9.81, marker="*", color="red", s=300,
                  edgecolor="k", zorder=5, label="折中最优")
    ax[0].scatter(swan_y[1] * 1000, swan_y[0] / 9.81, marker="D", color="darkgreen", s=110,
                  zorder=6, label="真天鹅几何 (123mm,1.72,0.89)")
    ax[0].set_xlabel("缓冲行程 (mm) ↓"); ax[0].set_ylabel("峰值加速度 (g) ↓")
    ax[0].set_title("m=10kg(天鹅级)目标空间:真鸟落在哪?", color=MA, fontweight="bold")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3)
    # 右:最优 L1 vs 体重 + 真实鸟类
    ms = MASSES
    L1opt = [np.array(results[m]["X"])[results[m]["sel"], 0] for m in ms]
    ax[1].plot(ms, L1opt, "-o", color=MA, lw=2.2, ms=6, label="Pareto 折中最优 L1(仿真寻优)")
    for name, (bm, bl) in REAL_BIRDS.items():
        ax[1].scatter(bm, bl, marker="D", s=90, zorder=5, label=f"{name} 实测 {bl:.0f}mm")
    ax[1].set_xlabel("体重 m (kg)"); ax[1].set_ylabel("跖骨长 L1 (mm)")
    ax[1].set_title("最优腿长随体重的规律 vs 真实鸟类", color=MA, fontweight="bold")
    ax[1].legend(fontsize=8.5); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "bird_pareto.png"), bbox_inches="tight")
    print(f"[pareto] wrote pareto_results.json + bird_pareto.png → {args.out}")


if __name__ == "__main__":
    main()

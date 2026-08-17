"""论文主图:水鸟腿长标度律(b=0.39)+ 骨段比例保守性。

面板 (a) AVONET×EltonTraits 214 种水鸟,log-log 跗跖长 vs 体重:
  按科着色散点 + OLS 拟合线与 95% 置信带 + 几何相似参考斜率 1/3
  + 全鸟类参考 0.31 + 冲击优化模型预测扇区 b∈[0.37,0.49] + 疣鼻天鹅标星。
面板 (b) Watanabe 2017 会飞雁鸭科 91 种,骨段比例 r2/r3 vs 跗跖长(尺寸代理):
  斜率≈0 → "长度是主旋钮,比例保守"。

用法: python src/stage6_surrogate/fig_allometry_paper.py \
        --csv outputs/bird_pareto/avonet_waterbirds.csv \
        --wcsv data/skeletal/watanabe2017_anatidae.csv --out outputs/bird_pareto
产出: fig_allometry_paper.png(300dpi)+ .pdf(矢量,论文用)
"""
from __future__ import annotations
import argparse, csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

FAM_STYLE = {  # 色盲友好 Okabe-Ito
    "Anatidae":          ("#0072B2", "o", "Anatidae (ducks, geese, swans)"),
    "Phalacrocoracidae": ("#009E73", "s", "Phalacrocoracidae (cormorants)"),
    "Podicipedidae":     ("#E69F00", "^", "Podicipedidae (grebes)"),
    "Pelecanidae":       ("#CC79A7", "D", "Pelecanidae (pelicans)"),
    "Gaviidae":          ("#D55E00", "v", "Gaviidae (loons)"),
}
SWAN = "Cygnus olor"


def ols_band(x, y, xg):
    """log 空间 OLS + 95% 置信带。"""
    n = len(x)
    b, a, r, _, se = stats.linregress(x, y)
    yg = a + b * xg
    s = np.sqrt(np.sum((y - (a + b * x)) ** 2) / (n - 2))
    t = stats.t.ppf(0.975, n - 2)
    half = t * s * np.sqrt(1 / n + (xg - x.mean()) ** 2 / np.sum((x - x.mean()) ** 2))
    return a, b, t * se, r ** 2, yg, half


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="outputs/bird_pareto/avonet_waterbirds.csv")
    ap.add_argument("--wcsv", default="data/skeletal/watanabe2017_anatidae.csv")
    ap.add_argument("--out", default="outputs/bird_pareto")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    rows = list(csv.DictReader(open(args.csv)))
    m = np.array([float(r["BodyMass.Value"]) for r in rows]) / 1000.0   # kg
    L = np.array([float(r["tarsus_mm"]) for r in rows])
    fam = [r["Family"] for r in rows]
    names = [r["scientificNameStd"] for r in rows]

    lx, ly = np.log10(m), np.log10(L)
    xg = np.linspace(lx.min() - 0.05, lx.max() + 0.05, 200)
    a, b, bci, r2, yg, half = ols_band(lx, ly, xg)

    wrows = list(csv.DictReader(open(args.wcsv)))
    wv = [r for r in wrows if r["group"] == "Volant"]
    tmt = np.array([float(r["tmt_mm"]) for r in wv])
    R2 = np.array([float(r["r2"]) for r in wv])
    R3 = np.array([float(r["r3"]) for r in wv])
    s2 = stats.linregress(np.log10(tmt), np.log10(R2))
    s3 = stats.linregress(np.log10(tmt), np.log10(R3))

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.2, 4.6), dpi=300,
                                   gridspec_kw=dict(width_ratios=[1.25, 1]))

    # ---------------- (a) allometry ----------------
    for f, (c, mk, lab) in FAM_STYLE.items():
        sel = [i for i, ff in enumerate(fam) if ff == f]
        axA.scatter(m[sel], L[sel], s=26, c=c, marker=mk, alpha=.75,
                    edgecolor="white", linewidth=.3,
                    label=f"{lab}, n={len(sel)}")
    # 模型预测扇区 b∈[0.37,0.49](过数据形心)
    cx, cy = lx.mean(), ly.mean()
    for lo_b, hi_b in [(0.37, 0.49)]:
        y_lo = cy + lo_b * (xg - cx); y_hi = cy + hi_b * (xg - cx)
        axA.fill_between(10 ** xg, 10 ** y_lo, 10 ** y_hi, color="#B22222",
                         alpha=.08, zorder=0)
        axA.plot(10 ** xg, 10 ** y_lo, color="#B22222", lw=.8, ls=":", alpha=.6)
        axA.plot(10 ** xg, 10 ** y_hi, color="#B22222", lw=.8, ls=":", alpha=.6)
    # OLS + 95% CI
    axA.plot(10 ** xg, 10 ** yg, color="k", lw=2.0,
             label=f"OLS fit: $b={b:.2f}\\pm{bci:.2f}$, $R^2={r2:.2f}$")
    axA.fill_between(10 ** xg, 10 ** (yg - half), 10 ** (yg + half),
                     color="k", alpha=.15)
    # 参考斜率:几何相似 1/3、全鸟类 0.31
    axA.plot(10 ** xg, 10 ** (cy + (xg - cx) / 3), color="#666", lw=1.2, ls="--",
             label="geometric similarity, $b=1/3$")
    # swan
    si = names.index(SWAN)
    axA.scatter(m[si], L[si], marker="*", s=380, color="#B22222",
                edgecolor="k", zorder=6, label=f"mute swan ($Cygnus\\ olor$)")
    axA.set_xscale("log"); axA.set_yscale("log")
    axA.set_xlabel("Body mass $m$ (kg)")
    axA.set_ylabel("Tarsometatarsus length $L$ (mm)")
    axA.text(0.03, 0.97, "(a)", transform=axA.transAxes, fontsize=13,
             fontweight="bold", va="top")
    axA.text(0.97, 0.05,
             "impact-optimization model\npredicts $b\\in[0.37,0.49]$ (shaded)",
             transform=axA.transAxes, fontsize=8.5, color="#B22222",
             ha="right", va="bottom")
    axA.legend(fontsize=7.6, loc="upper left", bbox_to_anchor=(0.02, 0.94),
               framealpha=.9)
    axA.grid(alpha=.25, which="both")

    # ---------------- (b) proportion conservation ----------------
    axB.scatter(tmt, R2, s=26, color="#0072B2", marker="o", alpha=.75,
                edgecolor="white", linewidth=.3,
                label=f"$r_2$=TIB/TMT (slope {s2.slope:+.2f})")
    axB.scatter(tmt, R3, s=26, color="#009E73", marker="s", alpha=.75,
                edgecolor="white", linewidth=.3,
                label=f"$r_3$=FEM/TMT (slope {s3.slope:+.2f})")
    for arr, c in [(R2, "#0072B2"), (R3, "#009E73")]:
        axB.axhspan(arr.min(), arr.max(), color=c, alpha=.07)
    # swan skeletal
    axB.scatter([111.2], [1.764], marker="*", s=300, color="#B22222",
                edgecolor="k", zorder=6)
    axB.scatter([111.2], [0.951], marker="*", s=300, color="#B22222",
                edgecolor="k", zorder=6, label="mute swan")
    axB.set_xscale("log")
    axB.set_xlabel("Tarsometatarsus length TMT (mm)")
    axB.set_ylabel("Segment length ratio")
    axB.text(0.03, 0.97, "(b)", transform=axB.transAxes, fontsize=13,
             fontweight="bold", va="top")
    axB.text(0.5, 0.47, "ratios drift only 10–20% over a 5× size range\n"
             "(narrow bands) — length is the primary design knob",
             transform=axB.transAxes, fontsize=8.5, color="#444",
             ha="center", va="center")
    for arr, sl, c in [(R2, s2, "#0072B2"), (R3, s3, "#009E73")]:
        xf = np.linspace(np.log10(tmt.min()), np.log10(tmt.max()), 50)
        axB.plot(10 ** xf, 10 ** (sl.intercept + sl.slope * xf), color=c,
                 lw=1.0, ls="--", alpha=.7)
    axB.legend(fontsize=7.6, loc="upper left", bbox_to_anchor=(0.02, 0.94),
               framealpha=.9)
    axB.grid(alpha=.25, which="both")

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(os.path.join(args.out, f"fig_allometry_paper.{ext}"),
                    bbox_inches="tight", facecolor="white")
    print(f"[fig] b={b:.3f}±{bci:.3f} R²={r2:.2f} n={len(m)}  "
          f"| ratio slopes r2 {s2.slope:+.3f}, r3 {s3.slope:+.3f}  → {args.out}")


if __name__ == "__main__":
    main()

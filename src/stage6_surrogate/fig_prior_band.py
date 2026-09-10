# -*- coding: utf-8 -*-
"""设计先验主图（更新版）：水鸟实测散点 + PGLS 校正后的先验带。
与旧版 figA_allometry 的差别：
  · 数据源改成 data/birdtree/pgls_data_full.csv（可复现，不再依赖 /tmp）
  · 只保留能挂到 BirdTree 上的水鸟 → n 从 213 变 201（就是 PGLS 用的那批）
  · 主线从 OLS b=0.391 换成 PGLS 中位 b=0.365，先验带用 PGLS 的 sd_res
  · 旧的 OLS 线保留成细虚线，标明「未做亲缘校正」，让改动一眼看得见
用法: python src/stage6_surrogate/fig_prior_band.py [--lang cn|en]
"""
import os, argparse, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="cn", choices=["cn", "en"])
LANG = ap.parse_args().lang
OUT = f"{U}/outputs/ppt_{LANG}"; os.makedirs(OUT, exist_ok=True)
_have = {f.name for f in fm.fontManager.ttflist}
_cjk = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if f in _have), None)
plt.rcParams.update({"font.sans-serif": ([_cjk] if (_cjk and LANG == "cn") else []) + ["DejaVu Sans"],
                     "font.family": "sans-serif", "axes.unicode_minus": False,
                     "font.size": 11, "axes.linewidth": .8,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
CRIM, GRN, GRY = "#8E2A34", "#2E7D5B", "#4a4a4a"
FAM_COL = {"Anatidae": "#3A6EA5", "Phalacrocoracidae": "#4E8D7C", "Podicipedidae": "#B0713F",
           "Gaviidae": "#7A5FA0", "Pelecanidae": "#C05B6B"}
FAM_NAME = {"cn": {"Anatidae": "雁鸭科", "Phalacrocoracidae": "鸬鹚科", "Podicipedidae": "䴙䴘科",
                   "Gaviidae": "潜鸟科", "Pelecanidae": "鹈鹕科"},
            "en": {k: k for k in FAM_COL}}[LANG]
T = {"cn": dict(title="水鸟腿长异速生长律：{n} 种水鸟实测与 ±2.5σ 带",
                xl="体重 (kg)", yl="跗跖骨长 $L_1$ (mm)",
                prior="水鸟拟合  b = {b:.3f}", band="±2.5σ 带  (σ = {s:.3f} dex)",
                geo="几何相似  b = 0.333",
                prod="产品区间 5–30 kg", extrap="12 kg 以上无水鸟数据，属外推",
                eq="拟合结果：  log$_{10}$ $L_1$ = %.3f + %.3f · log$_{10}$ $m$"
                   "        （$L_1$ 单位 mm，$m$ 单位 g；σ = %.3f dex）"),
     "en": dict(title="Waterbird leg allometry: {n} species and the ±2.5σ band",
                xl="body mass (kg)", yl="tarsometatarsus $L_1$ (mm)",
                prior="waterbirds  b = {b:.3f}", band="±2.5σ band  (σ = {s:.3f} dex)",
                geo="geometric similarity  b = 0.333",
                prod="product range 5-30 kg", extrap="no waterbird data above 12 kg - extrapolated",
                eq="fit:  log$_{10}$ $L_1$ = %.3f + %.3f · log$_{10}$ $m$"
                   "        ($L_1$ in mm, $m$ in g;  σ = %.3f dex)")}[LANG]

D = pd.read_csv(f"{U}/data/birdtree/pgls_data_full.csv")
W = D[(D.is_water == 1) & D.logL.notna() & D.log_m.notna()].copy()
x, y = W.log_m.values, W.logL.values                       # log10 g, log10 mm
b_ols, a_ols = np.polyfit(x, y, 1)
sd_ols = float(np.std(y - (a_ols + b_ols*x), ddof=2))
P5 = pd.read_csv(f"{U}/outputs/phylo/S5_prior.csv")
a_p, b_p, sd_p = float(np.median(P5.a)), float(np.median(P5.b)), float(np.median(P5.sd_res))
n = len(W)
print(f"n={n}  OLS b={b_ols:.3f} a={a_ols:.3f} σ={sd_ols:.4f} | PGLS b={b_p:.3f} a={a_p:.3f} σ={sd_p:.4f}")

fig, ax = plt.subplots(figsize=(12.4, 6.8))
xs = np.linspace(x.min()-.12, np.log10(34_000), 240)
xdat = float(x.max())
ax.fill_between(10**xs/1e3, 10**(a_p+b_p*xs-2.5*sd_p), 10**(a_p+b_p*xs+2.5*sd_p),
                color=CRIM, alpha=.075, lw=0, label=T["band"].format(s=sd_p))
for f_, c in FAM_COL.items():
    m = W.Family == f_
    if not m.any(): continue
    ax.scatter(10**x[m.values]/1e3, 10**y[m.values], s=17, c=c, alpha=.85, lw=0,
               label=f"{FAM_NAME[f_]} ({int(m.sum())})", zorder=3)
xm = float(np.median(x))
for bb, ls, lab in ((1/3, "--", T["geo"]),):
    aa = np.median(y) - bb*xm
    ax.plot(10**xs/1e3, 10**(aa+bb*xs), ls=ls, color=GRY, lw=1.7, label=lab, zorder=4)
ax.plot(10**xs/1e3, 10**(a_p+b_p*xs), color=CRIM, lw=3.4,
        label=T["prior"].format(b=b_p), zorder=6)
ax.set_xscale("log"); ax.set_yscale("log")
ax.axvspan(5, 10**xdat/1e3, color=GRN, alpha=.10, lw=0, zorder=1)
ax.axvspan(10**xdat/1e3, 30, color=GRN, alpha=.045, lw=0, hatch="///", ec=GRN, zorder=1)
ax.axvline(10**xdat/1e3, color=GRN, lw=1.2, ls=":", zorder=2)
ylo, yhi = 14, 230
ax.text(np.sqrt(5*30), yhi*.90, T["prod"], ha="center", va="top", fontsize=12,
        fontweight="bold", color=GRN, zorder=8)
ax.text(np.sqrt(10**xdat/1e3*30), 33, T["extrap"], ha="center", va="center",
        fontsize=10.5, color=GRN, rotation=90, zorder=8)
ax.set_ylim(ylo, yhi); ax.set_xlim(0.085, 34)
ax.set_xticks([0.1, 0.3, 1, 3, 10, 30]); ax.set_xticklabels(["0.1", "0.3", "1", "3", "10", "30"])
ax.set_yticks([20, 40, 80, 120]); ax.set_yticklabels(["20", "40", "80", "120"])
ax.set_xlabel(T["xl"], fontsize=13); ax.set_ylabel(T["yl"], fontsize=13)
ax.set_title(T["title"].format(n=n), fontsize=15, fontweight="bold", loc="left", pad=34)
ax.text(0, 1.012, T["eq"] % (a_p, b_p, sd_p), transform=ax.transAxes, ha="left", va="bottom",
        fontsize=14.5, fontweight="bold", color=CRIM,
        bbox=dict(boxstyle="round,pad=.42", fc="#fbf3f4", ec=CRIM, lw=1.6))
h, l = ax.get_legend_handles_labels()
o = [i for i, s in enumerate(l) if "(" in s and s.split("(")[-1].rstrip(")").isdigit()]
r = [i for i in range(len(l)) if i not in o]
lg1 = ax.legend([h[i] for i in o], [l[i] for i in o], loc="upper left",
                fontsize=10.5, frameon=False, handletextpad=.5)
ax.add_artist(lg1)
ax.legend([h[i] for i in r], [l[i] for i in r], loc="upper left", bbox_to_anchor=(.235, 1.0),
          fontsize=10.5, frameon=False, handletextpad=.8)
out = f"{OUT}/{LANG}_F_prior.png"
fig.savefig(out, dpi=170, bbox_inches="tight"); print("→", out)

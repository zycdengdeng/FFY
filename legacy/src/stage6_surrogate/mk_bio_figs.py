# -*- coding: utf-8 -*-
"""第六周汇报 · 生物线图表(HWI × 腿长)"""
import json, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm

FONT = "Noto Sans CJK JP"
plt.rcParams["font.sans-serif"] = [FONT, "WenQuanYi Zen Hei", "DejaVu Sans"]
plt.rcParams.update({"font.family": FONT, "axes.unicode_minus": False,
                     "font.size": 12, "axes.titlesize": 14, "axes.labelsize": 12,
                     "figure.dpi": 160, "savefig.bbox": "tight",
                     "axes.spines.top": False, "axes.spines.right": False})

OUT = "/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
SRC = "/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/run/out"
J = json.load(open(f"{SRC}/hwi_analysis.json"))

FL_ORD = {"Struthioniformes", "Rheiformes", "Casuariiformes", "Apterygiformes", "Sphenisciformes"}
FL_GEN = {"Tachyeres", "Nannopterum", "Rollandia", "Centropelma", "Podilymbus"}
FL_SPP = {"Anas aucklandica", "Anas nesiotis", "Anas chlorotis",
          "Podiceps taczanowskii", "Phalacrocorax harrisi"}
d = pd.read_csv(f"{SRC}/avonet_hwi.csv").drop_duplicates("scientificNameStd")
d = d[(d["BodyMass.Value"] > 0) & d["Hand.wing.Index"].notna() & d["Tarsus.Length"].notna()].copy()
_g = d["scientificNameStd"].str.split().str[0]
d = d[~(d["Order"].isin(FL_ORD) | _g.isin(FL_GEN)
        | d["scientificNameStd"].isin(FL_SPP))].copy()   # 剔除不会飞
d["log_m"] = np.log10(d["BodyMass.Value"])
d["u"] = (np.log10(d["Tarsus.Length"]) - (0.479 + 0.391 * d["log_m"])) / 0.0784
d = d[np.isfinite(d["u"])]
HWI, U = d["Hand.wing.Index"].values, d["u"].values

C_NEG, C_POS, C_HL, C_GREY = "#1b6ca8", "#c0392b", "#e67e22", "#95a5a6"

# ---------- 图 1:主关系 ----------
fig, ax = plt.subplots(figsize=(7.6, 5.2))
hb = ax.hexbin(HWI, U, gridsize=55, cmap="Blues", mincnt=1, bins="log", linewidths=0)
b, a = np.polyfit(HWI, U, 1)
xs = np.linspace(HWI.min(), HWI.max(), 50)
ax.plot(xs, a + b * xs, color=C_POS, lw=2.6, label=f"OLS 斜率 {b:.3f}")
ax.set_xlabel("手翼指数 HWI  (Kipp 距离 / 翼长) —— 飞行效率代理 →")
ax.set_ylabel("腿长残差 u  (相对水鸟异速先验的标准化残差)")
ax.set_title("飞得越好的鸟，腿越短", pad=12, fontweight="bold")
s1 = J["step1"]
ax.text(0.975, 0.955, f"n = {s1['model']['n']:,} 会飞鸟种\n"
        f"r = {s1['r_raw']:.3f}（原始）\n"
        f"r = {s1['r_partial_mass']:.3f}（控制体重后）\n"
        f"β_HWI = {s1['model']['beta'][1]:.4f}   t = {s1['model']['t'][1]:.0f}",
        transform=ax.transAxes, ha="right", va="top", fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=C_GREY, alpha=.93))
ax.axhline(0, color=C_GREY, lw=.9, ls=":")
ax.legend(loc="lower left", frameon=False, fontsize=11)
cb = fig.colorbar(hb, ax=ax, pad=.015); cb.set_label("物种数（对数）", fontsize=10)
fig.savefig(f"{OUT}/fig_hwi_main.png"); plt.close(fig)

# ---------- 图 2:各目内部 ----------
od = pd.DataFrame(J["step3_within_order"]).sort_values("r")
NAME = {"Cuculiformes":"鹃形目","Falconiformes":"隼形目","Suliformes":"鲣鸟目","Galliformes":"鸡形目",
        "Tinamiformes":"䳍形目","Passeriformes":"雀形目","Columbiformes":"鸽形目","Anseriformes":"雁形目",
        "Strigiformes":"鸮形目","Trogoniformes":"咬鹃目","Bucerotiformes":"犀鸟目","Psittaciformes":"鹦形目",
        "Apodiformes":"雨燕目","Gruiformes":"鹤形目","Charadriiformes":"鸻形目","Accipitriformes":"鹰形目",
        "Piciformes":"䴕形目","Coraciiformes":"佛法僧目","Pelecaniformes":"鹈形目","Caprimulgiformes":"夜鹰目",
        "Procellariiformes":"鹱形目","Gaviiformes":"潜鸟目","Podicipediformes":"䴙䴘目","Ciconiiformes":"鹳形目",
        "Sphenisciformes":"企鹅目","Coliiformes":"鼠鸟目","Musophagiformes":"蕉鹃目","Otidiformes":"鸨形目",
        "Phoenicopteriformes":"红鹳目","Pteroclidiformes":"沙鸡目","Cathartiformes":"美洲鹫目",
        "Eurypygiformes":"日鳽目","Mesitornithiformes":"拟鹑目","Phaethontiformes":"鹲形目",
        "Opisthocomiformes":"麝雉目","Leptosomiformes":"鹃鴗目","Cariamiformes":"叫鹤目"}
lab = [f"{NAME.get(o,o)}  (n={n})" for o, n in zip(od["order"], od["n"])]
SIG_POS = {"Strigiformes", "Trogoniformes"}          # t > 2 且 r > 0
cols = [C_HL if o in SIG_POS else (C_NEG if r < 0 else C_GREY)
        for o, r in zip(od["order"], od["r"])]
fig, ax = plt.subplots(figsize=(8.0, 6.9))
y = np.arange(len(od))
ax.barh(y, od["r"], color=cols, height=.72)
for i, (r_, t_) in enumerate(zip(od["r"], od["t"])):
    ax.text(r_ + (0.018 if r_ >= 0 else -0.018), i, f"t={t_:.1f}",
            va="center", ha="left" if r_ >= 0 else "right", fontsize=8.2, color="#666")
ax.set_yticks(y); ax.set_yticklabels(lab, fontsize=9.5)
ax.axvline(0, color="k", lw=1)
ax.set_xlabel("目内部  r(HWI, u)")
n_neg = int((od["r"] < 0).sum())
ax.set_title(f"不是「雀形目 vs 其它」：{n_neg}/{len(od)} 个目内部各自为负",
             pad=10, fontweight="bold")
ax.set_xlim(-0.86, 0.66)
fig.savefig(f"{OUT}/fig_hwi_orders.png"); plt.close(fig)

# ---------- 图 3:类群定位 ----------
fo = pd.DataFrame(J["step6_focus"])
fig, ax = plt.subplots(figsize=(7.6, 5.2))
OFF = {"水鸟(5科)": (-6, 20, "center"), "雁鸭科": (10, -34, "left"),
       "鸡形目": (0, -36, "center"), "全部会飞": (26, 12, "left"),
       "雀形目": (0, 18, "center"), "猛禽3目": (0, 18, "center")}
for _, r in fo.iterrows():
    isw = "水鸟" in r["name"] or "雁鸭" in r["name"]
    dx, dy, ha = OFF.get(r["name"], (0, 16, "center"))
    ax.scatter(r["hwi"], r["u"], s=np.clip(r["n"], 60, 900) ** .62 * 9,
               color=C_NEG if isw else C_GREY, alpha=.85, zorder=3,
               edgecolor="k", linewidth=.7)
    ax.annotate(f"{r['name']}\nb={r['b']:.3f}", (r["hwi"], r["u"]),
                textcoords="offset points", xytext=(dx, dy), ha=ha, fontsize=10.5,
                fontweight="bold" if isw else "normal", color=C_NEG if isw else "#444")
ax.axhline(0, color=C_GREY, lw=.9, ls=":")
ax.set_xlabel("HWI 中位数  →  飞行效率更高")
ax.set_ylabel("腿长残差 u 中位数")
ax.set_title("水鸟：全场 HWI 最高、腿相对最短", pad=12, fontweight="bold")
ax.text(0.02, 0.06, "与「水鸟长腿」的直觉相反 ——\n它们是长距离迁徙的高效飞行者",
        transform=ax.transAxes, fontsize=11, color=C_NEG,
        bbox=dict(boxstyle="round,pad=0.45", fc="#eaf2f8", ec=C_NEG, alpha=.9))
ax.set_ylim(-2.6, 6.2); ax.set_xlim(12, 56)
fig.savefig(f"{OUT}/fig_hwi_clades.png"); plt.close(fig)

# ---------- 图 4:系统发育衰减 → 为什么必须做 PGLS ----------
ag = J["step4_aggregated"]
lv = ["物种"] + [a["level"] for a in ag]
rr = [J["step1"]["r_partial_mass"]] + [a["r"] for a in ag]
nn = [J["step1"]["model"]["n"]] + [a["n_units"] for a in ag]
tt = [J["step1"]["model"]["t"][1]] + [a["t"] for a in ag]
fig, ax = plt.subplots(figsize=(7.6, 4.8))
x = np.arange(len(lv))
bars = ax.bar(x, np.abs(rr), color=[C_NEG, C_NEG, C_HL, C_POS], width=.6)
for i, (v, n, t) in enumerate(zip(rr, nn, tt)):
    ax.text(i, abs(v) + .022, f"|r| = {abs(v):.3f}\nn = {n:,}\nt = {t:.1f}",
            ha="center", fontsize=10.5)
ax.plot(x, np.abs(rr), color="#555", lw=1.6, ls="--", marker="o", ms=5, zorder=4)
ax.set_xticks(x); ax.set_xticklabels(lv)
ax.set_ylabel("|r(HWI, u)|")
ax.set_ylim(0, .82)
ax.set_title("聚合层级越高，相关越弱 —— 系统发育信号的典型指纹", pad=10, fontweight="bold")
ax.text(0.98, 0.94, "目级 t = −2.1，已到显著性边缘\n→ 8831 个物种不是 8831 份独立证据\n→ PGLS 是必需项，不是加分项",
        transform=ax.transAxes, ha="right", va="top", fontsize=11, color=C_POS,
        bbox=dict(boxstyle="round,pad=0.5", fc="#fdf2f0", ec=C_POS, alpha=.95))
fig.savefig(f"{OUT}/fig_hwi_phylo.png"); plt.close(fig)
print("FONT:", FONT, "| n =", len(d))

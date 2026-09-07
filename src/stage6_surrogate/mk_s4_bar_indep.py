# -*- coding: utf-8 -*-
"""支系跳变主图（更新版）：只用「互不嵌套」的 24 次独立跳变。
与旧版 mk_s4_bar 的差别：
  · 数据由 S4_redo40（40 次，其中 16 次嵌套在别的支系里）换成 S4_redo40_indep 的
    independent==True 子集（24 次），因为嵌套的事件不能当独立证据数两遍
  · 相关系数因此由 r=−0.40 变成 r=−0.59（去掉重复计数后反而更强）
  · 高/低飞行效率两半按中位数切，计数随之更新
用法: python src/stage6_surrogate/mk_s4_bar_indep.py [--lang cn|en]
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
                     "font.family": "sans-serif", "axes.unicode_minus": False, "font.size": 12,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
BLUE, RED = "#1b6ca8", "#c0392b"
ORD = {"Tinamiformes": "䳍形目", "Galliformes": "鸡形目", "Anseriformes": "雁形目",
 "Podicipediformes": "䴙䴘目", "Columbiformes": "鸽形目", "Pteroclidiformes": "沙鸡目",
 "Otidiformes": "鸨形目", "Cuculiformes": "鹃形目", "Caprimulgiformes": "夜鹰目",
 "Apodiformes": "雨燕目", "Gruiformes": "鹤形目", "Charadriiformes": "鸻形目",
 "Procellariiformes": "鹱形目", "Ciconiiformes": "鹳形目", "Suliformes": "鲣鸟目",
 "Pelecaniformes": "鹈形目", "Accipitriformes": "鹰形目", "Strigiformes": "鸮形目",
 "Trogoniformes": "咬鹃目", "Bucerotiformes": "犀鸟目", "Coraciiformes": "佛法僧目",
 "Piciformes": "䴕形目", "Falconiformes": "隼形目", "Psittaciformes": "鹦形目",
 "Passeriformes": "雀形目", "Musophagiformes": "蕉鹃目"}
FAM = {"Psittacidae": "鹦鹉科", "Hirundinidae": "燕科", "Laridae": "鸥科", "Procellariidae": "鹱科",
 "Diomedeidae": "信天翁科", "Trogonidae": "咬鹃科", "Alcedinidae": "翠鸟科", "Dicruridae": "卷尾科",
 "Meropidae": "蜂虎科", "Sylviidae": "莺科", "Ardeidae": "鹭科", "Motacillidae": "鹡鸰科",
 "Caprimulgidae": "夜鹰科", "Mimidae": "嘲鸫科", "Accipitridae": "鹰科", "Trochilidae": "蜂鸟科",
 "Phalacrocoracidae": "鸬鹚科", "Platysteiridae": "蓬头鹟科", "Apodidae": "雨燕科",
 "Columbidae": "鸠鸽科", "Scolopacidae": "鹬科", "Cuculidae": "杜鹃科", "Picidae": "啄木鸟科",
 "Furnariidae": "灶鸟科", "Thraupidae": "裸鼻雀科", "Muscicapidae": "鹟科", "Turdidae": "鸫科",
 "Timaliidae": "画眉科", "Pycnonotidae": "鹎科", "Estrildidae": "梅花雀科", "Corvidae": "鸦科",
 "Tyrannidae": "霸鹟科", "Rallidae": "秧鸡科", "Sturnidae": "椋鸟科", "Ploceidae": "织雀科",
 "Oriolidae": "黄鹂科", "Corvidae": "鸦科",
 "Podicipedidae": "䴙䴘科", "Pelecanidae": "鹈鹕科", "Falconidae": "隼科"}
T = {"cn": dict(title="飞行效率越高的支系，演化中越倾向于把腿缩短",
                yl="腿长的支系级跳变 Δu（标准差）", y2="该支系飞行效率 HWI 均值（灰线）",
                up="腿伸长", dn="腿缩短", hi="← 飞行效率高", lo="飞行效率低 →",
                boxhi="飞行效率高的 {k} 支（HWI {a:.0f}–{b:.0f}）\n其中 {s} 支是缩短",
                boxlo="飞行效率低的 {k} 支（HWI {a:.0f}–{b:.0f}）\n其中只有 {s} 支是缩短",
                foot="{n} 次互不嵌套的独立腿长跳变（每支系 ≥20 种）·  按该支系飞行效率从高到低排列  ·  r = {r:+.2f}"),
     "en": dict(title="The better a clade flies, the more it shortens its legs",
                yl="clade-level shift in leg length  Δu  [SD]", y2="clade mean flight efficiency HWI (grey)",
                up="legs longer", dn="legs shorter", hi="← better fliers", lo="poorer fliers →",
                boxhi="{k} better-flying clades (HWI {a:.0f}-{b:.0f})\n{s} of them shortened",
                boxlo="{k} poorer-flying clades (HWI {a:.0f}-{b:.0f})\nonly {s} of them shortened",
                foot="{n} mutually non-nested, independent shifts (clades with ≥20 species) · sorted by clade flight efficiency · r = {r:+.2f}")}[LANG]

S = pd.read_csv(f"{U}/outputs/phylo/S4_redo40_indep.csv")
S = S[S.independent == True].sort_values("mean_HWI", ascending=False).reset_index(drop=True)
def nm(r):
    """按 level 取名 —— 不能只看 top_family:一个「目」级支系的最大科并不代表这个支系。
    （旧版就是这么把 46 个种的整个鲣鸟目错标成鸬鹚科的。）"""
    if r.level == "family":
        return r.top_family if LANG == "en" else FAM.get(r.top_family, r.top_family)
    o = r.top_order
    if LANG == "en":
        return o if r.level == "order" else o + "+"
    return ORD.get(o, o) + ("" if r.level == "order" else "+")
seen, labels = {}, []
for r in S.itertuples():
    n_ = nm(r); seen[n_] = seen.get(n_, 0) + 1
    labels.append(n_ if seen[n_] == 1 else f"{n_}({r.n_tip})")
# 同名的第一个也补上 n,免得两根柱子一个带括号一个不带,看着像两回事
dupe = {k for k, v in seen.items() if v > 1}
labels = [f"{nm(r)}({r.n_tip})" if nm(r) in dupe else labels[i]
          for i, r in enumerate(S.itertuples())]
N = len(S); half = N // 2
hi_, lo_ = S.iloc[:half], S.iloc[half:]
r_all = float(np.corrcoef(S.mean_HWI, S.delta_u)[0, 1])

fig, ax = plt.subplots(figsize=(14.6, 7.4))
x = np.arange(N); cols = [BLUE if v < 0 else RED for v in S.delta_u]
ax.bar(x, S.delta_u, color=cols, width=.70, edgecolor="k", linewidth=.6, zorder=3)
ax.axhline(0, color="#333", lw=1.6, zorder=4)
YLO, YHI = -5.2, 4.3
ax.axhspan(0, YHI, color=RED, alpha=.045, zorder=0); ax.axhspan(YLO, 0, color=BLUE, alpha=.045, zorder=0)
for i, v in enumerate(S.delta_u):
    ax.text(i, v + (.13 if v > 0 else -.13), f"{v:+.1f}", ha="center",
            va="bottom" if v > 0 else "top", fontsize=10, color=cols[i], fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=48, ha="right", fontsize=11.5)
for t, c in zip(ax.get_xticklabels(), cols): t.set_color(c)
ax.set_ylabel(T["yl"], fontsize=13.5, labelpad=10)
ax.set_xlim(-1, N); ax.set_ylim(YLO, YHI)
for sp in ("top", "right"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y", labelsize=12); ax.grid(axis="y", color="#e8eaec", lw=.9, zorder=0)
bx = ax.twinx()
bx.plot(x, S.mean_HWI, color="#6b6b6b", lw=2.0, marker="o", ms=5, zorder=5, alpha=.9)
bx.set_ylabel(T["y2"], fontsize=13, color="#555", labelpad=12)
bx.set_ylim(0, 110); bx.set_yticks([10, 20, 30, 40, 50, 60, 70])
bx.tick_params(colors="#555", labelsize=11.5)
for sp in ("top", "left"): bx.spines[sp].set_visible(False)
bx.spines["right"].set_color("#999")
ax.text(.012, .055, T["hi"], transform=ax.transAxes, fontsize=13, color="#444", fontweight="bold")
ax.text(.62, .055, T["lo"], transform=ax.transAxes, fontsize=13, color="#444", fontweight="bold")
ax.text(.012, .86, T["up"], transform=ax.transAxes, fontsize=14, color=RED, fontweight="bold", va="top")
ax.text(.012, .27, T["dn"], transform=ax.transAxes, fontsize=14, color=BLUE, fontweight="bold", va="top")
ax.axvline(half - .5, color="#777", lw=1.6, ls="--", zorder=2)
ax.text((half-1)/2, 3.75, T["boxhi"].format(k=half, a=hi_.mean_HWI.min(), b=hi_.mean_HWI.max(),
        s=int((hi_.delta_u < 0).sum())), ha="center", va="center", fontsize=12.5, color=BLUE,
        fontweight="bold", bbox=dict(boxstyle="round,pad=.42", fc="#eaf2f8", ec=BLUE, alpha=.92))
ax.text(half + (N-half-1)/2, 3.75, T["boxlo"].format(k=N-half, a=lo_.mean_HWI.min(), b=lo_.mean_HWI.max(),
        s=int((lo_.delta_u < 0).sum())), ha="center", va="center", fontsize=12.5, color=RED,
        fontweight="bold", bbox=dict(boxstyle="round,pad=.42", fc="#fdf2f0", ec=RED, alpha=.92))
ax.text(.50, .012, T["foot"].format(n=N, r=r_all), transform=ax.transAxes, ha="center", va="bottom",
        fontsize=11.5, color="#444", bbox=dict(boxstyle="round,pad=.5", fc="#f7f7f6", ec="#c9ccd1"))
ax.set_title(T["title"], fontsize=17, fontweight="bold", pad=16)
out = f"{OUT}/{LANG}_F_shifts.png"
fig.savefig(out, dpi=165, bbox_inches="tight")
print(f"→ {out}  n={N}  r={r_all:+.3f}  高半缩短 {int((hi_.delta_u<0).sum())}/{half}  "
      f"低半缩短 {int((lo_.delta_u<0).sum())}/{N-half}")

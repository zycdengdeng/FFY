# -*- coding: utf-8 -*-
"""论文 A 主图(单张柱状):40 次腿长跳变,按该支系飞行效率排序"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
 "axes.unicode_minus":False,"font.size":12,"figure.dpi":180,"savefig.bbox":"tight"})
P="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
BLUE,RED="#1b6ca8","#c0392b"
ORD={"Tinamiformes":"䳍形目","Galliformes":"鸡形目","Anseriformes":"雁形目",
 "Podicipediformes":"䴙䴘目","Columbiformes":"鸽形目","Pteroclidiformes":"沙鸡目",
 "Otidiformes":"鸨形目","Cuculiformes":"鹃形目","Caprimulgiformes":"夜鹰目",
 "Apodiformes":"雨燕目","Gruiformes":"鹤形目","Charadriiformes":"鸻形目",
 "Procellariiformes":"鹱形目","Ciconiiformes":"鹳形目","Suliformes":"鲣鸟目",
 "Pelecaniformes":"鹈形目","Accipitriformes":"鹰形目","Strigiformes":"鸮形目",
 "Trogoniformes":"咬鹃目","Bucerotiformes":"犀鸟目","Coraciiformes":"佛法僧目",
 "Piciformes":"䴕形目","Falconiformes":"隼形目","Psittaciformes":"鹦形目",
 "Passeriformes":"雀形目","Musophagiformes":"蕉鹃目"}
FAM={"Psittacidae":"鹦鹉科","Hirundinidae":"燕科","Laridae":"鸥科","Procellariidae":"鹱科",
 "Diomedeidae":"信天翁科","Trogonidae":"咬鹃科","Alcedinidae":"翠鸟科","Dicruridae":"卷尾科",
 "Meropidae":"蜂虎科","Sylviidae":"莺科","Ardeidae":"鹭科","Motacillidae":"鹡鸰科",
 "Caprimulgidae":"夜鹰科","Mimidae":"嘲鸫科","Accipitridae":"鹰科","Trochilidae":"蜂鸟科",
 "Phalacrocoracidae":"鸬鹚科","Platysteiridae":"蓬头鹟科","Apodidae":"雨燕科",
 "Columbidae":"鸠鸽科","Scolopacidae":"鹬科","Cuculidae":"杜鹃科","Picidae":"啄木鸟科",
 "Furnariidae":"灶鸟科","Thraupidae":"裸鼻雀科","Muscicapidae":"鹟科","Turdidae":"鸫科",
 "Timaliidae":"画眉科","Pycnonotidae":"鹎科","Estrildidae":"梅花雀科","Corvidae":"鸦科",
 "Tyrannidae":"霸鹟科","Rallidae":"秧鸡科","Sturnidae":"椋鸟科","Ploceidae":"织雀科"}

S=pd.read_csv(f"{P}/S4_redo40.csv").sort_values("mean_HWI",ascending=False).reset_index(drop=True)
def nm(r):
    return FAM.get(r.top_family) if (r.purity>=0.9 and r.top_family in FAM) else ORD.get(r.top_order,r.top_order)
names=[nm(r) for _,r in S.iterrows()]
seen={}
labels=[]
for n in names:
    seen[n]=seen.get(n,0)+1
    labels.append(n if seen[n]==1 else f"{n}②" if seen[n]==2 else f"{n}③")

fig,ax=plt.subplots(figsize=(15.2,7.6))
x=np.arange(len(S))
cols=[BLUE if v<0 else RED for v in S.delta_u]
ax.bar(x,S.delta_u,color=cols,width=.72,edgecolor="k",linewidth=.6,zorder=3)
ax.axhline(0,color="#333",lw=1.6,zorder=4)
ax.axhspan(0,4.3,color=RED,alpha=.045,zorder=0)
ax.axhspan(-4.9,0,color=BLUE,alpha=.045,zorder=0)
# 柱顶数值
for i,v in enumerate(S.delta_u):
    ax.text(i,v+(0.13 if v>0 else -0.13),f"{v:+.1f}",ha="center",
            va="bottom" if v>0 else "top",fontsize=9,color=cols[i],fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(labels,rotation=52,ha="right",fontsize=11.5)
for t,c in zip(ax.get_xticklabels(),cols): t.set_color(c)
ax.set_ylabel("腿长的支系级跳变 Δu（标准差）",fontsize=13.5,labelpad=10)
ax.set_xlim(-1,len(S)); ax.set_ylim(-4.9,4.3)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.tick_params(axis="y",labelsize=12)
ax.grid(axis="y",color="#e8eaec",lw=.9,zorder=0)

# HWI 折线(次轴)
bx=ax.twinx()
bx.plot(x,S.mean_HWI,color="#6b6b6b",lw=2.0,marker="o",ms=4.5,zorder=5,alpha=.9)
bx.set_ylabel("该支系飞行效率 HWI 均值（灰线）",fontsize=13,color="#555",labelpad=12)
bx.set_ylim(0,110); bx.set_yticks([10,20,30,40,50,60,70])
bx.tick_params(colors="#555",labelsize=11.5)
for sp in ("top","left"): bx.spines[sp].set_visible(False)
bx.spines["right"].set_color("#999")

r=np.corrcoef(S.mean_HWI,S.delta_u)[0,1]
ax.text(0.012,0.055,"← 飞行效率高",transform=ax.transAxes,fontsize=13,color="#444",fontweight="bold")
ax.text(0.60,0.055,"飞行效率低 →",transform=ax.transAxes,fontsize=13,color="#444",fontweight="bold",ha="left")
ax.text(0.012,0.86,"腿伸长",transform=ax.transAxes,fontsize=14,color=RED,fontweight="bold",va="top")
ax.text(0.012,0.235,"腿缩短",transform=ax.transAxes,fontsize=14,color=BLUE,fontweight="bold",va="top")
# 高/低 HWI 两半的计数 —— 让趋势可数,不靠印象
ax.axvline(19.5,color="#777",lw=1.6,ls="--",zorder=2)
ax.text(9.5,3.75,"飞行效率高的 20 支（HWI 35–67）\n其中 15 支是缩短",ha="center",va="center",
        fontsize=12.5,color=BLUE,fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42",fc="#eaf2f8",ec=BLUE,alpha=.9))
ax.text(29.5,3.75,"飞行效率低的 20 支（HWI 12–34）\n其中只有 8 支是缩短",ha="center",va="center",
        fontsize=12.5,color=RED,fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.42",fc="#fdf2f0",ec=RED,alpha=.9))
ax.text(0.50,0.012,f"40 次幅度最大的腿长跳变（每支系 ≥20 种）·  按该支系飞行效率从高到低排列  ·  r = {r:+.2f}",
        transform=ax.transAxes,ha="center",va="bottom",fontsize=11.5,color="#444",
        bbox=dict(boxstyle="round,pad=0.5",fc="#f7f7f6",ec="#c9ccd1"))
ax.set_title("飞行效率越高的支系，演化中越倾向于把腿缩短",fontsize=17,fontweight="bold",pad=16)
fig.savefig(f"{OUT}/fig_S4_bar.png"); print("ok  r=%.3f"%r)

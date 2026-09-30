# -*- coding: utf-8 -*-
"""铁证图:族内比较 —— 同一个科里,飞得更好的那一半,腿更短"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
 "axes.unicode_minus":False,"font.size":12,"figure.dpi":180,"savefig.bbox":"tight"})
P="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
BLUE,RED="#1b6ca8","#c0392b"
CN={"Hydrobatidae":"海燕科","Glareolidae":"燕鸻科","Falconidae":"隼科","Charadriidae":"鸻科",
 "Formicariidae":"蚁鸫科","Phasianidae":"雉科","Thraupidae":"裸鼻雀科","Alcidae":"海雀科",
 "Cuculidae":"杜鹃科","Petroicidae":"鸲鹟科","Pycnonotidae":"鹎科","Fringillidae":"燕雀科",
 "Malaconotidae":"丛鵙科","Columbidae":"鸠鸽科","Threskiornithidae":"鹮科","Ardeidae":"鹭科",
 "Scolopacidae":"鹬科","Mimidae":"嘲鸫科","Trogonidae":"咬鹃科","Maluridae":"细尾鹩莺科",
 "Galbulidae":"鹟䴕科","Strigidae":"鸱鸮科"}
F=pd.read_csv(f"{P}/within_科.csv").sort_values("delta").reset_index(drop=True)
G=pd.read_csv(f"{P}/within_属.csv")
neg=(F.delta<0).sum(); n=len(F)
gneg=(G.delta<0).sum(); gn=len(G)

fig,ax=plt.subplots(figsize=(15.0,7.4))
x=np.arange(n); cols=[BLUE if v<0 else RED for v in F.delta]
ax.bar(x,F.delta,color=cols,width=.82,edgecolor="none",zorder=3)
ax.axhline(0,color="#333",lw=1.6,zorder=4)
ax.axhspan(0,2.2,color=RED,alpha=.05,zorder=0); ax.axhspan(-2.4,0,color=BLUE,alpha=.05,zorder=0)
# 两端的科用文字列表给出,不在柱上标(避免挤成一团)
def lst(df,k,sign):
    return "  ·  ".join(f"{CN.get(r.unit,r.unit)} {r.delta:+.1f}" for _,r in df.head(k).iterrows())
ax.set_xlim(-1.2,n+0.2); ax.set_ylim(-2.65,2.25)
ax.set_xticks([]); ax.tick_params(axis="y",labelsize=12)
ax.set_ylabel("腿长残差之差（标准差）\n高飞行效率半 − 低飞行效率半",fontsize=13,labelpad=10)
ax.set_xlabel(f"{n} 个科（每科 ≥15 种），按差值排序",fontsize=13,labelpad=10)
for sp in ("top","right"): ax.spines[sp].set_visible(False)
ax.grid(axis="y",color="#e8eaec",lw=.9,zorder=0)
ax.text(0.235,0.20,f"腿更短：{neg} 个科",transform=ax.transAxes,fontsize=18,
        color=BLUE,fontweight="bold",ha="center")
ax.text(0.705,0.925,f"腿更长：{n-neg} 个科",transform=ax.transAxes,fontsize=15.5,
        color=RED,fontweight="bold",ha="center")
ax.text(0.014,0.075,"缩短最强的科：  "+lst(F,8,-1),transform=ax.transAxes,
        fontsize=11.5,color=BLUE,va="bottom",
        bbox=dict(boxstyle="round,pad=0.42",fc="#eaf2f8",ec=BLUE,alpha=.85))
ax.text(0.705,0.855,"伸长的科：  "+lst(F.iloc[::-1],4,1),transform=ax.transAxes,
        fontsize=11.5,color=RED,va="top",ha="center",
        bbox=dict(boxstyle="round,pad=0.42",fc="#fdf2f0",ec=RED,alpha=.85))
ax.text(0.014,0.955,
    f"{neg}/{n} = {neg/n:.0%} 的科，族内飞得更好的一半腿更短\n"
    f"（若无关系应为 50%；z = 5.9，p < 1e-8）",
    transform=ax.transAxes,va="top",ha="left",fontsize=14,color=BLUE,fontweight="bold",
    bbox=dict(boxstyle="round,pad=0.55",fc="#eaf2f8",ec=BLUE,lw=1.6))
ax.text(0.014,0.72,
    "做法：每个科内部先扣掉体重的影响，再把该科物种按飞行效率\n"
    "分成高、低两半，比较两半的腿长残差。同科物种亲缘极近、\n"
    "生活方式相似 —— 差别只能来自飞行效率本身。",
    transform=ax.transAxes,va="top",ha="left",fontsize=11.5,color="#444",
    bbox=dict(boxstyle="round,pad=0.5",fc="#f7f7f6",ec="#c9ccd1"))
# 属级内嵌
ix=fig.add_axes([0.625,0.255,0.215,0.185])
ix.bar([0,1],[gneg,gn-gneg],color=[BLUE,RED],width=.62,edgecolor="none")
ix.axhline(gn/2,color="#666",lw=1.5,ls="--")
ix.text(1.58,gn/2,"随机\n期望",fontsize=9,color="#666",va="center")
for i,v in enumerate([gneg,gn-gneg]):
    ix.text(i,v+8,str(v),ha="center",fontsize=11.5,fontweight="bold",color=[BLUE,RED][i])
ix.set_xticks([0,1]); ix.set_xticklabels(["腿更短","腿更长"],fontsize=10.5)
ix.set_ylim(0,gn*0.78); ix.set_yticks([])
ix.set_title(f"更严:属内比较（{gn} 属，每属 ≥6 种）\n{gneg}/{gn} = {gneg/gn:.0%}，z = 4.1",
             fontsize=11,pad=6)
for sp in ("top","right","left"): ix.spines[sp].set_visible(False)
fig.suptitle("同一个科里，飞得更好的那一半，腿更短",fontsize=18,fontweight="bold",y=0.975)
fig.savefig(f"{OUT}/fig_within_clade.png"); print("ok")

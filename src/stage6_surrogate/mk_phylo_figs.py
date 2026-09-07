# -*- coding: utf-8 -*-
"""论文 A · 系统发育证据链两张核心图"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
 "axes.unicode_minus":False,"font.size":12,"axes.titlesize":13.5,"axes.labelsize":12,
 "figure.dpi":160,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
U="/mnt/user-data/uploads/FFY/FFY/outputs/phylo"
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
BLUE,RED,GREY,GOLD,GREEN="#1b6ca8","#c0392b","#95a5a6","#C08A2E","#2E7D5B"
CN={"Phalacrocoracidae":"鸬鹚科","Psittacidae":"鹦鹉科","Hirundinidae":"燕科",
 "Laridae":"鸥科","Trochilidae":"蜂鸟科","Procellariidae":"鹱科","Trogonidae":"咬鹃科",
 "Diomedeidae":"信天翁科","Sylviidae":"莺科","Ardeidae":"鹭科","Tyrannidae":"霸鹟科",
 "Timaliidae":"画眉科","Apodidae":"雨燕科","Accipitridae":"鹰科","Rallidae":"秧鸡科",
 "Scolopacidae":"鹬科","Columbidae":"鸠鸽科","Furnariidae":"灶鸟科","Thraupidae":"裸鼻雀科",
 "Muscicapidae":"鹟科","Turdidae":"鸫科","Corvidae":"鸦科","Estrildidae":"梅花雀科",
 "Pycnonotidae":"鹎科","Anatidae":"雁鸭科","Cuculidae":"杜鹃科",
 "Alcedinidae":"翠鸟科","Picidae":"啄木鸟科","Tyrannidae ":"霸鹟科"}

# ================= 图 1:S4 反复独立缩腿 =================
d=pd.read_csv(f"{U}/S4_shifts.csv")
fig,ax=plt.subplots(figsize=(8.4,5.6))
col=[BLUE if v<0 else RED for v in d.delta_u]
ax.scatter(d.mean_HWI,d.delta_u,s=np.sqrt(d.n_tip)*7+40,c=col,alpha=.72,
           edgecolor="k",linewidth=.7,zorder=3)
b,a=np.polyfit(d.mean_HWI,d.delta_u,1)
xs=np.linspace(d.mean_HWI.min()-3,d.mean_HWI.max()+3,50)
ax.plot(xs,a+b*xs,color="#555",lw=2,ls="--",zorder=2)
ax.axhline(0,color="k",lw=1)
# 手工偏移,避开互相压字
OFF={1:(-42,-4),2:(0,-24),3:(-4,-24),4:(0,-22),5:(0,14),7:(-42,0),9:(38,6),
     6:(-10,14),8:(0,14),10:(0,16),11:(-8,-24)}
SKIP={12}
for _,r in d.iterrows():
    if abs(r.delta_u)<1.4 or int(r["rank"]) in SKIP: continue
    dx,dy=OFF.get(int(r["rank"]),(0,14 if r.delta_u>0 else -22))
    nm=CN.get(r.dominant_family,r.dominant_family)
    if int(r["rank"])==5: nm="鸬鹚科(另一支)"
    ax.annotate(nm,(r.mean_HWI,r.delta_u),textcoords="offset points",xytext=(dx,dy),
        ha="center",fontsize=10.5,color=BLUE if r.delta_u<0 else RED,fontweight="bold")
ax.set_ylim(-4.8,2.6)
r=np.corrcoef(d.mean_HWI,d.delta_u)[0,1]
ax.set_xlabel("该支系的飞行效率 HWI 均值  →")
ax.set_ylabel("腿长的支系级跳变 Δu\n← 缩短        伸长 →")
ax.set_title("腿缩短的事件反复独立发生在高飞行效率支系",pad=12,fontweight="bold")
ax.text(0.975,0.17,f"20 个支系级最大跳变（每支 ≥20 种）\nr = {r:+.2f}\n"
        f"缩短的 15 支：HWI 均值 45.2\n伸长的 5 支：HWI 均值 24.3",
        transform=ax.transAxes,ha="right",va="bottom",fontsize=11,
        bbox=dict(boxstyle="round,pad=0.5",fc="#f6f1f1",ec=GREY,alpha=.95))
ax.text(0.03,0.05,"鸬鹚·鹦鹉·燕·鸥·蜂鸟·鹱\n彼此毫无亲缘关系\n→ 各自独立缩短了腿",
        transform=ax.transAxes,ha="left",va="bottom",fontsize=11,color=BLUE,
        bbox=dict(boxstyle="round,pad=0.5",fc="#eaf2f8",ec=BLUE,alpha=.9))
fig.savefig(f"{OUT}/fig_phylo_shifts.png"); plt.close(fig)

# ================= 图 2:S6 各类群 b vs b_eff =================
C=pd.read_csv(f"{U}/S6_clade_b.csv")
g=C.groupby("clade",sort=False).agg(b=("b","median"),se=("se","median"),n=("n","median"))
g=g.sort_values("b")
fig,ax=plt.subplots(figsize=(8.4,5.0))
y=np.arange(len(g))
ax.barh(y,g.b,xerr=g.se*1.96,color=[GREEN if i==g.index.get_loc("水鸟5科") else BLUE
        for i in range(len(g))],height=.62,
        error_kw=dict(ecolor="#444",capsize=4,lw=1.3),zorder=3)
ax.axvline(0.238,color=RED,lw=2.4,ls="--",zorder=4)
ax.text(0.231,len(g)-0.30,"物理涌现\nb_eff = 0.238",color=RED,ha="right",va="bottom",
        fontsize=11,fontweight="bold")
ax.axvline(0.391,color=GOLD,lw=1.8,ls=":",zorder=4)
ax.text(0.386,-0.45,"项目现用先验 0.391（未校正亲缘）",color=GOLD,ha="right",va="top",fontsize=10.5)
ax.set_yticks(y); ax.set_yticklabels([f"{i}  (n={int(g.n[i])})" for i in g.index],fontsize=11)
for i,(bb,se) in enumerate(zip(g.b,g.se)):
    ax.text(bb+se*1.96+0.006,i,f"{bb:.3f}",va="center",fontsize=10.5,color="#333")
ax.set_xlim(0,0.46); ax.set_ylim(-1.4,len(g)+0.7)
ax.set_xlabel("异速指数 b（系统发育校正，20 棵树中位；误差棒 95%CI）")
ax.set_title("物理选的指数比任何一个鸟类群都平",pad=12,fontweight="bold")
fig.savefig(f"{OUT}/fig_phylo_cladeb.png"); print("saved 2 figs")

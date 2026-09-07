# -*- coding: utf-8 -*-
"""论文 A 主图(单张):腿缩短反复独立发生在高飞行效率支系"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib.colors as mc
from matplotlib.lines import Line2D
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
 "axes.unicode_minus":False,"font.size":12.5,"figure.dpi":180,"savefig.bbox":"tight"})
P="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
BLUE,RED,INK="#1b6ca8","#c0392b","#1A1A1A"
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
 "Columbidae":"鸠鸽科","Scolopacidae":"鹬科","Cuculidae":"杜鹃科","Picidae":"啄木鸟科"}
def nm(r):
    return FAM.get(r.top_family) if (r.purity>=0.9 and r.top_family in FAM) \
           else ORD.get(r.top_order,r.top_order)

E=pd.read_csv(f"{P}/tree_edges.csv"); T=pd.read_csv(f"{P}/tree_tips.csv")
S=pd.read_csv(f"{P}/S4_events_on_tree.csv")

fig,(axT,axE)=plt.subplots(1,2,figsize=(13.4,9.6),sharey=True,
        gridspec_kw=dict(width_ratios=[1.0,1.22],wspace=0.015))

# ---------- 左:树 + HWI 色带 + 目名 ----------
for _,r in E.iterrows():
    axT.plot([r.x0,r.x0],[r.y0,r.y1],color="#c6c9ce",lw=1.4,zorder=1,solid_capstyle="round")
    axT.plot([r.x0,r.x1],[r.y1,r.y1],color="#c6c9ce",lw=1.4,zorder=1,solid_capstyle="round")
XT=T.x.max(); norm=mc.Normalize(10,62); cm=plt.get_cmap("YlGnBu")
for _,r in T.iterrows():
    axT.add_patch(plt.Rectangle((XT+4,r.y-.40),7,.80,color=cm(norm(r.HWI)),ec="white",lw=.7,zorder=3))
    axT.text(XT+15,r.y,ORD.get(r.order,r.order),va="center",ha="left",fontsize=12.5,color=INK)
axT.set_xlim(-4,XT+78); axT.axis("off")
axT.text(0,T.y.max()+2.2,"鸟类系统发育树",fontsize=13,color="#444",fontweight="bold")
axT.text(0,T.y.max()+1.25,"目级 · 每目取一代表种（真实子树）",fontsize=11,color="#777")
axT.text(XT+7.5,T.y.max()+1.25,"飞行\n效率",fontsize=10.5,color="#555",ha="center",va="center")
sm=plt.cm.ScalarMappable(norm=norm,cmap=cm); sm.set_array([])
cb=fig.colorbar(sm,ax=axT,fraction=.020,pad=.015,location="left",aspect=26)
cb.set_label("该目 HWI 中位数（飞行效率）",fontsize=11); cb.ax.tick_params(labelsize=10)

# ---------- 右:Δu 数轴,每目一行 ----------
for _,r in T.iterrows():
    axE.plot([-4.6,2.4],[r.y,r.y],color="#e8eaec",lw=1.0,zorder=1)
axE.axvline(0,color="#555",lw=1.4,zorder=2)
axE.axvspan(-4.7,0,color=BLUE,alpha=.045,zorder=0)
axE.axvspan(0,2.5,color=RED,alpha=.045,zorder=0)
for _,r in S.iterrows():
    axE.scatter(r.delta_u,r.y,s=abs(r.delta_u)*110+55,
        color=BLUE if r.delta_u<0 else RED,alpha=.85,edgecolor="k",lw=.8,zorder=4)
placed={}
for _,r in S.reindex(S.delta_u.abs().sort_values(ascending=False).index).iterrows():
    if abs(r.delta_u)<1.55: continue
    key=round(r.y); dy=17
    near=[]
    for k in (key-1,key,key+1): near+=placed.get(k,[])
    for x0,d0 in near:
        if abs(x0-r.delta_u)<0.85 and d0==dy: dy=-25; break
    placed.setdefault(key,[]).append((r.delta_u,dy))
    axE.annotate(nm(r),(r.delta_u,r.y),textcoords="offset points",xytext=(0,dy),
        ha="center",fontsize=11.5,color=BLUE if r.delta_u<0 else RED,fontweight="bold")
axE.set_xlim(-4.7,2.5); axE.set_ylim(-2.0,T.y.max()+2.9)
axE.set_xlabel("腿长的支系级跳变 Δu（单位：标准差）",fontsize=12.5,labelpad=8)
axE.set_xticks([-4,-3,-2,-1,0,1,2])
for sp in ("top","right","left"): axE.spines[sp].set_visible(False)
axE.tick_params(left=False,labelsize=11.5)
axE.text(-2.35,-1.45,"← 腿缩短",color=BLUE,fontsize=13.5,ha="center",fontweight="bold")
axE.text(1.25,-1.45,"腿伸长 →",color=RED,fontsize=13.5,ha="center",fontweight="bold")
axE.legend(handles=[
    Line2D([],[],marker="o",ls="",color=BLUE,ms=11,mec="k",label="腿缩短　23 次"),
    Line2D([],[],marker="o",ls="",color=RED,ms=11,mec="k",label="腿伸长　17 次")],
    loc="upper left",frameon=False,fontsize=12.5,bbox_to_anchor=(0.02,0.985))
axE.text(0.982,0.982,"每个圆点 = 一次演化跳变事件，大小 = 幅度\n"
    "共 40 次幅度最大的跳变（仅计含 ≥20 种的支系）\n空行 = 该目无跳变进入前 40",
    transform=axE.transAxes,ha="right",va="top",fontsize=10.5,color="#555",
    bbox=dict(boxstyle="round,pad=0.45",fc="#f7f7f6",ec="#c9ccd1"))
fig.suptitle("腿缩短的演化事件反复独立发生在高飞行效率的支系上",
             fontsize=17,fontweight="bold",y=0.965)
fig.savefig(f"{OUT}/fig_S4_tree.png"); print("ok")

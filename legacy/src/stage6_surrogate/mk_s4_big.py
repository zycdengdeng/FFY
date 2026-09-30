# -*- coding: utf-8 -*-
"""论文 A 主图:腿缩短在鸟类系统树上反复独立发生"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
 "axes.unicode_minus":False,"font.size":11.5,"axes.titlesize":13.5,"axes.labelsize":12,
 "figure.dpi":170,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
P="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
BLUE,RED,GREY,INK="#1b6ca8","#c0392b","#9aa0a6","#1A1A1A"
ORD={"Struthioniformes":"鸵鸟目","Tinamiformes":"䳍形目","Galliformes":"鸡形目",
 "Anseriformes":"雁形目","Podicipediformes":"䴙䴘目","Phoenicopteriformes":"红鹳目",
 "Columbiformes":"鸽形目","Pterocliformes":"沙鸡目","Pteroclidiformes":"沙鸡目","Otidiformes":"鸨形目",
 "Cuculiformes":"鹃形目","Caprimulgiformes":"夜鹰目","Apodiformes":"雨燕目",
 "Gruiformes":"鹤形目","Charadriiformes":"鸻形目","Gaviiformes":"潜鸟目",
 "Procellariiformes":"鹱形目","Sphenisciformes":"企鹅目","Ciconiiformes":"鹳形目",
 "Suliformes":"鲣鸟目","Pelecaniformes":"鹈形目","Accipitriformes":"鹰形目",
 "Strigiformes":"鸮形目","Coliiformes":"鼠鸟目","Trogoniformes":"咬鹃目",
 "Bucerotiformes":"犀鸟目","Coraciiformes":"佛法僧目","Piciformes":"䴕形目",
 "Cariamiformes":"叫鹤目","Falconiformes":"隼形目","Psittaciformes":"鹦形目",
 "Passeriformes":"雀形目","Musophagiformes":"蕉鹃目","Opisthocomiformes":"麝雉目",
 "Eurypygiformes":"日鳽目","Phaethontiformes":"鹲形目","Mesitornithiformes":"拟鹑目",
 "Leptosomiformes":"鹃鴗目","Cathartiformes":"美洲鹫目"}
FAM={"Psittacidae":"鹦鹉科","Hirundinidae":"燕科","Laridae":"鸥科","Procellariidae":"鹱科",
 "Diomedeidae":"信天翁科","Trogonidae":"咬鹃科","Alcedinidae":"翠鸟科","Dicruridae":"卷尾科",
 "Meropidae":"蜂虎科","Sylviidae":"莺科","Ardeidae":"鹭科","Motacillidae":"鹡鸰科",
 "Caprimulgidae":"夜鹰科","Mimidae":"嘲鸫科","Accipitridae":"鹰科","Podicipedidae":"䴙䴘科",
 "Trochilidae":"蜂鸟科","Phalacrocoracidae":"鸬鹚科","Platysteiridae":"蓬头鹟科",
 "Apodidae":"雨燕科","Tyrannidae":"霸鹟科","Timaliidae":"画眉科","Scolopacidae":"鹬科"}

E=pd.read_csv(f"{P}/tree_edges.csv"); T=pd.read_csv(f"{P}/tree_tips.csv")
S=pd.read_csv(f"{P}/S4_events_on_tree.csv")

fig=plt.figure(figsize=(14.6,8.6))
gs=fig.add_gridspec(1,2,width_ratios=[1.42,1],wspace=0.30)
ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1])

# ============ 左:系统树 + 事件 ============
for _,r in E.iterrows():
    ax.plot([r.x0,r.x0],[r.y0,r.y1],color="#c9ccd1",lw=1.25,zorder=1,solid_capstyle="round")
    ax.plot([r.x0,r.x1],[r.y1,r.y1],color="#c9ccd1",lw=1.25,zorder=1,solid_capstyle="round")
XT=T.x.max()
# HWI 色带
import matplotlib.colors as mc
norm=mc.Normalize(10,62); cm=plt.get_cmap("YlGnBu")
for _,r in T.iterrows():
    ax.add_patch(plt.Rectangle((XT+3,r.y-.40),5.5,.80,color=cm(norm(r.HWI)),
                 ec="white",lw=.6,zorder=3))
    ax.text(XT+12,r.y,ORD.get(r.order,r.order),va="center",ha="left",fontsize=10.5,color=INK)
# 事件点(按目定位,同目多事件横向错开)
S=S.sort_values("delta_u")
EV0=XT+58
cnt={}
for _,r in S.iterrows():
    k=r.top_order; i=cnt.get(k,0); cnt[k]=i+1
    c=BLUE if r.delta_u<0 else RED
    ax.scatter(EV0+i*6.4,r.y,s=abs(r.delta_u)*95+30,color=c,alpha=.85,
               edgecolor="k",lw=.6,zorder=4)
# 最大的 9 个事件标科名,固定一列,同目上下错开
BIG=S.reindex(S.delta_u.abs().sort_values(ascending=False).index).head(9)
cnt2={}
for _,r in BIG.iterrows():
    k=r.top_order; i=cnt2.get(k,0); cnt2[k]=i+1
    nm=FAM.get(r.top_family) if (r.purity>=0.9 and r.top_family in FAM) else ORD.get(r.top_order,r.top_order)
    ax.text(XT+168,r.y+(0.58*i-0.20),f"{nm} {r.delta_u:+.1f}",
            fontsize=10,va="center",ha="left",
            color=BLUE if r.delta_u<0 else RED,fontweight="bold")
ax.set_xlim(-4,XT+258); ax.set_ylim(-1.4,T.y.max()+2.9)
ax.axis("off")
ax.text(0,T.y.max()+2.1,"鸟类系统发育树（目级，每目取一代表种）",fontsize=11.5,color="#555")
ax.text(XT+3,T.y.max()+1.2,"飞行\n效率",fontsize=9.5,color="#555",ha="left",va="center")
ax.text(XT+58,T.y.max()+1.2,"腿长跳变事件（40 个）",fontsize=9.5,color="#555",ha="left",va="center")
sm=plt.cm.ScalarMappable(norm=norm,cmap=cm); sm.set_array([])
cb=fig.colorbar(sm,ax=ax,fraction=.022,pad=.02,location="left")
cb.set_label("该目 HWI 中位数",fontsize=10); cb.ax.tick_params(labelsize=9)
ax.legend(handles=[Line2D([],[],marker="o",ls="",color=BLUE,ms=10,label="腿缩短（23 个事件）"),
                   Line2D([],[],marker="o",ls="",color=RED,ms=10,label="腿伸长（17 个事件）")],
          loc="lower right",frameon=False,fontsize=10.5,bbox_to_anchor=(1.0,-0.02))
ax.set_title("腿长的大跳变散布在整棵树上 —— 彼此独立",pad=14,fontweight="bold",loc="left")

# ============ 右:Δu vs HWI ============
bx.axhline(0,color="k",lw=1,zorder=1)
b,a=np.polyfit(S.mean_HWI,S.delta_u,1); xs=np.linspace(8,70,50)
bx.plot(xs,a+b*xs,color="#555",lw=2,ls="--",zorder=2)
for _,r in S.iterrows():
    bx.scatter(r.mean_HWI,r.delta_u,s=np.sqrt(r.n_tip)*6+45,
               color=BLUE if r.delta_u<0 else RED,alpha=.72,edgecolor="k",lw=.7,zorder=3)
LB={"Suliformes":(-46,-2),"Psittacidae":(0,-22),"Hirundinidae":(-34,-6),"Laridae":(6,-22),
    "Apodiformes":(-44,-2),"Procellariidae":(40,2),"Diomedeidae":(34,-16),
    "Sylviidae":(-14,20),"Pelecaniformes":(40,-2),"Passeriformes":(30,-6),"Ardeidae":(0,16),
    "Alcedinidae":(-40,-4),"Trogonidae":(0,-22),"Meropidae":(30,-14),"Dicruridae":(-36,4),
    "Accipitridae":(34,2),"Timaliidae":(0,16),"Caprimulgidae":(30,10),"Charadriiformes":(0,16)}
seen=set()
for _,r in S.iterrows():
    if abs(r.delta_u)<1.15: continue
    key=r.top_family if r.top_family in FAM and r.purity>=0.9 else r.top_order
    nm=FAM.get(key,ORD.get(key,key))
    if nm in seen: continue
    seen.add(nm)
    dx,dy=LB.get(key,(0,14 if r.delta_u>0 else -20))
    bx.annotate(nm,(r.mean_HWI,r.delta_u),textcoords="offset points",xytext=(dx,dy),
        ha="center",fontsize=10,color=BLUE if r.delta_u<0 else RED,fontweight="bold")
rr=np.corrcoef(S.mean_HWI,S.delta_u)[0,1]
bx.set_xlabel("该支系飞行效率 HWI 均值  →")
bx.set_ylabel("腿长跳变 Δu\n← 缩短        伸长 →")
bx.set_title("缩短事件集中在高飞行效率支系",pad=14,fontweight="bold",loc="left")
bx.set_xlim(6,72); bx.set_ylim(-4.9,2.9)
bx.text(0.975,0.03,f"40 个支系级最大跳变（每支 ≥20 种）\nr = {rr:+.2f}\n"
        f"缩短 23 支：HWI 均值 42.2\n伸长 17 支：HWI 均值 29.7",
        transform=bx.transAxes,ha="right",va="bottom",fontsize=10.5,
        bbox=dict(boxstyle="round,pad=0.45",fc="#f6f1f1",ec=GREY,alpha=.95))
fig.savefig(f"{OUT}/fig_S4_main.png"); print("ok")

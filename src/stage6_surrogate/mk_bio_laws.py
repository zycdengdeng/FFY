# -*- coding: utf-8 -*-
"""两张新图:① HWI 进腿长公式  ② 觅食层位=腿的用途"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
 "axes.unicode_minus":False,"font.size":12.5,"figure.dpi":180,"savefig.bbox":"tight",
 "axes.spines.top":False,"axes.spines.right":False})
U="/mnt/user-data/uploads/FFY/FFY"
P="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
BLUE,RED,GREEN,GOLD="#1b6ca8","#c0392b","#2E7D5B","#C08A2E"
D=pd.read_csv(f"{U}/data/birdtree/pgls_data_full.csv"); D=D[D.HWI.notna()]

# ============ 图 1:HWI 进公式 ============
fig,(ax,bx)=plt.subplots(1,2,figsize=(13.6,5.6),gridspec_kw=dict(wspace=.26))
y=D.logL.values
X1=np.column_stack([np.ones(len(D)),D.log_m]); b1=np.linalg.lstsq(X1,y,rcond=None)[0]
X2=np.column_stack([np.ones(len(D)),D.log_m,D.HWI]); b2=np.linalg.lstsq(X2,y,rcond=None)[0]
r2=lambda p:1-((y-p)@(y-p))/((y-y.mean())@(y-y.mean()))
p1,p2=X1@b1,X2@b2
for a,p,lab,c,rr in ((ax,p1,"只用体重",GOLD,r2(p1)),(bx,p2,"体重 + 飞行效率 HWI",BLUE,r2(p2))):
    a.hexbin(p,y,gridsize=48,cmap="Blues" if c==BLUE else "YlOrBr",mincnt=1,bins="log",lw=0)
    lo,hi=1.0,2.5; a.plot([lo,hi],[lo,hi],color="#444",lw=2,ls="--")
    a.set_xlim(lo,hi); a.set_ylim(lo,hi)
    a.set_xlabel("公式预测的 log10(跗跖长 mm)"); a.set_ylabel("实测 log10(跗跖长 mm)")
    a.set_title(lab,fontsize=14,fontweight="bold",loc="left",pad=10)
    a.text(.03,.96,f"R² = {rr:.3f}",transform=a.transAxes,va="top",fontsize=19,
           color=c,fontweight="bold")
ax.text(.03,.80,"log10 L1 = 0.851 + 0.310 · log10 m",transform=ax.transAxes,va="top",
        fontsize=12.5,color="#333",bbox=dict(boxstyle="round,pad=.4",fc="white",ec=GOLD))
bx.text(.03,.80,"log10 L1 = 0.957 + 0.341 · log10 m − 0.0061 · HWI",transform=bx.transAxes,
        va="top",fontsize=12.5,color="#333",bbox=dict(boxstyle="round,pad=.4",fc="white",ec=BLUE))
bx.text(.97,.06,"HWI 每 +10 → 腿长 −13%\n麻雀(17)→水鸟(44) → 腿长 −31%",transform=bx.transAxes,
        ha="right",va="bottom",fontsize=12,color=BLUE,fontweight="bold",
        bbox=dict(boxstyle="round,pad=.45",fc="#eaf2f8",ec=BLUE))
fig.suptitle("把飞行效率放进腿长公式：解释力从 67% 提到 79%（n = 8,705）",
             fontsize=16.5,fontweight="bold",y=1.02)
fig.savefig(f"{OUT}/fig_law_hwi.png"); plt.close(fig)

# ============ 图 2:觅食层位 ============
S=pd.read_csv(f"{P}/forstrat.csv")
ORD=["空中","水域","树上","地面"]; COL={"空中":BLUE,"水域":"#3d8fc4","树上":"#9dbf9e","地面":RED}
fig,(ax,bx)=plt.subplots(1,2,figsize=(13.6,5.8),gridspec_kw=dict(width_ratios=[1.15,1],wspace=.28))
pos=np.arange(len(ORD))
data=[S[S.主用途==k].e.values for k in ORD]
bp=ax.boxplot(data,positions=pos,widths=.62,patch_artist=True,showfliers=False,
              medianprops=dict(color="k",lw=2.2))
for p,k in zip(bp["boxes"],ORD): p.set_facecolor(COL[k]); p.set_alpha(.75)
for i,k in enumerate(ORD):
    s=S[S.主用途==k]
    ax.text(i,3.15,f"n={len(s)}\nHWI {s.HWI.median():.0f}",ha="center",fontsize=11,color="#444")
ax.axhline(0,color="#666",lw=1.2,ls=":")
ax.set_xticks(pos); ax.set_xticklabels(ORD,fontsize=14)
ax.set_ylabel("腿长残差（已扣体重）",fontsize=13)
ax.set_xlabel("主要觅食层位 = 腿的主要用途",fontsize=13)
ax.set_ylim(-6.8,4.2)
ax.set_title("① 腿的用途决定腿长",fontsize=14.5,fontweight="bold",loc="left",pad=26)
ax.annotate("",xy=(0,-5.4),xytext=(3,-5.4),arrowprops=dict(arrowstyle="<->",color="#555",lw=1.8))
ax.text(1.5,-6.1,"空中觅食腿最短（HWI 66）→ 地面觅食腿最长（HWI 22）",
        ha="center",fontsize=12,color="#333",fontweight="bold")
# 右:偏相关
R=[("地面",0.293,28),("树上",-0.210,-19),("空中",-0.179,-16),("水域",-0.058,-5)]
yy=np.arange(len(R))[::-1]
bx.barh(yy,[r for _,r,_ in R],color=[COL[k] for k,_,_ in R],height=.6)
bx.axvline(0,color="k",lw=1.4)
for y_,(k,r,t) in zip(yy,R):
    bx.text(r+(0.012 if r>0 else -0.012),y_,f"{r:+.3f}  (t={t:+d})",
            va="center",ha="left" if r>0 else "right",fontsize=12,fontweight="bold",color=COL[k])
bx.set_yticks(yy); bx.set_yticklabels([k for k,_,_ in R],fontsize=14)
bx.set_xlim(-0.60,0.50); bx.set_xlabel("该层位觅食比例 与 腿长残差 的偏相关（控制体重）",fontsize=12.5)
bx.set_title("② 皮尔森偏相关",fontsize=14.5,fontweight="bold",loc="left",pad=12)
bx.text(.98,.30,"水域几乎无关（r = −0.06）\n→ 控制体重后，水鸟的腿\n   并不特别长",
        transform=bx.transAxes,ha="right",va="bottom",fontsize=11.5,color="#333",
        bbox=dict(boxstyle="round,pad=.45",fc="#f6f1f1",ec="#999"))
fig.suptitle("腿是多功能器官：飞行只是拉扯它的力量之一（n = 8,160）",
             fontsize=16.5,fontweight="bold",y=1.005)
fig.savefig(f"{OUT}/fig_law_forstrat.png"); print("ok")

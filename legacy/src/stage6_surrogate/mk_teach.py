# -*- coding: utf-8 -*-
"""教学图:族内比较到底在比什么 —— 以隼科为例"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
 "axes.unicode_minus":False,"font.size":12.5,"figure.dpi":180,"savefig.bbox":"tight"})
U="/mnt/user-data/uploads/FFY/FFY"
P="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/pgls"
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
BLUE,RED="#1b6ca8","#c0392b"
D=pd.read_csv(f"{U}/data/birdtree/pgls_data_full.csv"); D=D[D.HWI.notna()]
s=D[D.Family=="Falconidae"].copy()
med=np.median(s.HWI); s["hi"]=s.HWI>med
s["m_g"]=10**s.log_m; s["tar"]=10**s.logL

fig=plt.figure(figsize=(14.6,6.9))
gs=fig.add_gridspec(1,2,width_ratios=[1.05,1],wspace=0.24)
ax=fig.add_subplot(gs[0,0]); bx=fig.add_subplot(gs[0,1])

# ============ A:隼科内部 ============
for hi,c,lab in ((True,BLUE,f"飞得好的一半（HWI > {med:.0f}）"),
                 (False,RED,f"飞得差的一半（HWI < {med:.0f}）")):
    t=s[s.hi==hi]
    ax.scatter(t.m_g,t.tar,s=62,color=c,alpha=.72,edgecolor="k",lw=.7,zorder=3,label=lab)
    b,a=np.polyfit(np.log10(t.m_g),np.log10(t.tar),1)
    xs=np.logspace(np.log10(t.m_g.min()),np.log10(t.m_g.max()),40)
    ax.plot(xs,10**(a+b*np.log10(xs)),color=c,lw=2.6,zorder=4)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("体重（克，对数轴）",fontsize=13)
ax.set_ylabel("跗跖长 = 腿的主要骨段（毫米，对数轴）",fontsize=13)
ax.set_title("① 一个科内部：隼科 62 种",fontsize=15,fontweight="bold",loc="left",pad=12)
# 两个极端个案
A=s[s.tip=="Micrastur_plumbeus"].iloc[0]; B=s[s.tip=="Falco_concolor"].iloc[0]
for r,c,txt,dxy in ((A,RED,f"林隼 Micrastur plumbeus\nHWI 15 · {A.m_g:.0f} g · 跗跖 {A.tar:.0f} mm",(-10,30)),
                    (B,BLUE,f"烟隼 Falco concolor\nHWI 62 · {B.m_g:.0f} g · 跗跖 {B.tar:.0f} mm",(40,-46))):
    ax.scatter([r.m_g],[r.tar],s=210,facecolor="none",edgecolor=c,lw=2.6,zorder=5)
    ax.annotate(txt,(r.m_g,r.tar),textcoords="offset points",xytext=dxy,fontsize=11,
        color=c,fontweight="bold",ha="center" if dxy[0]<0 else "left",
        arrowprops=dict(arrowstyle="->",color=c,lw=1.6))
ax.annotate("",xy=(B.m_g,B.tar*1.06),xytext=(A.m_g,A.tar*0.94),
            arrowprops=dict(arrowstyle="<->",color="#444",lw=2.0,ls="--"))
ax.text(215,47,"体重更大，\n腿却短一半",fontsize=12.5,color="#333",ha="center",fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.4",fc="white",ec="#888",alpha=.92))
ax.legend(loc="upper left",frameon=False,fontsize=12)
ax.grid(alpha=.22,which="both",lw=.7)
ax.text(0.985,0.045,"同一个科 = 亲缘极近、生活方式相似\n蓝线整体压在红线下方\n→ 同样体重下，飞得好的腿更短",
        transform=ax.transAxes,ha="right",va="bottom",fontsize=11.5,color="#333",
        bbox=dict(boxstyle="round,pad=0.45",fc="#eaf2f8",ec=BLUE))

# ============ B:96 个科 ============
F=pd.read_csv(f"{P}/within_科.csv").sort_values("delta").reset_index(drop=True)
n=len(F); neg=int((F.delta<0).sum())
x=np.arange(n); cols=[BLUE if v<0 else RED for v in F.delta]
bx.bar(x,F.delta,color=cols,width=.85,edgecolor="none",zorder=3)
bx.axhline(0,color="#333",lw=1.5,zorder=4)
bx.set_xticks([]); bx.set_xlabel(f"{n} 个科（每科 ≥15 种），按差值排序",fontsize=13)
bx.set_ylabel("科内  飞得好的一半 − 飞得差的一半\n的腿长差（标准差）",fontsize=12.5)
bx.set_title("② 把这件事对 96 个科各做一遍",fontsize=15,fontweight="bold",loc="left",pad=12)
bx.set_ylim(-2.5,2.2); bx.set_xlim(-1,n)
for sp in ("top","right"): bx.spines[sp].set_visible(False)
bx.grid(axis="y",alpha=.25,lw=.7)
bx.annotate("",xy=(38,-2.15),xytext=(2,-2.15),arrowprops=dict(arrowstyle="<->",color=BLUE,lw=2))
bx.text(20,-2.32,f"{neg} 个科：飞得好的腿更短",color=BLUE,fontsize=13,ha="center",fontweight="bold")
bx.annotate("",xy=(n-1,1.95),xytext=(n-19,1.95),arrowprops=dict(arrowstyle="<->",color=RED,lw=2))
bx.text(n-10,2.02,f"{n-neg} 个科：反过来",color=RED,fontsize=12.5,ha="center",fontweight="bold")
bx.text(0.035,0.775,f"{neg} / {n} = {neg/n:.0%}\n\n若飞行与腿长无关，\n这个比例应该是 50%。\n\nz = 5.9，p < 1e-8",
        transform=bx.transAxes,fontsize=14,color=BLUE,fontweight="bold",va="center",
        bbox=dict(boxstyle="round,pad=0.6",fc="#eaf2f8",ec=BLUE,lw=2))
fig.suptitle("「族内比较」在做什么：同一个科里，飞得好的那一半，腿更短",
             fontsize=17.5,fontweight="bold",y=1.005)
fig.savefig(f"{OUT}/fig_within_teach.png"); print("ok")

# -*- coding: utf-8 -*-
"""隼科一页：两只隼的具体对照 + 全科 62 种的趋势。所有数字由 pgls_data_full.csv 现算。"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import FancyBboxPatch, Circle
_have={f.name for f in fm.fontManager.ttflist}
_cjk=next((f for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei","Droid Sans Fallback")
           if f in _have), None)
plt.rcParams["font.sans-serif"]=([_cjk] if _cjk else [])+["DejaVu Sans"]
plt.rcParams["font.family"]="sans-serif"; plt.rcParams["axes.unicode_minus"]=False
plt.rcParams["savefig.facecolor"]=plt.rcParams["figure.facecolor"]="white"
U=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=f"{U}/outputs/ppt_cn"; os.makedirs(OUT,exist_ok=True)
BLU,ORA,RED,GRY="#1b6ca8","#d98032","#c0392b","#6b6b6b"
A,B,S=0.479,0.391,0.0784   # u 的定义常数（水鸟先验）

D=pd.read_csv(f"{U}/data/birdtree/pgls_data_full.csv"); D=D[D.HWI.notna()&D.u.notna()]
F=D[D.Family=="Falconidae"].copy()
X=np.column_stack([np.ones(len(F)),F.log_m])
F["e"]=F.u.values-X@np.linalg.lstsq(X,F.u.values,rcond=None)[0]
F["L1"]=10**(A+B*F.log_m+S*F.u); F["m_g"]=10**F.log_m
r_all=float(np.corrcoef(F.HWI,F.e)[0,1]); cut=float(F.HWI.median())
hi,lo=F[F.HWI>cut],F[F.HWI<=cut]
d_half=float(np.median(hi.e)-np.median(lo.e))
P=F[F.tip=="Falco_concolor"].iloc[0]; Q=F[F.tip=="Micrastur_plumbeus"].iloc[0]

fig=plt.figure(figsize=(15.0,6.5))
gs=fig.add_gridspec(1,2,width_ratios=[1.05,1],wspace=.16)
# ---------- 左：两只隼 ----------
axL=fig.add_subplot(gs[0,0]); axL.set_xlim(0,1); axL.set_ylim(0,1); axL.axis("off")
axL.set_title("同一个科里的两只隼",fontsize=17,fontweight="bold",loc="left",pad=12)
SC=1/190.0   # mm → 轴坐标（腿画到 62mm 约占 0.65）
for k,(r,name,latin,col,x0,note) in enumerate([
        (P,"烟隼","Falco concolor",BLU,0.24,"开阔天空的长距离迁徙猎手\n高速平飞追捕 —— 腿是纯死重"),
        (Q,"林隼","Micrastur plumbeus",ORA,0.74,"南美雨林树冠层穿梭\n短距爆发 + 枝上站立抓握 —— 腿有用")]):
    L=r.L1*SC
    axL.plot([x0,x0],[0.30,0.30+L],color=col,lw=13,solid_capstyle="round",zorder=3)
    axL.add_patch(Circle((x0,0.29),0.018,fc="#f1eee6",ec="#333",lw=1.4,zorder=4))
    axL.plot([x0-0.15,x0+0.15],[0.275,0.275],color="#8a7f6d",lw=3.5,zorder=2)
    axL.annotate("",xy=(x0-0.085,0.30),xytext=(x0-0.085,0.30+L),
                 arrowprops=dict(arrowstyle="<->",lw=1.8,color="#333"))
    axL.text(x0-0.10,0.30+L/2,f"{r.L1:.0f} mm",rotation=90,ha="right",va="center",
             fontsize=15,fontweight="bold",color=col)
    axL.text(x0,0.94,f"{name}",ha="center",fontsize=19,fontweight="bold",color=col)
    _lat = latin.replace(" ", r"\ ")
    axL.text(x0, 0.885, r"$\it{" + _lat + "}$", ha="center", fontsize=12, color="#444")
    axL.text(x0,0.838,f"飞行效率 HWI = {r.HWI:.0f}\n体重 = {r.m_g:.0f} g",
             ha="center",va="top",fontsize=13.5,fontweight="bold",color="#222",linespacing=1.5)
    axL.text(x0,0.195,note,ha="center",va="top",fontsize=12,color="#333",linespacing=1.6)
axL.text(0.5,0.235,"上面两根按真实比例绘制的，是跗跖骨（腿的主骨）",
         ha="center",va="top",fontsize=11.5,color=GRY)
axL.add_patch(FancyBboxPatch((0.02,0.000),0.96,0.085,boxstyle="round,pad=.012",
                             fc="#eaf3fa",ec=BLU,lw=1.8,transform=axL.transAxes))
axL.text(0.5,0.0425,f"林隼比烟隼还轻 {P.m_g-Q.m_g:.0f} 克，腿却长了 {100*(Q.L1/P.L1-1):.0f}%",
         ha="center",va="center",fontsize=16,fontweight="bold",color=BLU)
# ---------- 右：全科 62 种 ----------
axR=fig.add_subplot(gs[0,1])
axR.scatter(F.HWI,F.e,s=62,c=["#9ec6e0" if h<=cut else "#7fb6d9" for h in F.HWI],
            ec="#2a6ea0",lw=.9,alpha=.9,zorder=3)
z=np.polyfit(F.HWI,F.e,1); xs=np.linspace(F.HWI.min()-2,F.HWI.max()+2,50)
axR.plot(xs,np.polyval(z,xs),color=RED,lw=3,zorder=4)
axR.axhline(0,color="#888",lw=1.2,ls="--")
for r,name,col,ax_xy,ha,va in [(P,"烟隼",BLU,(0.985,0.55),"right","center"),
                               (Q,"林隼",ORA,(0.020,0.965),"left","top")]:
    axR.scatter([r.HWI],[r.e],s=280,c=col,ec="k",lw=1.9,zorder=6)
    axR.annotate(f"{name}   HWI {r.HWI:.0f} · 腿 {r.L1:.0f} mm",xy=(r.HWI,r.e),
                 xytext=ax_xy,textcoords="axes fraction",fontsize=13,fontweight="bold",
                 color=col,ha=ha,va=va,zorder=8,
                 bbox=dict(boxstyle="round,pad=.35",fc="white",ec=col,lw=1.7),
                 arrowprops=dict(arrowstyle="-",lw=1.6,color=col,
                                 connectionstyle="arc3,rad=0.12"))
axR.set_xlabel("飞行效率  HWI",fontsize=13.5)
axR.set_ylabel("腿长残差（已扣掉体重的影响）",fontsize=13.5)
axR.set_title(f"整个隼科 {len(F)} 种，趋势一致",fontsize=17,fontweight="bold",loc="left",pad=12)
axR.grid(alpha=.25)
axR.text(.985,.80,f"相关系数  r = {r_all:+.2f}   (n = {len(F)})",transform=axR.transAxes,
         ha="right",va="top",fontsize=15,fontweight="bold",color=RED,
         bbox=dict(boxstyle="round,pad=.4",fc="white",ec=RED,lw=1.8))
axR.text(.02,.03,f"按 HWI = {cut:.0f} 把科内切两半：\n"
                 f"飞得好的一半中位残差 {np.median(hi.e):+.2f}，另一半 {np.median(lo.e):+.2f}\n"
                 f"差值 Δ = {d_half:+.2f}  —— 这就是族内比较图里隼科那一根柱子",
         transform=axR.transAxes,fontsize=11.5,color="#333",linespacing=1.6,
         bbox=dict(boxstyle="round,pad=.4",fc="#f6f6f6",ec="#bbb"))
for s_ in ("top","right"): axR.spines[s_].set_visible(False)
fig.suptitle("为什么说「同科之内也成立」—— 隼科的一个具体例子",
             fontsize=19,fontweight="bold",y=1.005)
out=f"{OUT}/cn_F_falcon.png"; fig.savefig(out,dpi=165,bbox_inches="tight")
print("→",out, f"| r={r_all:+.3f} Δ={d_half:+.2f} cut={cut:.1f} 烟隼 {P.L1:.1f}mm/{P.m_g:.0f}g 林隼 {Q.L1:.1f}mm/{Q.m_g:.0f}g")

# -*- coding: utf-8 -*-
"""英文版 A0 机械抽象单图（给英文 PPT 用）。几何与 fig_arms.py 的 12 kg 设计点一致。"""
import numpy as np, os, argparse
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon, Arc
AP = argparse.ArgumentParser(); AP.add_argument("--lang", default="en", choices=["cn", "en"])
LANG = AP.parse_args().lang
import matplotlib.font_manager as _fm
_have = {f.name for f in _fm.fontManager.ttflist}
_cjk = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if f in _have), None)
plt.rcParams.update({"font.family": "sans-serif", "axes.unicode_minus": False,
                     "font.sans-serif": ([_cjk] if (_cjk and LANG == "cn") else []) + ["DejaVu Sans"]})
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8b8a85"; CRIM="#8E2A34"
BLUE="#2a78d6"; ORANGE="#eb6834"; GREEN="#1baf7a"; BONE="#20628c"
L1=103.5; L2=1.78*L1; L3=1.07*L1
a1=np.radians(50.); a2=a1+np.radians(180-126.8); a3=a2-np.radians(180-139.2); rf=0.20*L1
P0=np.array([0.,0.3*rf+rf]); P1=P0+L1*np.array([np.cos(a1),np.sin(a1)])
P2=P1+L2*np.array([np.cos(a2),np.sin(a2)]); P3=P2+L3*np.array([np.cos(a3),np.sin(a3)])
u=lambda A,B:(B-A)/np.hypot(*(B-A)); perp=lambda v:np.array([-v[1],v[0]])
def bar(ax,A,B,w=8,col=BONE,z=3.5):
    ax.plot(*zip(A,B),color=col,lw=w,solid_capstyle="round",zorder=z)
    ax.plot(*zip(A,B),color="#fff",lw=w*.3,alpha=.35,solid_capstyle="round",zorder=z+.1)
def pin(ax,P,r=5.,col=INK,z=9,hollow=False):
    ax.add_patch(Circle(P,r,fc=SURF,ec=col,lw=1.7,zorder=z))
    if not hollow: ax.add_patch(Circle(P,1.5,fc=col,ec=col,zorder=z+1))
def spring(ax,A,B,n=7,w=5.5,col=ORANGE,lw=1.9,z=6):
    A=np.array(A,float);B=np.array(B,float);d=B-A;Ln=np.hypot(*d);uu=d/Ln;pp=perp(uu);lead=Ln*.2
    pts=[A,A+uu*lead]
    for i in range(2*n+1):
        t=lead+(Ln-2*lead)*(i+.5)/(2*n+1); pts.append(A+uu*t+pp*w*(1 if i%2==0 else -1))
    pts+=[B-uu*lead,B]; pts=np.array(pts); ax.plot(pts[:,0],pts[:,1],color=col,lw=lw,zorder=z)
def damper(ax,A,B,col=BLUE,lw=1.8,z=6,w=5.):
    A=np.array(A,float);B=np.array(B,float);d=B-A;Ln=np.hypot(*d);uu=d/Ln;pp=perp(uu)
    c1=A+uu*Ln*.38;c2=A+uu*Ln*.72
    ax.plot(*zip(A,c1),color=col,lw=lw,zorder=z)
    ax.add_patch(Polygon(np.array([c1+pp*w,c2+pp*w,c2-pp*w,c1-pp*w]),closed=False,fill=False,ec=col,lw=lw,zorder=z))
    ps=A+uu*Ln*.58
    ax.plot(*zip(ps+pp*w*.85,ps-pp*w*.85),color=col,lw=lw+1.,zorder=z)
    ax.plot(*zip(ps,B),color=col,lw=lw,zorder=z)
def shock(ax,J,A,B,lab,off=52,ec=CRIM):
    A=np.array(A,float);B=np.array(B,float);J=np.array(J,float)
    spring(ax,A,B); damper(ax,A,B); pin(ax,A,3.4); pin(ax,B,3.4)
    ax.plot(*zip(J,A),color=MUTED,lw=1.,ls=":",zorder=2); ax.plot(*zip(J,B),color=MUTED,lw=1.,ls=":",zorder=2)
    M=(A+B)/2
    ax.text(*(M+u(J,M)*off),lab,fontsize=10.5,color=INK,ha="center",va="center",zorder=11,
            bbox=dict(boxstyle="round,pad=0.32",fc=SURF,ec=ec,lw=1.4))
_L = {"cn": dict(ank="踝", kne="膝", hip="髋", body="机身  $m$",
                 gnd="地面：Hertz–Kelvin 接触，$k_c$", ash="踝减震器", hsh="髋减震器",
                 knee_free="膝：自由铰\n无弹簧、无阻尼",
                 seg=["$L_1$  跗跖骨", "$L_2 = r_2 L_1$  胫跗骨", "$L_3 = r_3 L_1$  股骨"]),
      "en": dict(ank="ankle", kne="knee", hip="hip", body="airframe  $m$",
                 gnd="ground: Hertz-Kelvin contact,  $k_c$", ash="ankle shock", hsh="hip shock",
                 knee_free="knee: free hinge\n(no spring, no damper)",
                 seg=["$L_1$", "$L_2 = r_2 L_1$", "$L_3 = r_3 L_1$"])}[LANG]
fig=plt.figure(figsize=(15.4,8.2),dpi=170); fig.patch.set_facecolor(SURF)
ax=fig.add_axes([0.01,0.03,0.40,0.90]); ax.axis("off"); ax.set_aspect("equal")
ax.set_xlim(-215,250); ax.set_ylim(-115,P3[1]+150); ax.set_facecolor(SURF)
ax.fill_between([-215,250],-52,0,color="#c9c4b8",alpha=.4,lw=0); ax.plot([-215,250],[0,0],color=INK2,lw=1.1)
for x in np.arange(-205,250,24):
    z=np.linspace(0,-34,12); xx=x+4*np.where(np.arange(12)%2==0,1,-1); xx[0]=x; xx[-1]=x
    ax.plot(xx,z,color=INK2,lw=.6,alpha=.5)
ax.text(244,-44,_L["gnd"],fontsize=10,color=INK2,ha="right")
ax.add_patch(Circle(P0,rf,fc="#f1eee6",ec=INK,lw=1.6,zorder=5))
bar(ax,P0,P1); bar(ax,P1,P2); bar(ax,P2,P3)
for P,nm,off,ha in ((P1,_L["ank"],(34,-34),"left"),(P2,_L["kne"],(-30,-16),"right"),(P3,_L["hip"],(-34,-42) if LANG=="cn" else (26,-40),"left" if LANG=="en" else "right")):
    pin(ax,P,hollow=(nm==_L["kne"]))
    ax.text(P[0]+off[0],P[1]+off[1],nm,fontsize=13,color=BLUE,fontweight="bold",
            ha=ha,va="center",zorder=12)
for A,B,nm,dx,f in ((P0,P1,_L["seg"][0],-1,.50),
                    (P1,P2,_L["seg"][1],-1,.62),
                    (P2,P3,_L["seg"][2],-1,.30)):
    A=np.array(A,float); B=np.array(B,float); M=A+(B-A)*f; n=perp(u(A,B))
    n = n if n[0]*dx>0 else -n
    ax.text(*(M+n*(52 if LANG=="cn" else 90)),nm,fontsize=11.5,color=BONE,ha="center",va="center",zorder=11,
            bbox=dict(boxstyle="round,pad=.26",fc="#eef4f8",ec=BONE,lw=1.0,alpha=.97))
# ---- 触地姿态角：θA 在踝（L1 与 L2 之间），θK 在膝（L2 与 L3 之间）----
def ang(A, B):
    d = np.asarray(B, float) - np.asarray(A, float)
    return np.degrees(np.arctan2(d[1], d[0])) % 360.
ANG = "#c2185b"
for Pj, t1, t2, lab, rr, lo in (
        (P1, ang(P1, P2), ang(P1, P0), r"$\theta_A$", 52., 0.58),
        (P2, ang(P2, P1), ang(P2, P3), r"$\theta_K$", 52., 0.58)):
    sweep = (t2 - t1) % 360.
    if sweep > 180.: t1, t2, sweep = t2, t1, 360. - sweep
    ax.add_patch(Arc(Pj, 2*rr, 2*rr, angle=0, theta1=t1, theta2=t2,
                     color=ANG, lw=2.6, zorder=8))
    mid = np.radians(t1 + sweep/2.)
    ax.text(Pj[0] + rr*lo*np.cos(mid), Pj[1] + rr*lo*np.sin(mid), lab,
            color=ANG, fontsize=15, fontweight="bold", ha="center", va="center", zorder=13,
            bbox=dict(boxstyle="round,pad=.22", fc="white", ec=ANG, lw=1.3))
shock(ax,P1,P1+u(P1,P2)*84,P1+u(P1,P0)*66,_L["ash"],off=72)
ax.text(P2[0]-46,P2[1]-40,_L["knee_free"],fontsize=10,color=CRIM,
        va="center",ha="right",zorder=11,bbox=dict(boxstyle="round,pad=.3",fc="#fbf0f1",ec=CRIM,lw=1.2))
shock(ax,P3,P3+u(P3,P2)*84,P3+np.array([64,4]),_L["hsh"],off=56)
ax.add_patch(Polygon(np.array([[P3[0]-70,P3[1]+16],[P3[0]+80,P3[1]+16],
                               [P3[0]+80,P3[1]-12],[P3[0]-70,P3[1]-12]]),closed=True,
                     fc="#d5deec",ec=INK,lw=1.3,zorder=2))
ax.add_patch(FancyBboxPatch((P3[0]-58,P3[1]+42),116,46,boxstyle="round,pad=2",
                            fc="#e6ecf6",ec=INK,lw=1.5,zorder=4))
ax.plot([P3[0],P3[0]],[P3[1],P3[1]+42],color=INK,lw=2.4,zorder=3)
ax.text(P3[0],P3[1]+65,_L["body"],fontsize=12,ha="center",va="center",zorder=6)
ax.annotate("", xy=(P3[0]+95,P3[1]+40), xytext=(P3[0]+95,P3[1]+108),
            arrowprops=dict(arrowstyle="-|>",lw=2.4,color=CRIM))
ax.text(P3[0]+103,P3[1]+76,"$v_0$",fontsize=13,color=CRIM,va="center")
ax.set_title(("A0 —— 我们真正拿去仿真的力学抽象" if LANG=="cn" else "A0 - the mechanical abstraction we simulate"),fontsize=15.5,fontweight="bold",loc="left",pad=12)
cx=fig.add_axes([0.435,0.03,0.55,0.90]); cx.axis("off"); cx.set_xlim(0,1); cx.set_ylim(0,1)
SRC="#7a7f87"
def card(y,h,title,rows,col):
    """一张卡：标题 + 三列（符号 / 说明 / 出处）。出处用斜体灰，和数值分开。"""
    cx.add_patch(FancyBboxPatch((0,y),1,h,boxstyle="round,pad=0.008",fc=SURF,ec=col,lw=1.4,
                                transform=cx.transAxes))
    cx.text(.025,y+h-.022,title,fontsize=12.6,color=col,fontweight="bold",va="top")
    yy = y+h-.085
    for sym,desc,src in rows:
        if sym:
            cx.text(.030,yy,sym ,fontsize=11.4,color=INK,va="top")
            cx.text(.185,yy,desc,fontsize=11.4,color=INK,va="top")
        else:
            cx.text(.030,yy,desc,fontsize=11.4,color=INK,va="top")
        cx.text(.545,yy,src ,fontsize=9.8,color=SRC,va="top",style="italic")
        yy -= h/(len(rows)+1.15)

TXT = {"cn": dict(
    dv="设计向量 x（9 维）", oc="工况向量 z（5 维）", fe="可行性判据",
    rows_x=[(r"$L_1$", "跗跖长", "AVONET 水鸟 101 种实测全距 (Tobias 2022)"),
            (r"$r_2,\,r_3$", r"分段比 $L_2/L_1,\;L_3/L_1$", "Watanabe 2017, Auk 134:672，雁鸭科 91 种"),
            (r"$\theta_A,\,\theta_K$", "触地踝角 / 膝角", "水鸟着水视频 12 段触水帧实测全距"),
            ("κ踝, κ膝, κ髋", "无量纲关节刚度", ""),
            (r"$\tau$", r"松弛时间 $c/k$", "")],
    rows_z=[(r"$\log_{10} m$", "机身质量  5 – 30 kg", "产品规格"),
            (r"$v_0$", "下沉速度  0.5 – 2.0 m/s", "Whitehead 2023，绿头鸭着水量级"),
            (r"$\log_{10} k_c$", "地面刚度  软土 – 混凝土", "岩土工程典型值"),
            (r"$g_{cap},\,s_{max}$", "验收上限  10 g, 24 mm", "产品规格")],
    rows_f=[("", "峰值过载 ≤ 10 g　　落震行程 ≤ 24 mm", ""),
            ("", "细长比与腿重预算  均满足", ""),
            ("", "由 Exudyn 多体落震仿真判定", "")]),
 "en": dict(
    dv="Design vector  x  (9 dimensions)", oc="Operating condition  z  (5 dimensions)",
    fe="Feasibility criteria",
    rows_x=[(r"$L_1$", "tarsometatarsus length", "AVONET, 101 waterbird spp. (Tobias 2022)"),
            (r"$r_2,\,r_3$", r"segment ratios $L_2/L_1,\;L_3/L_1$", "Watanabe 2017, Auk 134:672, 91 anatid spp."),
            (r"$\theta_A,\,\theta_K$", "touchdown ankle / knee angle", "waterbird landing video, 12 clips"),
            (r"$\kappa_{a},\kappa_{k},\kappa_{h}$", "dimensionless joint stiffness", ""),
            (r"$\tau$", r"relaxation time $c/k$", "")],
    rows_z=[(r"$\log_{10} m$", "airframe mass  5 - 30 kg", "product specification"),
            (r"$v_0$", "sink rate  0.5 - 2.0 m/s", "Whitehead 2023, mallard water landing"),
            (r"$\log_{10} k_c$", "ground stiffness  soil - concrete", "geotechnical typical values"),
            (r"$g_{cap},\,s_{max}$", "acceptance limits  10 g, 24 mm", "product specification")],
    rows_f=[("", r"peak deceleration $\leq$ 10 g    stroke $\leq$ 24 mm", ""),
            ("", "slenderness and leg-mass budget satisfied", ""),
            ("", "evaluated by an Exudyn multibody drop simulation", "")])}[LANG]
card(0.630,0.330,TXT["dv"],TXT["rows_x"],BLUE)
card(0.330,0.265,TXT["oc"],TXT["rows_z"],GREEN)
card(0.045,0.245,TXT["fe"],TXT["rows_f"],MUTED)
out=f"outputs/ppt_{LANG}/{LANG}_F8_a0.png"; os.makedirs(f"outputs/ppt_{LANG}",exist_ok=True)
fig.savefig(out,facecolor=SURF,bbox_inches="tight"); print("→",out)

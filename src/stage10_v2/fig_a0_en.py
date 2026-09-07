# -*- coding: utf-8 -*-
"""英文版 A0 机械抽象单图（给英文 PPT 用）。几何与 fig_arms.py 的 12 kg 设计点一致。"""
import numpy as np, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False})
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
fig=plt.figure(figsize=(15.4,8.2),dpi=170); fig.patch.set_facecolor(SURF)
ax=fig.add_axes([0.01,0.03,0.40,0.90]); ax.axis("off"); ax.set_aspect("equal")
ax.set_xlim(-215,250); ax.set_ylim(-115,P3[1]+150); ax.set_facecolor(SURF)
ax.fill_between([-215,250],-52,0,color="#c9c4b8",alpha=.4,lw=0); ax.plot([-215,250],[0,0],color=INK2,lw=1.1)
for x in np.arange(-205,250,24):
    z=np.linspace(0,-34,12); xx=x+4*np.where(np.arange(12)%2==0,1,-1); xx[0]=x; xx[-1]=x
    ax.plot(xx,z,color=INK2,lw=.6,alpha=.5)
ax.text(244,-44,"ground: Hertz–Kelvin contact,  $k_c$",fontsize=10,color=INK2,ha="right")
ax.add_patch(Circle(P0,rf,fc="#f1eee6",ec=INK,lw=1.6,zorder=5))
bar(ax,P0,P1); bar(ax,P1,P2); bar(ax,P2,P3)
for P,nm,off,ha in ((P1,"ankle",(26,-6),"left"),(P2,"knee",(30,10),"left"),(P3,"hip",(-26,-24),"right")):
    pin(ax,P,hollow=(nm=="knee"))
    ax.text(P[0]+off[0],P[1]+off[1],nm,fontsize=13,color=BLUE,fontweight="bold",
            ha=ha,va="center",zorder=12)
for A,B,nm,dx,f in ((P0,P1,"$L_1$  tarsometatarsus",-1,.50),
                    (P1,P2,"$L_2 = r_2 L_1$  tibiotarsus",-1,.62),
                    (P2,P3,"$L_3 = r_3 L_1$  femur",-1,.30)):
    A=np.array(A,float); B=np.array(B,float); M=A+(B-A)*f; n=perp(u(A,B))
    n = n if n[0]*dx>0 else -n
    ax.text(*(M+n*52),nm,fontsize=11,color=BONE,ha="center",va="center",zorder=11,
            bbox=dict(boxstyle="round,pad=.26",fc="#eef4f8",ec=BONE,lw=1.0,alpha=.97))
shock(ax,P1,P1+u(P1,P2)*84,P1+u(P1,P0)*66,"ankle shock\n$\\kappa_{ankle}$, $\\tau$",off=72)
ax.text(P2[0]-46,P2[1]-40,"knee: free hinge\n(no spring, no damper)",fontsize=10,color=CRIM,
        va="center",ha="right",zorder=11,bbox=dict(boxstyle="round,pad=.3",fc="#fbf0f1",ec=CRIM,lw=1.2))
shock(ax,P3,P3+u(P3,P2)*84,P3+np.array([64,4]),"hip shock\n$\\kappa_{hip}$, $\\tau$",off=56)
ax.add_patch(Polygon(np.array([[P3[0]-70,P3[1]+16],[P3[0]+80,P3[1]+16],
                               [P3[0]+80,P3[1]-12],[P3[0]-70,P3[1]-12]]),closed=True,
                     fc="#d5deec",ec=INK,lw=1.3,zorder=2))
ax.add_patch(FancyBboxPatch((P3[0]-58,P3[1]+42),116,46,boxstyle="round,pad=2",
                            fc="#e6ecf6",ec=INK,lw=1.5,zorder=4))
ax.plot([P3[0],P3[0]],[P3[1],P3[1]+42],color=INK,lw=2.4,zorder=3)
ax.text(P3[0],P3[1]+65,"airframe  $m$",fontsize=12,ha="center",va="center",zorder=6)
ax.annotate("", xy=(P3[0]+95,P3[1]+40), xytext=(P3[0]+95,P3[1]+108),
            arrowprops=dict(arrowstyle="-|>",lw=2.4,color=CRIM))
ax.text(P3[0]+103,P3[1]+76,"$v_0$",fontsize=13,color=CRIM,va="center")
ax.set_title("A0 — the mechanical abstraction we simulate",fontsize=15.5,fontweight="bold",loc="left",pad=12)
cx=fig.add_axes([0.435,0.03,0.55,0.90]); cx.axis("off"); cx.set_xlim(0,1); cx.set_ylim(0,1)
def card(y,h,title,body,col,mono=False):
    cx.add_patch(FancyBboxPatch((0,y),1,h,boxstyle="round,pad=0.008",fc=SURF,ec=col,lw=1.6,
                                transform=cx.transAxes))
    cx.text(.025,y+h-.022,title,fontsize=12.6,color=col,fontweight="bold",va="top")
    cx.text(.025,y+h-.075,body,fontsize=11.2,color=INK,va="top",linespacing=1.62,
            family=("DejaVu Sans Mono" if mono else "DejaVu Sans"))
card(0.700,0.290,"Design vector  x  (9 dimensions)",
"$L_1$            tarsometatarsus length      <- the biological prior sets this\n"
"$r_2$, $r_3$          segment ratios $L_2/L_1$, $L_3/L_1$\n"
"$\\theta_A$, $\\theta_K$        touchdown ankle / knee angle\n"
"$\\kappa_{ankle}$, $\\kappa_{knee}$, $\\kappa_{hip}$   dimensionless joint stiffness\n"
"$\\tau$             relaxation time  $c/k$",BLUE)
card(0.470,0.215,"Operating condition  z  (5 dimensions)",
"$\\log_{10} m$   airframe mass          5 - 30 kg\n"
"$v_0$        sink rate               0.6 - 2.4 m/s\n"
"$\\log_{10} k_c$  terrain stiffness      soil - concrete\n"
"$g_{cap}$, $s_{max}$  acceptance limits      10 g,  24 mm",GREEN)
card(0.240,0.215,"Why A0 and not the textbook 3-shock leg",
"Ablation over 3 architectures, 15/15 conditions covered by each:\n"
"peak deceleration  3.60 / 3.52 / 3.49 g  -  differences below the\n"
"seed-to-seed noise floor (2.9%).  So performance does not pick the\n"
"architecture; manufacturability does.  A0 drops one shock unit for free\n"
"and gains a 36 mm femur lever arm -> an off-the-shelf spring.",CRIM)
card(0.010,0.215,"Feasibility (what the generator must satisfy)",
"peak deceleration    $\\leq$ 10 g        stroke   $\\leq$ 24 mm\n"
"slenderness  and  leg-mass budget   satisfied\n"
"evaluated by an Exudyn multibody drop simulation,\n"
"2-pass, contact-resolved",MUTED)
out="outputs/ppt_en/en_F8_a0.png"; os.makedirs("outputs/ppt_en",exist_ok=True)
fig.savefig(out,facecolor=SURF,bbox_inches="tight"); print("→",out)

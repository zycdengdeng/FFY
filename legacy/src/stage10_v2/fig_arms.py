# -*- coding: utf-8 -*-
"""三臂消融的机械示意:A 独立+膝簧 / A0 独立−膝簧 / B 耦合−膝簧。
几何用 12 kg 的 v2.1 设计(产品区间 5–30 kg 内的代表点)。"""
import numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch, Polygon
sys.path.insert(0,"/tmp/pipeline_code/src/stage10_v2")
try:
    from cjkfont import setup as _c; _c()
except Exception:
    import matplotlib.font_manager as fm
    for n in ("Noto Sans CJK SC","Noto Sans CJK JP","Droid Sans Fallback"):
        if n in {f.name for f in fm.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"]=[n]+plt.rcParams["font.sans-serif"]; break
plt.rcParams["axes.unicode_minus"]=False
SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8b8a85"; CRIM="#8E2A34"
BLUE="#2a78d6"; ORANGE="#eb6834"; GREEN="#1baf7a"; ROD="#7a4bbf"; BONE="#20628c"

L1=103.5; L2=1.78*L1; L3=1.07*L1
a1=np.radians(50.); a2=a1+np.radians(180-126.8); a3=a2-np.radians(180-139.2)
rf=0.20*L1
P0=np.array([0., 0.3*rf+rf]); P1=P0+L1*np.array([np.cos(a1),np.sin(a1)])
P2=P1+L2*np.array([np.cos(a2),np.sin(a2)]); P3=P2+L3*np.array([np.cos(a3),np.sin(a3)])
u=lambda A,B:(B-A)/np.hypot(*(B-A)); perp=lambda v:np.array([-v[1],v[0]])

def bar(ax,A,B,w=8,col=BONE,z=3.5):
    ax.plot(*zip(A,B),color=col,lw=w,solid_capstyle="round",zorder=z)
    ax.plot(*zip(A,B),color="#ffffff",lw=w*.30,alpha=.35,solid_capstyle="round",zorder=z+.1)
def pin(ax,P,r=5.,col=INK,z=9,hollow=False):
    ax.add_patch(Circle(P,r,fc=SURF,ec=col,lw=1.7,zorder=z))
    if not hollow: ax.add_patch(Circle(P,1.5,fc=col,ec=col,zorder=z+1))
def spring(ax,A,B,n=7,w=5.5,col=ORANGE,lw=1.7,z=6):
    A=np.array(A,float);B=np.array(B,float);d=B-A;Ln=np.hypot(*d);uu=d/Ln;pp=perp(uu);lead=Ln*.2
    pts=[A,A+uu*lead]
    for i in range(2*n+1):
        t=lead+(Ln-2*lead)*(i+.5)/(2*n+1); pts.append(A+uu*t+pp*w*(1 if i%2==0 else -1))
    pts+=[B-uu*lead,B]; pts=np.array(pts); ax.plot(pts[:,0],pts[:,1],color=col,lw=lw,zorder=z)
def damper(ax,A,B,col=BLUE,lw=1.6,z=6,w=5.):
    A=np.array(A,float);B=np.array(B,float);d=B-A;Ln=np.hypot(*d);uu=d/Ln;pp=perp(uu)
    c1=A+uu*Ln*.38;c2=A+uu*Ln*.72
    ax.plot(*zip(A,c1),color=col,lw=lw,zorder=z)
    ax.add_patch(Polygon(np.array([c1+pp*w,c2+pp*w,c2-pp*w,c1-pp*w]),closed=False,fill=False,ec=col,lw=lw,zorder=z))
    ps=A+uu*Ln*.58
    ax.plot(*zip(ps+pp*w*.85,ps-pp*w*.85),color=col,lw=lw+1.,zorder=z)
    ax.plot(*zip(ps,B),color=col,lw=lw,zorder=z)
def shock(ax,J,A,B,lab,adj=False,off=52,col_s=ORANGE):
    A=np.array(A,float);B=np.array(B,float);J=np.array(J,float)
    spring(ax,A,B,col=col_s); damper(ax,A,B); pin(ax,A,3.4); pin(ax,B,3.4)
    ax.plot(*zip(J,A),color=MUTED,lw=1.,ls=":",zorder=2); ax.plot(*zip(J,B),color=MUTED,lw=1.,ls=":",zorder=2)
    M=(A+B)/2; ax.text(*(M+u(J,M)*off),lab,fontsize=8.4,color=INK,ha="center",va="center",zorder=11,
        bbox=dict(boxstyle="round,pad=0.30",fc=SURF,ec=(CRIM if adj else MUTED),lw=1.2))
def bracket(ax,P):
    ax.add_patch(Polygon(np.array([[P[0]-70,P[1]+16],[P[0]+80,P[1]+16],
                                   [P[0]+80,P[1]-12],[P[0]-70,P[1]-12]]),
                         closed=True,fc="#d5deec",ec=INK,lw=1.3,zorder=2))
def body(ax,P):
    ax.add_patch(FancyBboxPatch((P[0]-56,P[1]+42),112,44,boxstyle="round,pad=2",
                                fc="#e6ecf6",ec=INK,lw=1.5,zorder=4))
    ax.plot([P[0],P[0]],[P[1],P[1]+42],color=INK,lw=2.4,zorder=3)
    ax.text(P[0],P[1]+64,"机身 m",fontsize=9.5,color=INK,ha="center",va="center",zorder=6)
    ax.plot([P[0],P[0]],[P[1]+86,P[1]+150],color=MUTED,lw=2.4,alpha=.5,zorder=1)
def ground(ax,x0,x1):
    ax.fill_between([x0,x1],-52,0,color="#c9c4b8",alpha=.4,lw=0)
    ax.plot([x0,x1],[0,0],color=INK2,lw=1.1)
    for x in np.arange(x0+14,x1,24):
        z=np.linspace(0,-34,12); xx=x+4*np.where(np.arange(12)%2==0,1,-1); xx[0]=x; xx[-1]=x
        ax.plot(xx,z,color=INK2,lw=.6,alpha=.5)
def skeleton(ax,knee_free=False):
    ax.add_patch(Circle(P0,rf,fc="#f1eee6",ec=INK,lw=1.6,zorder=5))
    bar(ax,P0,P1); bar(ax,P1,P2); bar(ax,P2,P3)
    for P,nm,dx,hl in ((P1,"踝",-1,False),(P2,"膝",1,knee_free),(P3,"髋",-1,False)):
        pin(ax,P,hollow=hl)
        ax.text(P[0]+dx*15,P[1]-11,nm,fontsize=10.5,color=BLUE,fontweight="bold",
                ha=("right" if dx<0 else "left"),zorder=10)
    bracket(ax,P3); body(ax,P3); ground(ax,-135,215)

fig=plt.figure(figsize=(17.2,9.4),dpi=185); fig.patch.set_facecolor(SURF)
fig.suptitle("三臂消融的机械构型:A → A0 去掉一套减震器,A0 → B 再用连杆锁掉一个自由度",
             fontsize=14.2,color=CRIM,x=.022,ha="left",y=.975)
fig.text(.022,.932,"A0 与 B 只差一根连杆 —— 所以 A0→B 就是「锁自由度」的纯净代价。几何取 12 kg(产品区间 5–30 kg 内)。",
         fontsize=9.6,color=INK2)

RES={"A":("3.60 g","15/15"),"A0":("3.52 g","15/15"),"B":("3.49 g","15/15")}
TITLES={"A":("A  独立 + 膝簧","3 自由度 · 3 套减震器 · 变量 (κ踝, κ膝, τ),κ髋固定"),
        "A0":("A0  独立 − 膝簧","3 自由度 · 2 套减震器 · 变量 (κ踝, κ髋, τ),κ膝 = 0"),
        "B":("B  耦合 − 膝簧","2 自由度 · 2 套减震器 · 同 A0,加连杆")}
TCOL={"A":MUTED,"A0":BLUE,"B":ROD}

for k,arm in enumerate(("A","A0","B")):
    ax=fig.add_axes([0.015+k*0.245,0.055,0.235,0.80]); ax.set_facecolor(SURF); ax.axis("off")
    ax.set_aspect("equal"); ax.set_xlim(-140,225); ax.set_ylim(-110,P3[1]+185)
    skeleton(ax,knee_free=(arm!="A"))
    # 踝:三臂都有,且是唯一可调的那套
    shock(ax,P1,P1+u(P1,P2)*84,P1+u(P1,P0)*66,"踝:可调",adj=True,off=50)
    if arm=="A":
        shock(ax,P2,P2+u(P2,P1)*80,P2+u(P2,P3)*55,"膝:可调",off=44)
        shock(ax,P3,P3+u(P3,P2)*40,P3+np.array([64,4]),"髋:固定",off=40)
    else:
        ax.text(P2[0]-20,P2[1]-30,"膝:自由铰\n无弹簧无阻尼",fontsize=8.4,color=CRIM,va="center",ha="right",
                zorder=11,bbox=dict(boxstyle="round,pad=0.28",fc="#fbf0f1",ec=CRIM,lw=1.1))
        shock(ax,P3,P3+u(P3,P2)*84,P3+np.array([64,4]),"髋:可调",adj=False,off=52)
    if arm=="B":
        nf=perp(u(P3,P2)); nf=-nf if nf[0]>0 else nf
        ns=perp(u(P2,P1)); ns=-ns if ns[0]>0 else ns
        Pb=P3+nf*50.; Q=P2+ns*50.
        ax.plot(*zip(P2,Q),color=BONE,lw=4.,zorder=3.6)
        bar(ax,Pb,Q,w=5.5,col=ROD,z=4); pin(ax,Pb,3.6); pin(ax,Q,3.6)
        ax.text(*(Pb+Q)/2+np.array([-20,0]),"耦合连杆 144mm\n耦合比 −1.22",fontsize=8.6,color=ROD,
                ha="right",va="center",fontweight="bold",zorder=11,
                bbox=dict(boxstyle="round,pad=0.28",fc=SURF,ec=ROD,lw=1.2))
    t,sub=TITLES[arm]
    ax.set_title(f"{t}\n{sub}",fontsize=11.2,color=TCOL[arm],loc="left",pad=6)
    g,cov=RES[arm]
    ax.text(-136,-66,f"12 kg 结果:峰值中位 {g} · 覆盖 {cov}",fontsize=9.6,color=INK,
            bbox=dict(boxstyle="round,pad=0.34",fc="#f2f0ea",ec=TCOL[arm],lw=1.3))

cx=fig.add_axes([0.755,0.055,0.235,0.80]); cx.axis("off"); cx.set_xlim(0,1); cx.set_ylim(0,1)
def card(y,h,title,body_,col):
    cx.add_patch(FancyBboxPatch((0,y),1,h,boxstyle="round,pad=0.008",fc=SURF,ec=col,lw=1.4,
                                transform=cx.transAxes))
    cx.text(.03,y+h-.02,title,fontsize=10.3,color=col,fontweight="bold",va="top")
    cx.text(.03,y+h-.058,body_,fontsize=8.5,color=INK,va="top",linespacing=1.62)
card(0.760,0.235,"消融结果:三臂等价",
"峰值中位 3.60 / 3.52 / 3.49 g,覆盖全部 15/15。\n"
"A→A0 与 A0→B 的净代价均 ≤5%,且符号在两个\n"
"种子之间翻转 —— 而同一条臂换种子自己就差\n"
"2.9%(最大 8.7%)。差异落在噪声地板以下。\n\n"
"⇒ 性能不是判据,架构由制造性单独决定。", GREEN)
card(0.520,0.225,"A → A0:膝那套可以白拿掉",
"膝减震器去掉后,膝仍是自由铰(3 自由度不变),\n"
"只是不再有弹簧和阻尼。性能没有可测变化。\n\n"
"这与更早\"膝是平坦方向\"的观察一致,\n"
"现在有正式消融背书:三套减震器变两套,免费。", BLUE)
card(0.275,0.230,"A0 → B:锁自由度也不要钱",
"连杆把小腿铰接到机身支座上,跨过膝,\n"
"构成 机身–股骨–小腿–连杆 四连杆:\n"
"膝角成为髋角的函数,自由度 3 → 2。\n\n"
"代价同样在噪声内。但引入闭链,需全行程\n"
"校核间隙 —— 换来的只是远端质量更低。", ROD)
card(0.010,0.250,"选型建议:A0",
"                 A       A0        B\n"
"减震器          3 套     2 套     2 套\n"
"股骨可用力臂   24 mm    36 mm    36 mm\n"
"髋簧规格      1385     616      616  kN/m\n"
"闭链风险        无       无       有\n\n"
"A0 免费拿到 36mm 力臂与可买的簧,零件最少,\n"
"不引入闭链。B 只在远端质量成硬指标时才用。", CRIM)

fig.savefig("fig_三臂构型.png",facecolor=SURF,bbox_inches="tight")
print("→ fig_三臂构型.png")

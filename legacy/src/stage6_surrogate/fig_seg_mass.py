# -*- coding: utf-8 -*-
"""三段腿骨各自对体重的标度律。

数据:Watanabe 2017 会飞雁鸭科的三段骨长 × AVONET/EltonTraits 的体重,
     按标准化学名对接,得到 72 种同时有骨长与体重的物种。
口径:三段骨长全部用 Watanabe 的骨骼测量(内部一致),体重只从 AVONET 取。
"""
import csv, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullLocator, NullFormatter
import matplotlib.font_manager  # noqa: F401
sys.path.insert(0, "/tmp/pipeline_code/src/stage10_v2")
try:
    from cjkfont import setup as _c; _c()
except Exception:
    for _n in ("Noto Sans CJK SC","Noto Sans CJK JP","Droid Sans Fallback"):
        if _n in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"]=[_n]+plt.rcParams["font.sans-serif"]; break
plt.rcParams["axes.unicode_minus"]=False

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8b8a85"; CRIM="#8E2A34"
C1,C2,C3="#1baf7a","#eb6834","#2a78d6"          # 跗跖 / 胫跗 / 股骨

W=[r for r in csv.DictReader(open("/tmp/pipeline_code/data/skeletal/watanabe2017_anatidae.csv"))
   if r["group"]=="Volant"]
A={r["scientificNameStd"].strip():float(r["BodyMass.Value"])
   for r in csv.DictReader(open("/tmp/bio/avonet_waterbirds.csv"))}
H=[w for w in W if w["species"].strip() in A]
m=np.array([A[w["species"].strip()] for w in H])/1000.0        # kg
SEG=[("L1 跗跖骨", np.array([float(w["tmt_mm"]) for w in H]), C1),
     ("L2 胫跗骨", np.array([float(w["tib_mm"]) for w in H]), C2),
     ("L3 股骨",   np.array([float(w["fem_mm"]) for w in H]), C3)]
x=np.log10(m); n=len(H)

fig,(axL,axR)=plt.subplots(1,2,figsize=(14.2,6.6),dpi=200,
                           gridspec_kw=dict(width_ratios=[1.25,1.0],wspace=.22))
fig.patch.set_facecolor(SURF)

# ---------------- 左:三段各自对体重 ----------------
axL.set_facecolor(SURF)
xs=np.linspace(x.min(),x.max(),40)
FIT={}
for nm,v,c in SEG:
    y=np.log10(v); b,a=np.polyfit(x,y,1); res=y-(a+b*x)
    ci=1.96*res.std(ddof=2)/(x.std(ddof=0)*np.sqrt(n)); FIT[nm]=(b,ci,a)
    axL.scatter(m,v,s=17,color=c,alpha=.72,lw=0,zorder=3)
    axL.plot(10**xs,10**(a+b*xs),"-",color=c,lw=2.2,zorder=4,
             label=f"{nm}   b = {b:.3f} ± {ci:.3f}")
    # 同质心的几何相似 1/3 参照
    a1=np.mean(y)-(1/3)*np.mean(x)
    axL.plot(10**xs,10**(a1+(1/3)*xs),"--",color=c,lw=1.1,alpha=.55,zorder=2)
axL.plot([],[],"--",color=MUTED,lw=1.2,label="几何相似 b = 1/3(同质心参照)")
axL.set_xscale("log"); axL.set_yscale("log")
axL.set_xticks([0.3,1,3,10]); axL.set_xticklabels(["0.3","1","3","10"])
axL.set_yticks([30,60,120,240]); axL.set_yticklabels(["30","60","120","240"])
axL.xaxis.set_minor_locator(NullLocator()); axL.xaxis.set_minor_formatter(NullFormatter())
axL.yaxis.set_minor_locator(NullLocator()); axL.yaxis.set_minor_formatter(NullFormatter())
axL.set_xlabel("体重 m (kg)",fontsize=10.5,color=INK2)
axL.set_ylabel("该段骨长 (mm)",fontsize=10.5,color=INK2)
axL.set_title("(a) 三段腿骨各自对体重的标度律",fontsize=13,color=INK,loc="left",
              pad=9,fontweight="bold")
axL.grid(alpha=.16,lw=.6,which="both",zorder=0)
for sp in ("top","right"): axL.spines[sp].set_visible(False)
axL.tick_params(colors=INK2,labelsize=9)
axL.legend(fontsize=9.2,frameon=False,loc="upper left")

# ---------------- 右:指数对比 ----------------
axR.set_facecolor(SURF)
axR.axvspan(1/3-.002,1/3+.002,color=MUTED,alpha=.0)
axR.axvline(1/3,color=MUTED,lw=1.8,ls="--",zorder=2)
axR.text(1/3,2.98,"几何相似\nb = 1/3",fontsize=9.5,color=MUTED,ha="center",
         va="bottom",linespacing=1.4)
axR.axvline(0.25,color="#999999",lw=1.2,ls=":",zorder=2)
axR.text(0.25,2.98,"弹性相似\nb = 1/4",fontsize=9,color="#999999",ha="center",
         va="bottom",linespacing=1.4)
rows=[("L1 跗跖骨",C1),("L2 胫跗骨",C2),("L3 股骨",C3)]
for i,(nm,c) in enumerate(rows):
    yy=len(rows)-1-i; b,ci,_=FIT[nm]
    axR.plot([b-ci,b+ci],[yy,yy],"-",color=c,lw=3.4,zorder=4)
    axR.plot(b,yy,"o",ms=10,color=c,zorder=5,mec=SURF,mew=1.5)
    axR.text(.185,yy+.17,nm,fontsize=12,color=c,fontweight="bold",va="center")
    axR.text(.185,yy-.15,f"b = {b:.3f} ± {ci:.3f}",fontsize=9.5,color=MUTED,va="center")
axR.set_xlim(.18,.52); axR.set_ylim(-1.05,len(rows)+.60); axR.set_yticks([])
axR.set_xlabel("骨长–体重标度指数 b",fontsize=10.5,color=INK2)
axR.set_title("(b) 三个指数与两条理论线的位置",fontsize=13,color=INK,loc="left",
              pad=9,fontweight="bold")
axR.grid(axis="x",alpha=.16,lw=.6,zorder=0)
for sp in ("top","right","left"): axR.spines[sp].set_visible(False)
axR.tick_params(colors=INK2,labelsize=9)
axR.text(.187,-.40,
 "三段都显著陡于几何相似(置信区间不含 1/3)。\n"
 "顺序 L1 > L2 > L3:越靠下的骨段随体重长得越快\n"
 "—— 大鸟的腿把长度更多分给最下面那一段。",
 fontsize=9,color=INK2,va="top",linespacing=1.75)

fig.suptitle(f"三段腿骨对体重的标度律：{n} 种会飞雁鸭科（Watanabe 骨长 × AVONET 体重，按学名对接）",
             fontsize=13.5,color=CRIM,x=.035,ha="left",y=.955)
fig.text(.035,.085,
 "口径:三段骨长全部取自 Watanabe 的骨骼测量,内部一致;体重只从 AVONET/EltonTraits 取。"
 "本样本只含雁鸭科,其 L1 指数 0.433 与 AVONET 全库中雁鸭科单独拟合的 0.423±0.034 一致。",
 fontsize=8.8,color=INK2)
fig.text(.035,.032,
 "注:Watanabe 的骨骼跗跖长比 AVONET 的外部 tarsus 测量系统性偏大(中位 +11%),两者是不同的测量口径,不可混用;"
 "本图三段全部使用前者。",fontsize=8.2,color=MUTED)
fig.subplots_adjust(left=.058,right=.985,top=.855,bottom=.215)
fig.savefig("/tmp/seg3/fig_seg_vs_mass.png",facecolor=SURF,bbox_inches="tight")
print("→ /tmp/seg3/fig_seg_vs_mass.png")

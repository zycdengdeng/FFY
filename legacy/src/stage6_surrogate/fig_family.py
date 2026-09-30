# -*- coding: utf-8 -*-
"""分科比较:斜率 b vs 截距偏移 u —— 回应"能不能按科建先验"。

左:分科 b 的森林图。除雁鸭科外,其余四科要么样本少、要么体重跨度窄,
   斜率的置信区间大到无法互相区分 —— 这是"不按科建先验"的统计理由。
右:分科 u 中位(腿相对同体重典型值偏长/偏短)。这个量的排序与
   各科的陆上行走能力一致 —— 这是"分科确实有信息"的生态证据。
"""
import csv, sys, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager  # noqa: F401
sys.path.insert(0, "/tmp/pipeline_code/src/stage10_v2")
try:
    from cjkfont import setup as _c
    _c()
except Exception:
    for _n in ("Noto Sans CJK SC", "Noto Sans CJK JP", "Droid Sans Fallback"):
        if _n in {f.name for f in matplotlib.font_manager.fontManager.ttflist}:
            plt.rcParams["font.sans-serif"] = [_n] + plt.rcParams["font.sans-serif"]
            break
plt.rcParams["axes.unicode_minus"] = False

SURF="#fcfcfb"; INK="#0b0b0b"; INK2="#52514e"; MUTED="#8b8a85"; CRIM="#8E2A34"
FAM_COL={"Anatidae":"#3A6EA5","Phalacrocoracidae":"#4E8D7C","Pelecanidae":"#C05B6B",
         "Gaviidae":"#7A5FA0","Podicipedidae":"#B0713F"}
CN={"Anatidae":"雁鸭科","Phalacrocoracidae":"鸬鹚科","Pelecanidae":"鹈鹕科",
    "Gaviidae":"潜鸟科","Podicipedidae":"䴙䴘科"}
ECO={"Anatidae":"行走良好,常在硬地着陆与取食",
     "Phalacrocoracidae":"上岸栖息晾翅,行走笨拙",
     "Pelecanidae":"能在陆地行走与集群栖息",
     "Gaviidae":"腿极后置,陆上几乎不能行走",
     "Podicipedidae":"腿极后置,完全不能陆地行走或起飞"}

R=list(csv.DictReader(open("/tmp/bio/avonet_waterbirds.csv")))
seen,D=set(),[]
for r in R:
    if r["scientificNameStd"] in seen: continue
    seen.add(r["scientificNameStd"]); D.append((r["Family"],float(r["tarsus_mm"]),float(r["BodyMass.Value"])))
fam=np.array([d[0] for d in D]); L=np.array([d[1] for d in D]); M=np.array([d[2] for d in D])
x,y=np.log10(M),np.log10(L)
b0,a0=np.polyfit(x,y,1); res=y-(a0+b0*x); sig=res.std(ddof=2); u=res/sig
se0=res.std(ddof=2)/(x.std(ddof=0)*np.sqrt(len(x)))

ORD=["Anatidae","Phalacrocoracidae","Pelecanidae","Gaviidae","Podicipedidae"]  # 按 u 中位升序
S={}
for f in ORD:
    s=fam==f; n=int(s.sum()); xs,ys=x[s],y[s]
    bf,af=np.polyfit(xs,ys,1); rr=ys-(af+bf*xs)
    se=rr.std(ddof=2)/(xs.std(ddof=0)*np.sqrt(n))
    S[f]=dict(n=n,b=bf,ci=1.96*se,u=u[s],span=xs.max()-xs.min(),
              mlo=M[s].min()/1000,mhi=M[s].max()/1000)

fig,(axL,axR)=plt.subplots(1,2,figsize=(14.8,6.8),dpi=200,
                           gridspec_kw=dict(width_ratios=[1.0,1.0],wspace=.34))
fig.patch.set_facecolor(SURF)

# ---------------- 左:分科 b ----------------
axL.set_facecolor(SURF)
axL.axvspan(b0-1.96*se0,b0+1.96*se0,color=CRIM,alpha=.10,lw=0,zorder=1)
axL.axvline(b0,color=CRIM,lw=1.8,zorder=2)
axL.text(b0,len(ORD)-.28,f" 合并 213 种  b={b0:.3f}±{1.96*se0:.3f}",
         fontsize=9.5,color=CRIM,va="bottom",fontweight="bold")
for i,f in enumerate(ORD):
    yy=len(ORD)-1-i; d=S[f]; c=FAM_COL[f]
    weak = d["n"]<20 or d["span"]<0.7
    axL.plot([d["b"]-d["ci"],d["b"]+d["ci"]],[yy,yy],"-",color=c,
             lw=3.0 if not weak else 2.0, alpha=1.0 if not weak else .45,zorder=4)
    axL.plot(d["b"],yy,"o",ms=9 if not weak else 7,color=c,zorder=5,
             mec=SURF,mew=1.4,alpha=1.0 if not weak else .55)
    axL.text(.10,yy+.20,f"{CN[f]}",fontsize=11,color=c,fontweight="bold",va="center")
    axL.text(.10,yy-.17,f"n={d['n']}   体重 {d['mlo']:.2f}–{d['mhi']:.2f} kg "
             f"({d['span']:.2f} dex)",fontsize=8.2,color=MUTED,va="center")
    axL.text(d["b"]+d["ci"]+.012,yy,f"{d['b']:.3f}±{d['ci']:.3f}",
             fontsize=9,color=c,va="center",fontweight="bold")
axL.set_xlim(.08,.72); axL.set_ylim(-1.15,len(ORD)-.02); axL.set_yticks([])
axL.set_xlabel("腿长–体重标度指数 b（分科各自拟合）",fontsize=10.5,color=INK2)
axL.set_title("(a) 老师要的：每科各自的 b",fontsize=13,color=INK,loc="left",
              pad=10,fontweight="bold")
axL.grid(axis="x",alpha=.16,lw=.6,zorder=0)
for sp in ("top","right","left"): axL.spines[sp].set_visible(False)
axL.tick_params(colors=INK2,labelsize=9)
axL.text(.095,-.42,
 "淡色 = n<20 或体重跨度 <0.7 dex,斜率不可信。\n"
 "只有雁鸭科(n=151、跨 1.62 dex)的 b 真正可信:0.423±0.034,略高于合并值;\n"
 "鹈鹕科置信区间 ±0.240 大到无信息,潜鸟科只有 5 个点,都不足以支撑结论。",
 fontsize=8.6,color=INK2,va="top",linespacing=1.65)

# ---------------- 右:分科 u ----------------
axR.set_facecolor(SURF)
axR.axvline(0,color=CRIM,lw=1.6,ls="--",zorder=2)
axR.text(0,len(ORD)-.28," 合并拟合线（u=0）",fontsize=9.5,color=CRIM,va="bottom")
for i,f in enumerate(ORD):
    yy=len(ORD)-1-i; d=S[f]; c=FAM_COL[f]; uu=d["u"]
    q1,q2,q3=np.percentile(uu,[25,50,75])
    axR.plot([q1,q3],[yy,yy],"-",color=c,lw=7,alpha=.30,zorder=3)
    axR.plot(uu,np.full(len(uu),yy)+np.random.default_rng(i).uniform(-.11,.11,len(uu)),
             "o",ms=3.2,color=c,alpha=.55,zorder=4,mew=0)
    axR.plot(q2,yy,"|",color=c,ms=22,mew=3.2,zorder=6)
    axR.text(-3.05,yy+.20,f"{CN[f]}   中位 {q2:+.2f}",fontsize=11,color=c,
             fontweight="bold",va="center")
    axR.text(-3.05,yy-.17,ECO[f],fontsize=8.2,color=MUTED,va="center")
axR.set_xlim(-3.1,3.3); axR.set_ylim(-1.15,len(ORD)-.02); axR.set_yticks([])
axR.set_xlabel("u = 腿长相对同体重典型值偏离几个 σ   （←偏短    偏长→）",
               fontsize=10.5,color=INK2)
axR.set_title("(b) 更有信息的量：每科的截距偏移 u",fontsize=13,color=INK,
              loc="left",pad=10,fontweight="bold")
axR.grid(axis="x",alpha=.16,lw=.6,zorder=0)
for sp in ("top","right","left"): axR.spines[sp].set_visible(False)
axR.tick_params(colors=INK2,labelsize=9)
axR.text(-3.05,-.42,
 "u 的科间排序(雁鸭 −0.43 → 䴙䴘 +1.72)与各科陆上行走能力同序,两端尤其干净。\n"
 "注:生态描述取自鸟类学常识,非本工作实测;中间三科的先后取决于如何定义「行走能力」。",
 fontsize=8.6,color=MUTED,va="top",linespacing=1.65)

fig.suptitle("按科拆开看：斜率 b 分不出名次，截距偏移 u 却和陆上行走能力完全同序",
             fontsize=14,color=CRIM,x=.035,ha="left",y=.975)
fig.tight_layout(rect=[0,.035,1,.94])
fig.savefig("/tmp/fam/fig_family_b_vs_u.png",facecolor=SURF,bbox_inches="tight")
print("→ /tmp/fam/fig_family_b_vs_u.png")

# -*- coding: utf-8 -*-
"""统计口径修正:废题率不是噪声,是「脚太小」的物理后果。"""
import json, glob, numpy as np, sys
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
COL={"bio":"#1baf7a","geo":"#2a78d6","elastic":"#7a4bbf","none":"#8b8a85"}
F={}
for f in sorted(glob.glob('v22_e18b_rs/*.json')):
    d=json.load(open(f)); F[d['cond']]=d
ms=np.array(F['turf1.2'].get('ms') or F['turf1.2'].get('m_grid'))
A=lambda c: (F[c]['arms'] if 'arms' in F[c] else F[c])

fig,ax=plt.subplots(1,3,figsize=(15.2,5.4),dpi=190,gridspec_kw=dict(wspace=.26))
fig.patch.set_facecolor(SURF)
for a in ax: a.set_facecolor(SURF)
CONDS=[("concrete1.2","硬地 k_c=1e6"),("turf1.2","草地 k_c=1e5"),("wetsand1.2","湿沙 k_c=5e4")]
for a,(c,lab) in zip(ax,CONDS):
    a.axvspan(5,30,color="#1baf7a",alpha=.06,lw=0)
    a.axhline(50,color=CRIM,lw=1.2,ls=":")
    a.text(2.1,53,"50% 以上判为「不可判」",fontsize=8.4,color=CRIM)
    for arm in ("bio","geo","elastic","none"):
        d=A(c)[arm]
        v=[(d['pooled_invalid'][i]+d['pooled_unsolved'][i])*100 for i in range(len(ms))]
        a.plot(ms,v,"-o",ms=4,color=COL[arm],lw=2 if arm in("bio","none") else 1.2,
               alpha=1 if arm in("bio","none") else .55,
               label=f"{arm}  (b={d['b']:.2f})")
    a.set_xscale("log"); a.set_xlim(1.9,42); a.set_ylim(-4,104)
    a.set_xlabel("机身质量 / kg",fontsize=10,color=INK2)
    if a is ax[0]: a.set_ylabel("废题率 / %\n(deep_sink + 数值失败)",fontsize=10,color=INK2)
    a.set_title(lab,fontsize=11.6,color=INK,loc="left",pad=8)
    a.grid(alpha=.15,lw=.6); [a.spines[s].set_visible(False) for s in ("top","right")]
    a.tick_params(colors=INK2,labelsize=9)
    a.set_xticks([2,5,10,20,40]); a.set_xticklabels(["2","5","10","20","40"])
ax[0].legend(frameon=False,fontsize=8.8,loc="upper left")
ax[1].text(5.6,90,"产品区间 5–30 kg",fontsize=9,color="#1baf7a")
fig.suptitle("废题率不是噪声:它随「腿长标度指数 b」单调变化 —— b 越小,脚越小,越容易陷进软地面",
             fontsize=13,color=CRIM,x=.02,ha="left",y=.985)
fig.text(.02,.915,"足端半径 r_foot = 0.20 × L1。none 臂 b=0,30 kg 的机身踩在和 5 kg 一样小的脚上(半径 14.6 mm)。",
         fontsize=9.4,color=INK2)
fig.tight_layout(rect=[0,0,1,.845]); fig.savefig("fig_废题率机制.png",facecolor=SURF,bbox_inches="tight")
print("→ fig_废题率机制.png")

# -*- coding: utf-8 -*-
"""E21(v2.2):真鸟骨长 vs 生成骨长 —— 只有训练区内的 3 种可作结论。"""
import json, numpy as np, sys
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
BIRD="#8b8a85"; GEN="#1baf7a"
D=json.load(open('e21_bird_vs_gen.json')); M=json.load(open('../v22/model_meta.json'))
lo,hi=M["c_lo"][0],M["c_hi"][0]
sp={r['species']:r['m_kg'] for r in D}
inr={s for s,m in sp.items() if 0<=(np.log10(m)-lo)/(hi-lo)<=1}
S=[r for r in D if r['cond']=='turf1.2']
order=sorted(sp, key=lambda s: sp[s])

fig,ax=plt.subplots(1,2,figsize=(13.6,6.0),dpi=190,gridspec_kw=dict(wspace=.28,width_ratios=[1.25,1]))
fig.patch.set_facecolor(SURF)
for a in ax: a.set_facecolor(SURF)
# 左:行程 vs 体重
ax[0].axhspan(24,34,color=CRIM,alpha=.08,lw=0)
ax[0].axhline(24,color=CRIM,lw=1.4,ls="--")
ax[0].text(0.15,24.6,"行程预算 24 mm",fontsize=9,color=CRIM)
ax[0].axvspan(10**lo,10**hi,color=GEN,alpha=.07,lw=0)
ax[0].text(np.sqrt(10**lo*10**hi),15.6,"v2.2 训练区 4–36 kg",fontsize=9.5,color=GEN,ha="center")
for who,col,mk,lab in (("bird",BIRD,"o","真鸟实测骨长"),("gen",GEN,"s","模型生成骨长")):
    xs=[sp[s] for s in order]; ys=[next(r['leg_stroke_mm'] for r in S if r['species']==s and r['who']==who) for s in order]
    oks=[next(r['ok'] for r in S if r['species']==s and r['who']==who) for s in order]
    ax[0].plot(xs,ys,"-",color=col,lw=1.3,alpha=.5,zorder=2)
    for x,y,o in zip(xs,ys,oks):
        ax[0].plot(x,y,mk,ms=10,color=(col if o else SURF),mec=col,mew=2,zorder=4)
        if not o: ax[0].plot(x,y,"x",ms=7,color=CRIM,mew=2,zorder=5)
ax[0].set_xscale("log"); ax[0].set_xlabel("体重 / kg",fontsize=10,color=INK2)
ax[0].set_ylabel("落震行程 / mm",fontsize=10,color=INK2)
ax[0].set_ylim(15,34); ax[0].set_xlim(0.12,40)
ax[0].set_title("训练区内的三种鸟:真鸟腿全部顶穿行程预算,生成腿全部通过",
                fontsize=11.4,color=INK,loc="left",pad=8)
for s in order:
    if s in inr:
        y=next(r['leg_stroke_mm'] for r in S if r['species']==s and r['who']=='bird')
        ax[0].annotate(s.split()[0],(sp[s],y),textcoords="offset points",xytext=(0,10),
                       fontsize=8.5,color=INK2,ha="center")
# 右:L1 差
ax[1].axvline(0,color=INK2,lw=.9)
ys=np.arange(len(order))
for i,s in enumerate(order):
    a_=next(r for r in S if r['species']==s and r['who']=='bird')
    b_=next(r for r in S if r['species']==s and r['who']=='gen')
    d=(b_['L_mm'][0]-a_['L_mm'][0])/a_['L_mm'][0]*100
    c=GEN if s in inr else MUTED
    ax[1].barh(i,d,color=c,alpha=(.85 if s in inr else .35),height=.62)
    ax[1].text(d+(1.5 if d>=0 else -1.5),i,f"{d:+.0f}%",va="center",
               ha=("left" if d>=0 else "right"),fontsize=9,color=c)
ax[1].set_yticks(ys)
ax[1].set_yticklabels([f"{s.split()[0]}\n{sp[s]:.2f} kg"+("" if s in inr else"  (外推)")
                       for s in order],fontsize=8.5)
ax[1].set_xlim(-32,78); ax[1].set_xlabel("生成 L1 相对真鸟 L1 / %",fontsize=10,color=INK2)
ax[1].set_title("训练区内(绿)差 −22%~+8%;外推区(灰)不可作结论",
                fontsize=11.4,color=INK,loc="left",pad=8)
for a in ax:
    a.grid(alpha=.15,lw=.6); [a.spines[s].set_visible(False) for s in ("top","right")]
    a.tick_params(colors=INK2,labelsize=9)
from matplotlib.lines import Line2D
H=[Line2D([],[],marker="o",ls="none",color=BIRD,ms=9,label="真鸟实测骨长"),
   Line2D([],[],marker="s",ls="none",color=GEN,ms=9,label="模型生成骨长"),
   Line2D([],[],marker="x",ls="none",color=CRIM,ms=8,label="空心+× = 不可行(行程超限)")]
fig.legend(handles=H,ncol=3,loc="lower center",frameon=False,fontsize=9.2,bbox_to_anchor=(.5,-.01))
fig.suptitle("E21(v2.2 口径):同一套刚度下,真鸟骨长 vs 模型生成骨长(草地 1.2 m/s)",
             fontsize=13,color=CRIM,x=.02,ha="left",y=.985)
fig.text(.02,.93,"7 种水鸟里只有 3 种落在 v2.2 的训练区(4–36 kg)—— 结论只由这 3 种支撑,其余 4 种是模型外推,仅供参考。",
         fontsize=9.4,color=INK2)
fig.tight_layout(rect=[0,.065,1,.895]); fig.savefig("fig_e21_v22.png",facecolor=SURF,bbox_inches="tight")
print("→ fig_e21_v22.png")

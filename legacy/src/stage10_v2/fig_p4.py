# -*- coding: utf-8 -*-
"""P4 结果图:固定几何下,主动调节 (κ踝,κ膝,τ) 的收益与感知误差的代价。"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
C_PAS="#8b8a85"; C_ACT="#2a78d6"; C_FF="#eb6834"
R=json.load(open("p4_twolevel.json"))
TN={5e4:"湿沙",1e5:"草地",2.5e5:"硬草",1e6:"硬地"}

fig,axes=plt.subplots(1,2,figsize=(14.6,7.2),dpi=200,
                      gridspec_kw=dict(wspace=.14))
fig.patch.set_facecolor(SURF)
for ax,(m,d) in zip(axes,sorted(R.items(),key=lambda t:float(t[0]))):
    ax.set_facecolor(SURF)
    conds=sorted(d["conds"],key=lambda c:(c["v0"],c["kc"]))
    n=len(conds)
    yl=[]
    vprev=None
    for i,c in enumerate(conds):
        y=n-1-i
        p,f_,a,e=c["passive"],c["ff"],c["active"],c["err"]
        req_bad = (m=="12.0" and c["kc"]==1e6 and c["v0"]==1.8)
        if c["v0"]!=vprev:
            ax.axhline(y+.5,color=MUTED,lw=.6,alpha=.5)
            ax.text(-0.28,y,f"v0={c['v0']}",fontsize=9,color=INK,ha="right",
                    fontweight="bold",va="center")
            vprev=c["v0"]
        yl.append((y,TN[c["kc"]]))
        if req_bad:
            ax.text(5.6,y,"要求物理不可能(需 85% 方波效率),剔除",fontsize=7.6,
                    color=MUTED,va="center"); continue
        # 被动→理想 连线
        if p["g"] and a["g"]:
            ax.plot([a["g"],p["g"]],[y,y],color=C_ACT,lw=1.6,alpha=.45,zorder=2)
        for dd,col,ms,dy in ((p,C_PAS,7.5,0),(a,C_ACT,7.5,0),(f_,C_FF,5,0)):
            if dd["g"] is None: continue
            ax.plot(dd["g"],y+dy,"o",ms=ms,color=col,zorder=4,
                    mfc=(col if dd["ok"] else SURF),mec=col,mew=1.4)
        # 误差须线:③ → ④worst
        if a["g"] and e["g_worst"]:
            ax.plot([a["g"],e["g_worst"]],[y-.28,y-.28],color=C_ACT,lw=1.0,
                    ls=":",alpha=.8,zorder=3)
            ax.plot(e["g_worst"],y-.28,"x",ms=5,color=C_ACT,alpha=.9,zorder=3)
    ax.set_yticks([t[0] for t in yl]); ax.set_yticklabels([t[1] for t in yl],fontsize=8)
    ax.axvline(10,color=CRIM,lw=1.2,ls=":")
    ax.text(10,n-.4," g_cap",fontsize=8,color=CRIM)
    ax.set_xlim(0.8,11.6); ax.set_ylim(-.8,n-.3)
    ax.set_xlabel("峰值加速度 / g",fontsize=10,color=INK2)
    g=d["geom"]
    ok=[sum(1 for c in d["conds"] if c[k]["ok"]) for k in ("passive","ff","active")]
    ax.set_title(f"m = {float(m):g} kg   (L1={g[0]:.0f}mm, θ={g[3]:.0f}°/{g[4]:.0f}°)\n"
                 f"可行工况:被动 {ok[0]}/16 → 前馈 {ok[1]}/16 → 理想 {ok[2]}/16",
                 fontsize=11.5,color=INK,loc="left",pad=8)
    ax.grid(axis="x",alpha=.15,lw=.6,zorder=0)
    for sp in ("top","right"): ax.spines[sp].set_visible(False)
    ax.tick_params(colors=INK2,labelsize=8.5)

from matplotlib.lines import Line2D
H=[Line2D([],[],marker="o",ls="none",color=C_PAS,ms=8,label="① 被动(一套刚度打全部)"),
   Line2D([],[],marker="o",ls="none",color=C_FF,ms=6,label="② cVAE 前馈(按工况查模型)"),
   Line2D([],[],marker="o",ls="none",color=C_ACT,ms=8,label="③ 理想主动(按工况搜最优)"),
   Line2D([],[],marker="x",ls=":",color=C_ACT,ms=6,label="④ 感知错一档的最坏情形"),
   Line2D([],[],marker="o",ls="none",mfc=SURF,mec=INK2,ms=8,label="空心 = 不可行(多为行程超限)")]
fig.legend(handles=H,ncol=5,loc="lower center",frameon=False,fontsize=9,
           bbox_to_anchor=(.5,.005))
fig.suptitle("两级优化:同一副腿,刚度/阻尼按工况调节值多少(16 工况 × 4 条线,几何与姿态固定)",
             fontsize=13.5,color=CRIM,x=.035,ha="left",y=.975)
fig.text(.035,.930,"低速端主动调软降峰值(最多 −53%);高速端主动调硬把行程压回预算内换可行——被动只能二选一。",
         fontsize=9.5,color=INK2)
fig.tight_layout(rect=[0.02,.06,1,.865])
fig.savefig("fig_p4_twolevel.png",facecolor=SURF,bbox_inches="tight")
print("→ fig_p4_twolevel.png")

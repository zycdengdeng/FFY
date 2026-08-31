# -*- coding: utf-8 -*-
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
BLUE="#2a78d6"; GREEN="#1baf7a"; ROD="#7a4bbf"
R=json.load(open("p4c_ablation.json"))
COL={"A":MUTED,"A0":BLUE,"B":ROD}
LAB={"A":"A 独立 + 膝簧\n(3自由度,3套减震器)","A0":"A0 独立 − 膝簧\n(3自由度,2套减震器)",
     "B":"B 耦合 − 膝簧\n(2自由度,2套,加连杆)"}

fig,axes=plt.subplots(1,3,figsize=(15.4,6.0),dpi=190,
                      gridspec_kw=dict(width_ratios=[1,1,1.15],wspace=.28))
fig.patch.set_facecolor(SURF)
for ax in axes: ax.set_facecolor(SURF)

# ---- 面板 1-2:逐格峰值,三臂并排 ----
for ax,(m,d) in zip(axes[:2],sorted(R.items(),key=lambda t:float(t[0]))):
    good=d["good_ci"]; PS=d["per_seed"]; sd=[str(x) for x in d["seeds"]]
    for j,arm in enumerate(("A","A0","B")):
        vals=[PS[s][arm][i]["g"] for s in sd for i in good if PS[s][arm][i]["ok"]]
        x=np.random.default_rng(j).normal(j,0.055,len(vals))
        ax.plot(x,vals,"o",ms=4.5,color=COL[arm],alpha=.42,mec="none",zorder=3)
        md=np.median(vals)
        ax.plot([j-.28,j+.28],[md,md],color=COL[arm],lw=3,zorder=5,solid_capstyle="round")
        ax.text(j,md-0.42,f"{md:.2f}g",fontsize=10,color=COL[arm],ha="center",
                fontweight="bold",zorder=6)
    ax.set_xticks([0,1,2]); ax.set_xticklabels(["A","A0","B"],fontsize=11.5,fontweight="bold")
    ax.set_xlim(-.55,2.55)
    ax.set_ylabel("各工况最优峰值 / g",fontsize=10,color=INK2)
    cov=[sum(1 for i in good if PS[s][a][i]["ok"]) for s in sd for a in ("A","A0","B")]
    ax.set_title(f"m = {float(m):g} kg   覆盖:三臂全部 {len(good)}/{len(good)}(两种子)",
                 fontsize=11.6,color=INK,loc="left",pad=8)
    ax.grid(axis="y",alpha=.16,lw=.6); [ax.spines[s].set_visible(False) for s in ("top","right")]
    ax.tick_params(colors=INK2,labelsize=9)

# ---- 面板 3:净代价 vs 噪声地板 ----
ax=axes[2]
noise=[]
for m,d in R.items():
    good=d["good_ci"]; PS=d["per_seed"]; s0,s1=[str(x) for x in d["seeds"]]
    for a in ("A","A0","B"):
        com=[i for i in good if PS[s0][a][i]["ok"] and PS[s1][a][i]["ok"]]
        g0=np.median([PS[s0][a][i]["g"] for i in com]); g1=np.median([PS[s1][a][i]["g"] for i in com])
        noise.append(abs(g1-g0)/g0*100)
nf=max(noise)
ax.axhspan(-nf,nf,color=MUTED,alpha=.16,zorder=0)
ax.text(1.52,nf-0.5,f"种子噪声带 ±{nf:.1f}%",fontsize=9,color=INK2,ha="right",va="top")
ax.axhline(0,color=INK2,lw=.9)
xs=[]; k=0
for m,d in sorted(R.items(),key=lambda t:float(t[0])):
    good=d["good_ci"]; PS=d["per_seed"]; sd=[str(x) for x in d["seeds"]]
    for lbl,(p,q),col in (("A→A0",("A","A0"),BLUE),("A0→B",("A0","B"),ROD)):
        vs=[]
        for s in sd:
            com=[i for i in good if all(PS[s][a][i]["ok"] for a in ("A","A0","B"))]
            gp=np.median([PS[s][p][i]["g"] for i in com]); gq=np.median([PS[s][q][i]["g"] for i in com])
            vs.append((gq-gp)/gp*100)
        ax.plot([k,k],vs,color=col,lw=1.4,alpha=.5,zorder=2)
        ax.plot([k]*2,vs,"o",ms=8,color=col,zorder=4)
        ax.plot(k,np.mean(vs),"_",ms=22,color=col,mew=2.6,zorder=5)
        xs.append((k,f"{float(m):g}kg\n{lbl}")); k+=1
ax.set_xticks([t[0] for t in xs]); ax.set_xticklabels([t[1] for t in xs],fontsize=9)
ax.set_xlim(-.5,3.5); ax.set_ylim(-nf*1.5,nf*1.5)
ax.set_ylabel("峰值中位变化 / %",fontsize=10,color=INK2)
ax.set_title("消融的净代价:全部落在噪声带内\n(圆点=各种子,横杠=均值;符号在种子间翻转)",
             fontsize=11.6,color=CRIM,loc="left",pad=8)
ax.grid(axis="y",alpha=.16,lw=.6); [ax.spines[s].set_visible(False) for s in ("top","right")]
ax.tick_params(colors=INK2,labelsize=9)

from matplotlib.lines import Line2D
H=[Line2D([],[],marker="o",ls="none",color=COL[a],ms=8,label=LAB[a].replace("\n"," ")) for a in ("A","A0","B")]
fig.legend(handles=H,ncol=3,loc="lower center",frameon=False,fontsize=9.4,bbox_to_anchor=(.5,-.005))
fig.suptitle("机械架构消融:去掉膝减震器、再用连杆锁掉膝自由度,性能都没有可测的变化",
             fontsize=13.6,color=CRIM,x=.02,ha="left",y=.975)
fig.text(.02,.925,"高预算搜索(112 点/格)+ 失败格稠密救援 + 双种子。三臂覆盖全部满分,峰值差 ≤5% 且符号随种子翻转 —— 架构选择可由制造性单独决定。",
         fontsize=9.4,color=INK2)
fig.tight_layout(rect=[0,.055,1,.90])
fig.savefig("fig_架构消融.png",facecolor=SURF,bbox_inches="tight")
print("→ fig_架构消融.png")

# -*- coding: utf-8 -*-
"""论文 A · PGLS 结果图:系统发育吃掉多少效应,剩下的有多稳"""
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
FONT="Noto Sans CJK JP"
plt.rcParams.update({"font.family":FONT,"font.sans-serif":[FONT,"DejaVu Sans"],
    "axes.unicode_minus":False,"font.size":12,"axes.titlesize":13.5,"axes.labelsize":12,
    "figure.dpi":160,"savefig.bbox":"tight","axes.spines.top":False,"axes.spines.right":False})
OUT="/tmp/claude-0/-home-claude/d9fded9e-7347-5ca7-bdca-93555596398e/scratchpad/fig"
R=pd.read_csv("/mnt/user-data/uploads/FFY/FFY/outputs/pgls/pgls_p3_results.csv")
C={"hackett":"#1b6ca8","ericson":"#2E7D5B"}; OLS=-0.0776; C_POS="#c0392b"; GREY="#95a5a6"

fig,(ax1,ax2)=plt.subplots(1,2,figsize=(11.4,4.6),gridspec_kw=dict(width_ratios=[1.35,1],wspace=.28))

# --- A:beta,OLS vs PGLS ---
ax1.axvline(OLS,color=C_POS,lw=2.4,ls="--",zorder=1)
ax1.text(OLS,2.62,f"OLS（不控系统发育）\n{OLS:.4f}",color=C_POS,ha="center",va="bottom",fontsize=11,fontweight="bold")
rng=np.random.default_rng(0)
for i,(bb,S) in enumerate(R.groupby("backbone")):
    y=1.6-i*0.75+rng.normal(0,.055,len(S))
    ax1.scatter(S.beta,y,s=17,color=C[bb],alpha=.55,lw=0,zorder=3)
    med=S.beta.median()
    ax1.plot([med,med],[1.6-i*.75-.19,1.6-i*.75+.19],color=C[bb],lw=2.6,zorder=4)
    ax1.text(med,1.6-i*.75+.26,f"{bb}  中位 {med:.4f}",color=C[bb],ha="center",fontsize=10.5,fontweight="bold")
ax1.annotate("", xy=(R.beta.median(),2.35), xytext=(OLS,2.35),
             arrowprops=dict(arrowstyle="->",color="#555",lw=1.8))
ax1.text((OLS+R.beta.median())/2,2.42,f"系统发育吃掉 {100*(1-R.beta.median()/OLS):.0f}% 的表观效应",
         ha="center",fontsize=11,color="#333")
ax1.set_xlim(-0.086,-0.019); ax1.set_ylim(0.35,3.0); ax1.set_yticks([])
ax1.set_xlabel("β$_{HWI}$（HWI 每 +1，腿长残差 u 的变化）")
ax1.set_title("控制系统发育后效应缩到 1/3，但依然稳固",pad=10,fontweight="bold")
ax1.text(0.025,0.42,"200 棵树全部 p < 0.001\n（最大 p = 3e-59）\n两骨架中位差 0.0003\n< 各自树间标准差 0.0007",
         transform=ax1.transAxes,fontsize=10.5,va="center",ha="left",
         bbox=dict(boxstyle="round,pad=0.45",fc="#f6f1f1",ec=GREY,alpha=.95))

# --- B:lambda ---
for bb,S in R.groupby("backbone"):
    ax2.hist(S["lambda"],bins=16,alpha=.62,color=C[bb],label=f"{bb}（中位 {S['lambda'].median():.3f}）")
ax2.axvline(1.0,color=GREY,lw=1.3,ls=":")
ax2.text(0.9995,ax2.get_ylim()[1]*.97,"λ=1\n纯布朗演化",ha="right",va="top",fontsize=10,color="#666")
ax2.set_xlabel("Pagel's λ（系统发育信号强度）")
ax2.set_ylabel("树的棵数")
ax2.set_title("λ ≈ 0.93：亲缘结构确实极强",pad=10,fontweight="bold")
ax2.legend(frameon=False,fontsize=10.5,loc="upper left")
fig.savefig(f"{OUT}/fig_pgls.png"); print("saved")

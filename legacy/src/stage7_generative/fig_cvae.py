# -*- coding: utf-8 -*-
"""cVAE 架构图:训练时用编码器,推理时只用解码器。数值全部取自 cvae_r40.pt。"""
import numpy as np, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
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
BLU="#2a78d6"; ORA="#eb6834"; GRN="#1baf7a"; GREY="#b9b3a8"

def box(ax,x,y,w,h,txt,fc,ec="none",fs=9.5,tc="#ffffff",lsp=1.5,z=3):
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=0.012,rounding_size=0.03",
                                fc=fc,ec=ec,lw=1.4,zorder=z))
    ax.text(x+w/2,y+h/2,txt,ha="center",va="center",fontsize=fs,color=tc,
            zorder=z+1,linespacing=lsp)

def arrow(ax,p0,p1,c=INK2,lw=1.6,ls="-"):
    ax.add_patch(FancyArrowPatch(p0,p1,arrowstyle="-|>,head_width=3.2,head_length=6",
                                 color=c,lw=lw,ls=ls,zorder=2,
                                 shrinkA=1,shrinkB=1))

fig,(axA,axB)=plt.subplots(1,2,figsize=(14.4,6.2),dpi=200,
                           gridspec_kw=dict(wspace=.10))
fig.patch.set_facecolor(SURF)
for ax in (axA,axB):
    ax.set_facecolor(SURF); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")

# ============ (a) 训练 ============
axA.set_title("(a) 训练时：编码器 + 解码器",fontsize=13,color=INK,loc="left",
              pad=6,fontweight="bold")
box(axA,.015,.80,.245,.10,"设计 u  (7 维)\n已知的好设计",BLU,fs=9.5)
box(axA,.315,.785,.30,.115,"工况 c  (5 维)\nlog m · v0 · log k_c\ng_cap · s_max",ORA,fs=8.6,lsp=1.45)
arrow(axA,(.14,.795),(.25,.70)); arrow(axA,(.455,.780),(.38,.70))
box(axA,.16,.60,.31,.095,"拼接  [u , c] = 12 维",GREY,fs=9.5,tc="#3b372f")
arrow(axA,(.315,.595),(.315,.515))
box(axA,.12,.40,.39,.115,"编码器  Linear(12→64) · SiLU\nLinear(64→64) · SiLU",
    "#4a5568",fs=9)
arrow(axA,(.315,.395),(.315,.325))
box(axA,.13,.235,.16,.085,"μ  (3)","#7A5FA0",fs=10)
box(axA,.34,.235,.16,.085,"log σ²  (3)","#7A5FA0",fs=10)
axA.text(.515,.278,"clamp\n[−8,8]",fontsize=7.6,color=MUTED,va="center",linespacing=1.4)
arrow(axA,(.21,.230),(.28,.16)); arrow(axA,(.42,.230),(.35,.16))
box(axA,.19,.055,.25,.10,"重参数化采样\nz = μ + σ·ε   (3 维)",GRN,fs=9.5)
axA.text(.315,.02,"↓  与 c 拼接 → 解码器（见右图）",fontsize=9,color=INK2,ha="center")
axA.add_patch(FancyBboxPatch((.60,.055),.37,.62,
    boxstyle="round,pad=0.015,rounding_size=0.03",fc="#f4f1ec",ec=MUTED,lw=1.0,zorder=1))
axA.text(.785,.635,"损失函数",fontsize=11,color=CRIM,ha="center",fontweight="bold")
axA.text(.785,.545,"重构 MSE  +  β · KL",fontsize=11.5,color=INK,ha="center")
axA.text(.785,.44,
 "β = 0.02，前 30% epoch 线性预热\n"
 "Adam  lr = 1e-3 · batch 256 · 300 epoch\n\n"
 "预热的目的是防后验坍缩——\n但实测仍然坍缩了（见下方注）",
 fontsize=8.8,color=INK2,ha="center",va="top",linespacing=1.8)
axA.text(.785,.12,"编码器只在训练时用\n训练完就丢掉",fontsize=10,color=CRIM,
         ha="center",va="center",fontweight="bold",linespacing=1.6)

# ============ (b) 推理 ============
axB.set_title("(b) 推理时：只用解码器",fontsize=13,color=INK,loc="left",
              pad=6,fontweight="bold")
box(axB,.05,.82,.24,.095,"随机噪声\nz ~ N(0, I)  (3 维)",GRN,fs=9.2)
box(axB,.38,.82,.30,.095,"工况 c  (5 维)\n你想要什么样的着陆",ORA,fs=9.2)
arrow(axB,(.17,.815),(.27,.725)); arrow(axB,(.53,.815),(.43,.725))
box(axB,.20,.625,.30,.09,"拼接  [z , c] = 8 维",GREY,fs=9.5,tc="#3b372f")
arrow(axB,(.35,.620),(.35,.545))
box(axB,.145,.415,.41,.12,"解码器  Linear(8→64) · SiLU\nLinear(64→64) · SiLU\nLinear(64→7)",
    "#4a5568",fs=8.8)
arrow(axB,(.35,.410),(.35,.345))
box(axB,.19,.245,.32,.09,"Sigmoid  →  u ∈ [0,1]^7",BLU,fs=10)
arrow(axB,(.35,.240),(.35,.175),c=CRIM,lw=2.0)
box(axB,.10,.055,.50,.115,
    "生物先验展开  expand(u, m)\nL1 = 10^(a + b·log m + σ·u_L) ,  r2 , r3 , κ×3 , τ",
    CRIM,fs=9,lsp=1.7)
axB.text(.62,.115,"← 网络到这一步\n   才知道毫米和牛顿",fontsize=9,color=CRIM,
         va="center",linespacing=1.6,fontweight="bold")
axB.text(.755,.30,"网络全程只在\n[0,1]^7 的立方体里工作",fontsize=9.5,color=INK2,
         ha="center",va="center",linespacing=1.7)

fig.suptitle("cVAE 架构：两层 64 宽的全连接网络，隐变量 3 维，总参数 10,573",
             fontsize=14,color=CRIM,x=.035,ha="left",y=.965)
fig.text(.035,.042,
 "规模是刻意做小的：训练对最终 13,891 条、设计维度只有 7，再大就过拟合。"
 "条件 c 同时进编码器和解码器，是标准 cVAE 做法。",fontsize=9,color=INK2)
fig.text(.035,.014,
 "注：实测同一条件下 216 次采样在 7 维上的中位变异系数仅 0.21%——解码器基本忽略了 z，"
 "即后验坍缩。原因不在 β（β 小反而抑制坍缩），而在训练集只保留「可行且在 Pareto 前沿」的设计，"
 "每个条件下本就几乎只剩一个点。",fontsize=8.4,color=MUTED)
fig.tight_layout(rect=[0,.065,1,.94])
fig.savefig("/tmp/arch/fig_cvae_arch.png",facecolor=SURF,bbox_inches="tight")
print("→ /tmp/arch/fig_cvae_arch.png")

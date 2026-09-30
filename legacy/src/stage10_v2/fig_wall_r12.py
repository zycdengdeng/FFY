# -*- coding: utf-8 -*-
"""E20@r12：验收框(10 g × 24 mm)在硬地 2.0 m/s 处为空 —— 能量下界,不是模型缺陷。"""
import json, os, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib.font_manager as fm
_have={f.name for f in fm.fontManager.ttflist}
_cjk=next((f for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei","Droid Sans Fallback")
           if f in _have), None)
# 拉丁字形交给 DejaVu Sans（Droid Sans Fallback 缺拉丁与数字），中日韩交给 CJK 字体
plt.rcParams["font.sans-serif"]=([_cjk] if _cjk else [])+["DejaVu Sans"]
plt.rcParams["font.family"]="sans-serif"
plt.rcParams["axes.unicode_minus"]=False
U=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
A=json.load(open(f"{U}/outputs/v23_e20_r12/e20_bio.json")); MET=A["met"]
Z=np.load(f"{U}/outputs/v23_e20_r12/e20_bio_raw.npz",allow_pickle=True)
ig,ist=MET.index("peak_g"),MET.index("leg_stroke_mm")
GC,SM=10.0,24.0
CS=[("concrete1.2","硬地 · 1.2 m/s","#2E7D5B"),("turf2.0","草地 · 2.0 m/s","#d98032"),
    ("wetsand2.0","湿沙 · 2.0 m/s","#7a4bbf"),("concrete2.0","硬地 · 2.0 m/s","#c0392b")]
fig,ax=plt.subplots(1,4,figsize=(16.4,4.5),sharey=True,gridspec_kw=dict(wspace=.09))
for k,(c,lab,col) in enumerate(CS):
    B=ax[k]; m=Z[c+"__met"]; g=m[...,ig].ravel(); s=m[...,ist].ravel()
    f=np.isfinite(g)&np.isfinite(s); g,s=g[f],s[f]
    ok=(g<=GC)&(s<=SM)
    B.add_patch(Rectangle((0,0),SM,GC,fc="#e8f5ee" if ok.sum() else "#fdecea",
                          ec="#2E7D5B" if ok.sum() else "#c0392b",lw=2.2,zorder=1))
    B.scatter(s[~ok],g[~ok],s=5,c="#bbbbbb",alpha=.55,lw=0,zorder=2)
    if ok.sum(): B.scatter(s[ok],g[ok],s=6,c=col,alpha=.65,lw=0,zorder=3)
    B.axhline(GC,color="#c0392b",lw=1.6,ls="--"); B.axvline(SM,color="#c0392b",lw=1.6,ls="--")
    B.set_xlim(8,34); B.set_ylim(0,16); B.set_xlabel("落震行程 / mm",fontsize=12)
    B.set_title(f"{lab}\n可行 {100*ok.mean():.1f}%   (n={len(g):,})",fontsize=13.5,
                fontweight="bold",color=col if ok.sum() else "#c0392b")
    B.grid(alpha=.2)
    if k==0: B.set_ylabel("峰值过载 / g",fontsize=12.5)
    if c=="concrete2.0":
        B.text(.5,.955,f"验收框里一个候选都没有\n单独过 10 g 的 {int((g<=GC).sum()):,} 个"
                       f"  ·  单独过 24 mm 的 {int((s<=SM).sum()):,} 个",
               transform=B.transAxes,ha="center",va="top",fontsize=11.5,fontweight="bold",
               color="#c0392b",bbox=dict(boxstyle="round,pad=.38",fc="white",ec="#c0392b",lw=1.5))
        v,ss=2.0,0.024
        B.axhline(v*v/(2*ss)/9.81,color="#111",lw=1.8,ls=":")
        B.text(33.5,v*v/(2*ss)/9.81+.25,f"能量下界 v²/2s = {v*v/(2*ss)/9.81:.1f} g（平均值）",
               ha="right",fontsize=10.5,color="#111")
fig.suptitle("验收框 (10 g × 24 mm) 在「硬地 + 2.0 m/s」处为空 —— 这是能量守恒的边界,不是模型跑不出来",
             fontsize=15.5,fontweight="bold",y=1.045)
fig.text(.5,-.055,"在 24 mm 内把 2.0 m/s 停住,平均减速度就已经是 8.5 g;任何真实力型的峰值都比平均高 20–60%,"
                  "必然越过 10 g。地面越软,地面自己吸掉的行程越多,这一格才重新变得可行(草地 85.5%、湿沙 96.1%)。",
         ha="center",fontsize=12,color="#333")
out=f"{U}/outputs/v23_e20_r12/fig_验收框空格.png"
fig.savefig(out,dpi=165,bbox_inches="tight"); print("→",out)

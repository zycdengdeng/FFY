# -*- coding: utf-8 -*-
"""英文版：数据集构建 + 从生物先验到生成器的管线（两张图）。"""
import os, numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
plt.rcParams.update({"font.family":"DejaVu Sans","axes.unicode_minus":False,
                     "savefig.facecolor":"white","figure.facecolor":"white"})
U=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT=f"{U}/outputs/ppt_en"; os.makedirs(OUT,exist_ok=True)
BLU,ORA,GRN,PUR,GRY="#1b6ca8","#d98032","#2E7D5B","#7a4bbf","#6b6b6b"
D=pd.read_csv(f"{U}/data/birdtree/pgls_data_full.csv"); D=D[D.HWI.notna()&D.u.notna()]
n=len(D); nfam=D.Family.nunique(); nord=D.Order.nunique()

def bx(ax,x,y,w,h,t,b,col,fs=11.5,tfs=13):
    """标题贴框顶,正文在标题下方的剩余空间里垂直居中 —— 免得行数一多就压到框底。"""
    ax.add_patch(FancyBboxPatch((x,y),w,h,boxstyle="round,pad=.010",fc="white",ec=col,lw=2.2))
    ax.text(x+w/2,y+h-.045,t,ha="center",va="top",fontsize=tfs,fontweight="bold",color=col)
    ax.text(x+w/2,y+(h-.105)/2+.012,b,ha="center",va="center",fontsize=fs,color="#222",
            linespacing=1.55)
def ar(ax,p,q,col=GRY,lw=2.6):
    ax.add_patch(FancyArrowPatch(p,q,arrowstyle="-|>",mutation_scale=22,lw=lw,color=col,
                                 shrinkA=2,shrinkB=2))

# ---------- F7a: dataset ----------
fig,ax=plt.subplots(figsize=(14.6,7.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
bx(ax,.005,.655,.30,.30,"AVONET",
   "Tobias et al. 2022, Ecol Lett\n"
   "11,009 species x 11 morphological traits,\n"
   "measured on museum skins:\n"
   "tarsus, wing, Kipp's distance, mass",BLU,fs=11)
bx(ax,.005,.345,.30,.30,"EltonTraits 1.0",
   "Wilman et al. 2014, Ecology\n"
   "diet (10 categories) and foraging\n"
   "stratum (7 strata) for every\n"
   "extant bird species",ORA,fs=11)
bx(ax,.005,.035,.30,.30,"BirdTree",
   "Jetz et al. 2012, Nature\n"
   "9,993 tips, 2 backbones; we draw\n"
   "100 trees per backbone to carry\n"
   "topological uncertainty",GRN,fs=11)
bx(ax,.385,.25,.235,.50,"our merged table",
   f"name crosswalk\nBirdLife <-> BirdTree\n"
   f"{100*8705/8831:.1f}% matched\n\n"
   f"{n:,} species\n{nfam} families · {nord} orders\n\n"
   "flightless taxa removed",PUR,fs=11.5,tfs=13.5)
ax.text(.728,.955,"columns we actually use",fontsize=13.5,fontweight="bold",color="#222")
rows=[("tip","species name on the tree",GRY),
      ("$L_1$","tarsometatarsus length (mm)",BLU),
      ("$m$","body mass (g)",BLU),
      ("HWI","hand-wing index = Kipp's distance / wing length\n"
             "the standard proxy for flight efficiency",ORA),
      ("$u$","standardised leg-length residual\n"
             r"$u=[\log_{10}L_1-(a+b\log_{10}m)]/\sigma$",PUR),
      ("ForStrat","% of foraging done in water / ground / trees / air",GRN),
      ("Family, Order","clade membership, for the within-clade tests",GRY)]
y=.875
for k,v,c in rows:
    ax.text(.728,y,k,fontsize=12.5,fontweight="bold",color=c,va="top")
    ax.text(.878,y,v,fontsize=11.3,color="#222",va="top",linespacing=1.5)
    y-=.105 if "\n" not in v else .145
for yy in (.755,.755):
    pass
ar(ax,(.31,.80),(.382,.63)); ar(ax,(.31,.49),(.382,.50)); ar(ax,(.31,.18),(.382,.37))
ar(ax,(.625,.50),(.715,.50),col=PUR)
ax.set_title("The dataset we built  —  nothing new was measured; the value is in the join",
             fontsize=15.5,fontweight="bold",loc="left")
fig.savefig(f"{OUT}/en_F7_dataset.png",dpi=170,bbox_inches="tight"); plt.close(fig)

# ---------- F9: pipeline ----------
fig,ax=plt.subplots(figsize=(15.2,6.0)); ax.set_xlim(0,1); ax.set_ylim(0,1); ax.axis("off")
W,H,Y=.183,.50,.36
steps=[(0.0000,"8,705 birds","the leg-length law\n"
        r"$\log_{10}L_1 = a + b\,\log_{10}m$"+"\n"
        r"$b=0.365$,  $\sigma=0.081$ dex"+"\n\n"
        "phylogenetically\ncorrected, 200 trees",GRN),
       (0.2085,"design prior","for a target mass:\n"
        "a 1-sigma corridor of\nleg lengths, not one\nnumber\n\n"
        "the search space is\nnarrowed, not fixed",PUR),
       (0.411,"generator","conditional model\n"
        r"$z$ (5-dim) $\rightarrow$ $x$ (9-dim)"+"\n\n"
        "trained on Exudyn\ndrop simulations\nof A0 legs",BLU),
       (0.6135,"feasibility","10 g  /  24 mm\nslenderness\nleg-mass budget\n\n"
        "every candidate is\nsimulated, not scored\nby a surrogate",ORA),
       (0.816,"what emerges","the generator's own\n"
        r"scaling  $b_{eff}=0.238$"+"\n\n"
        "flatter than biology:\nthe machine affords\nstiffer joints than\na bird can","#c0392b")]
for x,t,bb,c in steps:
    bx(ax,x,Y,W,H,t,bb,c,fs=10.5,tfs=13)
for x,_,_,_ in steps[:-1]:
    ar(ax,(x+W,Y+H/2),(x+W+.019,Y+H/2),lw=2.2)
ax.text(.006,.24,"The biology does not design the leg.  It tells the generator where to look —\n"
                 "and the physics then tells us where biology's answer stops being the right one for a 30 kg machine.",
        fontsize=12.5,color="#333",linespacing=1.7,
        bbox=dict(boxstyle="round,pad=.5",fc="#f6f6f6",ec="#bbb"))
ax.set_title("From a biological law to a machine  —  where the prior actually enters",
             fontsize=15.5,fontweight="bold",loc="left")
fig.savefig(f"{OUT}/en_F9_pipeline.png",dpi=170,bbox_inches="tight"); plt.close(fig)
print("→",OUT,"| n=",n,"fam=",nfam,"ord=",nord)

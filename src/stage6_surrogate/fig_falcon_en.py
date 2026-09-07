# -*- coding: utf-8 -*-
"""隼科一页的英文版：复用中文脚本的布局，只换字符串与输出目录。"""
import os
HERE=os.path.dirname(os.path.abspath(__file__))
src=open(os.path.join(HERE,"fig_falcon.py"),encoding="utf-8").read()
M=[('OUT=f"{U}/outputs/ppt_cn"','OUT=f"{U}/outputs/ppt_en"'),
   ('_cjk=next((f for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei","Droid Sans Fallback")\n           if f in _have), None)',
    '_cjk=None'),
   ('/cn_F_falcon.png','/en_F_falcon.png'),
   ('(P,"烟隼","Falco concolor",BLU,0.24,"开阔天空的长距离迁徙猎手\\n高速平飞追捕 —— 腿是纯死重")',
    '(P,"Sooty Falcon","Falco concolor",BLU,0.24,"long-distance migrant, hunts on the wing\\nin open sky - the leg is dead weight")'),
   ('(Q,"林隼","Micrastur plumbeus",ORA,0.74,"南美雨林树冠层穿梭\\n短距爆发 + 枝上站立抓握 —— 腿有用")',
    '(Q,"Plumbeous Forest-falcon","Micrastur plumbeus",ORA,0.74,"threads the Amazon canopy: short bursts\\nplus perching and grasping - the leg earns its keep")'),
   ('"同一个科里的两只隼"','"Two falcons from the same family"'),
   ('f"飞行效率 HWI = {r.HWI:.0f}\\n体重 = {r.m_g:.0f} g"','f"HWI = {r.HWI:.0f}\\nbody mass = {r.m_g:.0f} g"'),
   ('"上面两根按真实比例绘制的，是跗跖骨（腿的主骨）"','"the two bars are tarsometatarsi, drawn to scale"'),
   ('f"林隼比烟隼还轻 {P.m_g-Q.m_g:.0f} 克，腿却长了 {100*(Q.L1/P.L1-1):.0f}%"',
    'f"{P.m_g-Q.m_g:.0f} g lighter, yet {100*(Q.L1/P.L1-1):.0f}% more leg"'),
   ('(P,"烟隼",BLU,(0.985,0.55),"right","center")','(P,"Sooty Falcon",BLU,(0.985,0.55),"right","center")'),
   ('(Q,"林隼",ORA,(0.020,0.965),"left","top")','(Q,"Forest-falcon",ORA,(0.020,0.965),"left","top")'),
   ('f"{name}   HWI {r.HWI:.0f} · 腿 {r.L1:.0f} mm"','f"{name}   HWI {r.HWI:.0f} · leg {r.L1:.0f} mm"'),
   ('"飞行效率  HWI"','"flight efficiency   HWI"'),
   ('"腿长残差（已扣掉体重的影响）"','"leg-length residual (body mass removed)"'),
   ('f"整个隼科 {len(F)} 种，趋势一致"','f"All {len(F)} species of Falconidae show the same trend"'),
   ('f"相关系数  r = {r_all:+.2f}   (n = {len(F)})"','f"r = {r_all:+.2f}   (n = {len(F)})"'),
   ('f"按 HWI = {cut:.0f} 把科内切两半：\\n"\n                 f"飞得好的一半中位残差 {np.median(hi.e):+.2f}，另一半 {np.median(lo.e):+.2f}\\n"\n                 f"差值 Δ = {d_half:+.2f}  —— 这就是族内比较图里隼科那一根柱子"',
    'f"Split the family at HWI = {cut:.0f}:\\n"\n                 f"median residual {np.median(hi.e):+.2f} for the better fliers, {np.median(lo.e):+.2f} for the rest\\n"\n                 f"difference Δ = {d_half:+.2f} - this is the Falconidae bar in the within-family figure"'),
   ('"为什么说「同科之内也成立」—— 隼科的一个具体例子"',
    '"What \\"it holds inside a family\\" looks like - one concrete case in Falconidae"'),
   ('_lat = latin.replace(" ", r"\\ ")','_lat = latin.replace(" ", r"\\ ")')]
for a,b in M:
    assert a in src, "未匹配: "+a[:55]
    src=src.replace(a,b)
exec(compile(src,"fig_falcon_en","exec"),{"__name__":"__main__","__file__":__file__})

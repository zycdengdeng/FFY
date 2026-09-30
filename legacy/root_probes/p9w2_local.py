# -*- coding: utf-8 -*-
"""P9w-2 · 轮足定标（理想止回分段近似）：低 Fr=locked（棘轮咬合），高 Fr=brake（棘轮自由）"""
import sys,json,itertools,collections; sys.path.insert(0,'.')
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import wheel_probe as W, physics_v2 as P
BASE={**P.SCEN_BIRD_X,"hip_damp_unified":True,"foot_mode":"bearing"}
DESIGNS=[[100.,1.7,1.05,3.,4.,10.,.025,140.,135.,45.],
         [120.,1.8,1.06,3.,4.,10.,.025,140.,135.,50.],
         [140.,1.9,1.10,3.,4.,10.,.025,140.,135.,50.],
         [110.,1.7,1.00,2.5,4.,9.,.03,135.,130.,42.],
         [130.,1.8,1.10,3.5,5.,11.,.02,145.,140.,52.],
         [120.,1.75,1.05,3.,4.,10.,.025,140.,135.,48.]]
MASS=[5.,8.,12.]
def Lref(m): return (10**(0.479+0.391*np.log10(m*1000))/1000)*3.85
def job(a):
    di,m,fr,mode,tau=a
    vx=fr*np.sqrt(9.81*Lref(m))
    b=dict(BASE); b["T"]=1.2 if mode=="brake" else b.get("T",0.30)
    r=W.eval_wheel(DESIGNS[di],m,1.2,1e6,base=b,v_x=vx,planar=True,mode=mode,
                   pitch_free=False,wheel=dict(tau_max=tau),guards=True)
    r.update(di=di,m=m,fr=fr,tau=tau,vx=round(vx,2)); return r
jobs=[]
for di,m in itertools.product(range(6),MASS):
    for fr in (0.,0.5,1.0): jobs.append((di,m,fr,"locked",0))
    for fr in (2.0,4.0):
        for tau in (8.,16.): jobs.append((di,m,fr,"brake",tau))
with ProcessPoolExecutor(max_workers=4) as ex:
    rows=list(ex.map(job,jobs))
json.dump(rows,open("p9w2.json","w"))
tab=collections.defaultdict(lambda:collections.Counter())
met=collections.defaultdict(list)
for r in rows:
    k=(r["fr"],r["mode"],r["tau"])
    if r.get("fail"): tab[k][r["fail"]]+=1
    else:
        tab[k]["ok"]+=1
        met[k].append((r["peak_g"],abs(r["foot_dx_signed_mm"]),r["leg_stroke_mm"]))
print(f"{'Fr':>4}{'态':>8}{'tau':>5}{'n':>4}{'ok':>4}{'失败构成':<28}{'peak_g中位':>10}{'滚/滑距中位mm':>13}")
for k in sorted(tab):
    c=tab[k]; n=sum(c.values()); ok=c.get("ok",0)
    fails=" ".join(f"{a}:{b}" for a,b in c.items() if a!="ok") or "-"
    if met[k]:
        pg=float(np.median([x[0] for x in met[k]])); dx=float(np.median([x[1] for x in met[k]]))
        print(f"{k[0]:>4}{k[1]:>8}{k[2]:>5.0f}{n:>4}{ok:>4} {fails:<28}{pg:>10.2f}{dx:>13.1f}")
    else:
        print(f"{k[0]:>4}{k[1]:>8}{k[2]:>5.0f}{n:>4}{ok:>4} {fails:<28}{'—':>10}{'—':>13}")

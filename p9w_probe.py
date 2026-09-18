# -*- coding: utf-8 -*-
"""P9w · 轮足定标探针（A100，锁俯仰）。约 15 min @ 128 workers。
只回答单点平面模型能诚实回答的三件事：
  1. 轮足版 Fr 上界重定标（裸足是 0；轮足应 >0）
  2. 刹车力矩 tau_max 多大才能在 Fr=0 锁住轮（不滚开）
  3. 单向轴承方向确认（Fr=0 vs Fr>0 滚动方向是否相反）
翻倒（tip-over）不在此——需前后轮距模型，见备忘 v3。
用法： OMP_NUM_THREADS=1 python p9w_probe.py --workers 128 --out outputs/v26_p9w
"""
import sys, os, json, argparse, itertools
sys.path.insert(0,"src/stage10_v2"); sys.path.insert(0,".")
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import wheel_probe as W, physics_v2 as P

BASE={**P.SCEN_BIRD_X,"hip_damp_unified":True,"foot_mode":"bearing"}
# 6 个代表设计（跨质量），可行盒中点附近
DESIGNS=[[100.,1.7,1.05,3.,4.,10.,.025,140.,135.,45.],
         [120.,1.8,1.06,3.,4.,10.,.025,140.,135.,50.],
         [140.,1.9,1.10,3.,4.,10.,.025,140.,135.,50.],
         [110.,1.7,1.00,2.5,4.,9.,.03,135.,130.,42.],
         [130.,1.8,1.10,3.5,5.,11.,.02,145.,140.,52.],
         [120.,1.75,1.05,3.,4.,10.,.025,140.,135.,48.]]
MASS=[5.,8.,12.]; FR=[0.,0.5,1.,2.,4.]; TAU=[4.,8.,16.]
def Lref(m): return P.SCEN_BIRD_X and (10**(0.479+0.391*np.log10(m*1000))/1000)*3.85
def _job(a):
    di,m,fr,tau=a; x=DESIGNS[di]
    vx=fr*np.sqrt(9.81*Lref(m))
    r=W.eval_wheel(x,m,1.2,1e6,base=BASE,v_x=vx,planar=True,mode="brake",
                   pitch_free=False,wheel=dict(tau_max=tau),mat="al7075")
    r.update(di=di,m=m,fr=fr,tau=tau,vx=vx); return r
def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--workers",type=int,default=8)
    ap.add_argument("--out",default="outputs/v26_p9w"); a=ap.parse_args()
    os.makedirs(a.out,exist_ok=True)
    jobs=list(itertools.product(range(len(DESIGNS)),MASS,FR,TAU))
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        rows=[r for r in ex.map(_job,jobs) if not r.get("fail")]
    json.dump(rows,open(os.path.join(a.out,"p9w.json"),"w"),ensure_ascii=False)
    # 汇总：每个 (Fr,tau) 的锁轮率（轮 ω<5 视为锁住）与滚动方向
    print(f"{'Fr':>5}{'tau':>6}{'锁轮率%':>9}{'足端Δx中位mm':>14}{'peak_g中位':>11}")
    for fr in FR:
        for tau in TAU:
            sub=[r for r in rows if r["fr"]==fr and r["tau"]==tau]
            if not sub: continue
            lock=np.mean([r["wheel_spin_max"]<5 for r in sub])*100
            dx=np.median([r["foot_dx_signed_mm"] for r in sub])
            pg=np.median([r["peak_g"] for r in sub])
            print(f"{fr:>5.1f}{tau:>6.0f}{lock:>9.0f}{dx:>14.1f}{pg:>11.2f}")
    print(f"\n{len(rows)} 有效 / {len(jobs)} 总 → {a.out}/p9w.json")
if __name__=="__main__": main()

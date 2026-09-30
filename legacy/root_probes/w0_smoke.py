# -*- coding: utf-8 -*-
"""W0 冒烟 · 轮足三态对照（本地/云端可跑，单核几秒）。
结论见《足端方案备忘_轮足》v2。用法： python w0_smoke.py"""
import sys, os
sys.path.insert(0,".")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src", "stage10_v2"))
import numpy as np, wheel_probe as W, physics_v2 as P

X=[120.,1.8,1.06,3.0,4.0,10.0,0.025,140.,135.,50.]   # 代表设计（10kg 档）
BASE={**P.SCEN_BIRD_X,"hip_damp_unified":True,"foot_mode":"bearing"}

def block(vx,label):
    print(f"\n{label} · 10kg 硬地 · al7075")
    print(f"{'态':<8}{'peak_g':>8}{'a_res':>8}{'腿行程mm':>10}{'轮ω':>7}{'足端Δx签名mm':>15}")
    for mode in ["rigid","locked0","locked","free","brake"]:
        kw={}
        if mode=="locked0":   # 受控对照：轮径=裸足 r_foot、轮质量≈0，应严格≈rigid(npass=1)
            kw=dict(mode="locked",wheel=dict(r_w=0.012,m_w=0.001))
        else:
            kw=dict(mode=mode)
        r=W.eval_wheel(X,10.,1.2,1e6,base=BASE,v_x=vx,planar=True,
                       pitch_free=False,mat="al7075",**kw)
        r["mode"]=mode
        if r.get("fail"): print(f"{mode:<8} FAIL[{r['fail']}]"); continue
        print(f"{mode:<8}{r['peak_g']:>8.2f}{r['a_res_g']:>8.2f}"
              f"{r['leg_stroke_mm']:>10.2f}{r['wheel_spin_max']:>7.1f}{r['foot_dx_signed_mm']:>15.2f}")

if __name__=="__main__":
    block(0.0,"Fr=0（垂直着陆）")
    block(1.04,"Fr≈0.5（v_x=1.04，前向着陆）")
    print("\n验收：① locked≈rigid（轮体不破坏接触物理）")
    print("      ② Fr=0 足端滚 −x，Fr>0 滚 +x → 单向轴承许+x锁-x，两种着陆双赢")
    print("      ③ 放开俯仰会翻倒 → 翻倒判据需前后轮距，单点平面模型装不下（v3）")

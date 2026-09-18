# -*- coding: utf-8 -*-
"""轮足探针 · v3 预研（不进主工厂，不改 v2.5 物理）

在 v2.5 bird 拓扑上把「刚性足球」换成「能独立自转的轮」：
  · 轮体 = 一个小圆盘，通过绕 y 的旋转副接在跖骨足端；接触球移到轮上。
  · 刹车力矩律（反离心）：τ_b(ω) = τ_max · clip(1 − |ω|/ω_th, 0, 1) · (−sign ω)
        触地瞬间 ω≈0 → 刹车满咬合 ≈ 橡胶固定足；高速滚 → 刹车松 → 自由滚。
  · 单向轴承 oneway ∈ {0, +1, −1}：只许一个方向自转，反向加大刚度锁住。
  · 机体俯仰自由度 pitch_free：轮足版必须放开——刹车减速度与翻倒直接耦合，
        锁着俯仰测出的刹车结论是假的（这也是 E29 翻倒三相图并入此处的原因）。

W0 冒烟三态（验证新物理没跑偏 + 定单向轴承方向）：
    rigid  : 基线，足球焊死在跖骨（= exu_eval_v2）
    locked : 轮存在但刹车无穷大（τ_max 极大）→ 应 ≈ rigid（校准）
    free   : 轮自由滚（τ_max=0）→ 水平力被滚掉，滑移/俯仰应显著变大
"""
from __future__ import annotations
import sys, numpy as np
sys.path.insert(0, ".")
import physics_v2 as P
import exudyn as exu
from exudyn.utilities import *          # noqa
from exudyn.utilities import (InertiaCuboid, InertiaSphere, NodeGenericODE1,
                              ObjectGenericODE1, SensorBody)

rotY = P.rotY


def _wheel_inertia(m_w, r_w):
    # 圆盘绕自转轴 y： I = 1/2 m r²；另两轴 1/4 m r²。用 cuboid 近似成薄盘。
    return InertiaCuboid(density=m_w / (np.pi * r_w**2 * (0.3*r_w)),
                         sideLengths=[2*r_w, 0.3*r_w, 2*r_w])


def eval_wheel(x7, m, v0, kc, zeta_c=0.15, base=None, v_x=0.0, planar=True,
               mode="rigid", wheel=None, pitch_free=None, mat="al7075", guards=True):
    """mode ∈ {rigid, locked, free, brake}. wheel=dict(r_w, m_w, tau_max, omega_th, oneway)."""
    if mode == "rigid":                       # 直接走验证过的 v2.5，加机体俯仰可选
        r = P.eval_v2(x7, m, v0, kc=kc, zeta_c=zeta_c, base=base, v_x=v_x,
                      planar=planar, mat=mat, npass=1, keep_history=False)
        return dict(mode="rigid", peak_g=r["peak_a"]/9.81, a_res_g=r["a_res"]/9.81,
                    leg_stroke_mm=r["leg_stroke"]*1e3, sink_mm=r.get("sink",0.)*1e3,
                    body_pitch_deg=0.0, wheel_spin_max=0.0,
                    foot_dx_signed_mm=r.get("foot_dx_mm",0.0),
                    x_drift_mm=r.get("x_drift",0.0)*1e3, r_w_mm=r.get("r_foot",0.0)*1e3,
                    mu=r.get("mu_used",0.0))

    w = dict(r_w=0.030, m_w=0.04, tau_max=8.0, omega_th=20.0, oneway=0)
    if wheel: w.update(wheel)
    if pitch_free is None:
        pitch_free = (mode != "rigid")        # 轮足默认放开俯仰

    # ---- 场景尺寸化（复用 v2.5 的 size_x_v2）----
    scen = {**(base or P.SCEN_BIRD_X), "m": m, "v0": v0, "kc": kc,
            "zeta_c": zeta_c, "v_x": v_x, "planar": planar,
            "foot_mode": base.get("foot_mode", "bearing") if base else "bearing",
            "hip_damp_unified": True, "T": (base or {}).get("T", 0.30),
            "h": (base or {}).get("h", 2e-4), "g": 9.81}
    # 两遍定点：先猜杆件质量→定尺→回代（和 eval_v2 一致，简化为单遍猜测即可，探针用）
    s = P.size_x_v2(scen, x7)
    l1, r2, r3 = float(x7[0])/1000., float(x7[1]), float(x7[2])
    l1m = l1; l2 = r2*l1m; l3 = r3*l1m
    g = s["g"]; vx = float(s.get("v_x", 0.0))
    m1, m2, m3 = s["seg_mass"]
    a1 = s["q1_0"]; a2 = a1 + (np.pi - s["thetaA"]); a3 = a2 - (np.pi - s["thetaK"])
    dvec = lambda a: np.array([np.cos(a), 0., np.sin(a)])
    r_w = float(w["r_w"])
    Fp = np.array([0., 0., s["gap0"] + r_w])        # 足端（轮心）离地 = 轮半径
    A = Fp + l1m*dvec(a1); K = A + l2*dvec(a2); H = K + l3*dvec(a3)

    SC = exu.SystemContainer(); mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])

    def rod(name, P0, P1, ang, mass):
        L = np.linalg.norm(P1-P0); com = 0.5*(P0+P1)
        inertia = InertiaCuboid(density=mass/(L*0.02*0.02), sideLengths=[L,0.02,0.02])
        return mbs.CreateRigidBody(name=name, referencePosition=list(com),
                                   referenceRotationMatrix=rotY(ang),
                                   initialVelocity=[vx,0,-v0], inertia=inertia,
                                   gravity=[0,0,-g])
    tarso = rod("tarso", Fp, A, a1, m1)
    tibio = rod("tibio", A, K, a2, m2)
    femur = rod("femur", K, H, a3, m3)
    body = mbs.CreateRigidBody(name="payload", referencePosition=list(H),
                               initialVelocity=[vx,0,-v0],
                               inertia=InertiaSphere(mass=m, radius=0.12),
                               gravity=[0,0,-g])
    # 机体约束：x,z 自由；y 锁；转动 —— pitch_free 时放开绕 y
    ca = [0,1,0,1,(0 if pitch_free else 1),1]
    mbs.CreateGenericJoint(bodyNumbers=[ground, body], position=list(H),
                           constrainedAxes=ca)

    # ---- 关节扭簧阻尼（线性，探针不加 Zener）----
    for (jn, b0, b1, Pj, kj, cj) in [("hip",body,femur,H,s["k_hip"],s["c_hip"]),
                                     ("knee",femur,tibio,K,s["k_knee"],s["c_knee"]),
                                     ("ankle",tibio,tarso,A,s["k_ankle"],s["c_ankle"])]:
        mbs.CreateRevoluteJoint(bodyNumbers=[b0,b1], position=list(Pj), axis=[0,1,0])
        mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0,b1], position=list(Pj),
                                        axis=[0,1,0], stiffness=kj, damping=cj)

    # ---- 轮体 + 旋转副 + 刹车力矩 ----
    wheel_b = mbs.CreateRigidBody(name="wheel", referencePosition=list(Fp),
                                  referenceRotationMatrix=rotY(a1),
                                  initialVelocity=[vx,0,-v0],
                                  inertia=_wheel_inertia(w["m_w"], r_w),
                                  gravity=[0,0,-g])
    tau_max = float(w["tau_max"]); om_th = float(w["omega_th"]); oneway = int(w["oneway"])
    if mode == "locked":
        # 轮焊死在跖骨 = 刚性橡胶足（校准基准：应≈rigid，只多轮子那点质量/半径差）
        mbs.CreateGenericJoint(bodyNumbers=[tarso, wheel_b], position=list(Fp),
                               constrainedAxes=[1,1,1,1,1,1])
    else:
        mbs.CreateRevoluteJoint(bodyNumbers=[tarso, wheel_b], position=list(Fp), axis=[0,1,0])
    if mode == "free":
        tau_max = 0.0
    # brake / locked 共用同一个反离心律 + 单向锁

    LOCK_C = 80.0     # locked: 纯大阻尼锁住自转（N·m·s/rad），数值友好
    def _brake(mbs_, t_, item_, rot, rot_t, k_, d_, off_,
               _tm=tau_max, _oth=om_th, _ow=oneway, _mode=mode, _lc=LOCK_C):
        om = rot_t
        if _mode == "locked":
            return -_lc * om
        frac = max(0.0, 1.0 - abs(om)/max(_oth,1e-6))   # 反离心：越快越松
        tau = -np.sign(om) * _tm * frac
        if _ow != 0 and np.sign(om) == -np.sign(_ow) and om != 0.0:
            tau += -_ow * 200.0 * abs(rot)              # 单向：反向刚性锁
        return tau
    if mode != "locked":
        con = mbs.CreateTorsionalSpringDamper(bodyNumbers=[tarso, wheel_b],
                                              position=list(Fp), axis=[0,1,0],
                                              stiffness=0.0, damping=0.0)
        mbs.SetObjectParameter(con, "springTorqueUserFunction", _brake)

    # ---- 接触：球在轮心，随轮滚 ----
    quad = [[-3,-3,0],[3,-3,0],[3,3,0],[-3,3,0]]
    mbs.CreateSphereQuadContact(bodyNumbers=[wheel_b, ground],
                                localPosition0=[0,0,0], radiusSphere=r_w,
                                contactStiffness=s["kc"], contactDamping=s["cc"],
                                dynamicFriction=s["mu"], frictionProportionalZone=1e-3)

    OV = exu.OutputVariableType
    sAcc = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OV.Acceleration))
    sPos = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OV.Position))
    sBodRot = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OV.Rotation))
    sWhlRot = mbs.AddSensor(SensorBody(bodyNumber=wheel_b, storeInternal=True, outputVariableType=OV.Rotation))
    sWhlAV  = mbs.AddSensor(SensorBody(bodyNumber=wheel_b, storeInternal=True, outputVariableType=OV.AngularVelocity))
    sTarAV  = mbs.AddSensor(SensorBody(bodyNumber=tarso, storeInternal=True, outputVariableType=OV.AngularVelocity))
    sFoot = mbs.AddSensor(SensorBody(bodyNumber=wheel_b, storeInternal=True,
                                     localPosition=[0,0,0], outputVariableType=OV.Position))
    sAccR = [mbs.AddSensor(SensorBody(bodyNumber=b, storeInternal=True, outputVariableType=OV.Acceleration))
             for b in (femur, tibio, tarso, wheel_b)]

    mbs.Assemble()
    ss = exu.SimulationSettings()
    ss.timeIntegration.endTime = s["T"]
    ss.timeIntegration.numberOfSteps = int(s["T"]/s["h"])
    ss.timeIntegration.generalizedAlpha.spectralRadius = 0.7
    ss.timeIntegration.verboseMode = 0
    ss.solutionSettings.writeSolutionToFile = False
    ss.solutionSettings.sensorsWritePeriod = s["h"]
    try:
        from hf_exudyn import silence_solver
        with silence_solver():
            mbs.SolveDynamic(ss)
    except Exception as e:
        return dict(fail="solver", mode=mode, err=str(e)[:80])

    acc = mbs.GetSensorStoredData(sAcc); pos = mbs.GetSensorStoredData(sPos)
    t = acc[:,0]; ax = acc[:,1]; az = acc[:,3]; z = pos[:,3]; x = pos[:,1]
    if not np.all(np.isfinite(az)):
        return dict(fail="nonfinite", mode=mode)
    a_res = float(np.max(np.hypot(ax, az)))
    peak_g = float(np.max(np.abs(az)))/9.81
    stroke = float(z[0]-np.min(z))
    zf = mbs.GetSensorStoredData(sFoot)[:,3]
    sink = float(max(0.0, r_w - np.min(zf)))
    leg_stroke = float(max(0.0, stroke - sink))
    if guards and leg_stroke > 0.6*(l1m+l2+l3):
        return dict(fail="collapse", mode=mode)
    if guards and sink > 0.9*r_w:
        return dict(fail="deep_sink", mode=mode)
    body_pitch = float(np.degrees(np.max(np.abs(mbs.GetSensorStoredData(sBodRot)[:,2]))))
    om_w = mbs.GetSensorStoredData(sWhlAV)[:,2]
    om_t = mbs.GetSensorStoredData(sTarAV)[:,2]
    n = min(len(om_w), len(om_t))
    om = om_w[:n] - om_t[:n]                 # 相对跖骨的自转（修：原为绝对角速度）
    spin_max = float(np.max(np.abs(om)))
    # 足端接触点水平位移（含滚动，因为轮就是靠滚）
    fx = mbs.GetSensorStoredData(sFoot)[:,1]
    # 接触窗口内净水平位移（触地→最低点），带符号——用来定单向轴承方向
    Fz = float(m)*az.copy()
    for j,sj in enumerate(sAccR):
        a=mbs.GetSensorStoredData(sj); n=min(len(a),len(az)); Fz[:n]+=[m1,m2,m3,w["m_w"]][j]*a[:n,3]
    Mtot = m+m1+m2+m3+w["m_w"]; Fz = Fz + Mtot*g
    on = Fz > 0.2*Mtot*g
    i0 = int(np.argmax(on)) if on.any() else 0
    imin = i0 + int(np.argmin(z[i0:])) if i0 < len(z)-1 else i0
    foot_dx_signed = float(fx[imin]-fx[i0]) if imin < len(fx) else 0.0
    return dict(mode=mode, peak_g=peak_g, a_res_g=a_res/9.81,
                leg_stroke_mm=leg_stroke*1e3, sink_mm=sink*1e3,
                body_pitch_deg=body_pitch, wheel_spin_max=spin_max,
                foot_dx_signed_mm=foot_dx_signed*1e3, x_drift_mm=float(x[-1]-x[0])*1e3,
                r_w_mm=r_w*1e3, mu=float(s["mu"]))

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
    oLock = None
    if mode == "locked":
        # 轮焊死在跖骨 = 刚性橡胶足（校准基准：应≈rigid，只多轮子那点质量/半径差）
        mbs.CreateGenericJoint(bodyNumbers=[tarso, wheel_b], position=list(Fp),
                               constrainedAxes=[1,1,1,1,1,1])
    else:
        mbs.CreateRevoluteJoint(bodyNumbers=[tarso, wheel_b], position=list(Fp), axis=[0,1,0])
        if mode == "bc":
            # B+C 复合的棘轮：休眠扭簧。preStep 检测到反向相对转速（λ=+1 标定的啮合方向）
            # 时，读该扭簧自己口径下的连续转角设为 offset，再上刚度/阻尼 ——
            # 咬合瞬间零力零位移（齿咬在当前齿位），彻底避开"锁回装配参考角"的弹回灾难。
            oLock = mbs.CreateTorsionalSpringDamper(bodyNumbers=[tarso, wheel_b],
                                                    position=list(Fp), axis=[0,1,0],
                                                    stiffness=0.0, damping=0.0)
    if mode == "free":
        tau_max = 0.0
    # brake / locked 共用同一个反离心律 + 单向锁

    LOCK_C = 80.0     # locked: 纯大阻尼锁住自转（N·m·s/rad），数值友好
    _rst = {"hi": 0.0, "lo": 0.0}   # 棘轮锚：已达到的最大前向角（每次仿真独立）
    K_RATCH, C_RATCH = 40.0, 0.1    # 止回刚度 N·m/rad、止回阻尼
    def _brake(mbs_, t_, item_, rot, rot_t, k_, d_, off_,
               _tm=tau_max, _oth=om_th, _ow=oneway, _mode=mode, _lc=LOCK_C, _st=_rst):
        om = rot_t
        if _mode == "locked":
            return -_lc * om
        # 反离心刹车（对称作用）+ 平滑化 sign，避免 ω≈0 处颤振
        frac = max(0.0, 1.0 - abs(om)/max(_oth,1e-6))
        tau = -np.tanh(om/0.5) * _tm * frac
        # 单向轴承：反向速度阻尼式（平滑），允许微小蠕滑，数值友好。
        # _ow=+1 允许 ω>0 自由；ω<0 被强阻尼压住（半 tanh 门控，无不连续）。
        if _ow != 0:
            gate = 0.5*(1.0 - np.tanh((_ow*om)/0.2))   # 反向≈1，正向≈0
            tau += -20.0 * gate * om
        return tau
    if mode != "locked":
        con = mbs.CreateTorsionalSpringDamper(bodyNumbers=[tarso, wheel_b],
                                              position=list(Fp), axis=[0,1,0],
                                              stiffness=0.0, damping=0.0)
        mbs.SetObjectParameter(con, "springTorqueUserFunction", _brake)

    # ---- 接触：球在轮心，随轮滚 ----
    quad = [[-30,-30,0],[30,-30,0],[30,30,0],[-30,30,0]]   # 轮足会滚远，地面要铺够
    mbs.CreateSphereQuadContact(bodyNumbers=[wheel_b, ground],
                                localPosition0=[0,0,0], radiusSphere=r_w,
                                quadPoints=quad,
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

    if mode == "bc":
        _latch = {"on": False, "t_on": None}
        _sW, _sT = sWhlAV, sTarAV
        K_LATCH, C_LATCH = 40.0, 0.05
        def _prestep(mbs_, t_):
            if not _latch["on"] and t_ > 0.004:
                rel = mbs_.GetSensorValues(_sW)[1] - mbs_.GetSensorValues(_sT)[1]
                if rel < -0.5:                       # 反向（棘轮啮合方向）
                    phi = float(np.atleast_1d(mbs_.GetObjectOutput(
                        oLock, exu.OutputVariableType.Rotation))[0])
                    mbs_.SetObjectParameter(oLock, "offset", phi)
                    mbs_.SetObjectParameter(oLock, "stiffness", K_LATCH)
                    mbs_.SetObjectParameter(oLock, "damping", C_LATCH)
                    _latch["on"] = True; _latch["t_on"] = t_
            return True
        mbs.SetPreStepUserFunction(_prestep)
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
    spin_signed = float(om[np.argmax(np.abs(om))])
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
                body_pitch_deg=body_pitch, wheel_spin_max=spin_max, wheel_spin_signed=spin_signed,
                foot_dx_signed_mm=foot_dx_signed*1e3, x_drift_mm=float(x[-1]-x[0])*1e3,
                latch_t=(_latch["t_on"] if mode=="bc" else None),
                r_w_mm=r_w*1e3, mu=float(s["mu"]))


# ============================================================================
# M2 · 双腿轮距版（v3 主拓扑探针）
#   机体放开俯仰，前后两条同构腿（各自 B+C 轮），轮距 wb。
#   翻倒判据（E29 复活）：max|pitch| > TIP_DEG 记 tipover。
#   每腿按 m/2 定尺（两腿分担），接触参数沿用 v2.5 口径。
# ============================================================================
TIP_DEG = 40.0

def eval_wheel2(x7, m, v0, kc, zeta_c=0.15, base=None, v_x=0.0,
                mode="bc", wheel=None, wb=0.30, mat="al7075",
                T=1.2, h=2e-4, guards=True):
    """双腿轮足。mode ∈ {locked, brake, bc}（rigid 无意义——单腿基线用 eval_wheel）。
    返回始终带 metrics（能算出来的都算），fail 标签附加而非提前吞掉。"""
    w = dict(r_w=0.030, m_w=0.04, tau_max=8.0, omega_th=20.0)
    if wheel: w.update(wheel)
    r_w = float(w["r_w"]); wb = float(wb)

    scen = {**(base or P.SCEN_BIRD_X), "m": 0.5*m, "v0": v0, "kc": kc,
            "zeta_c": zeta_c, "v_x": v_x, "planar": True,
            "foot_mode": "bearing", "hip_damp_unified": True,
            "T": T, "h": h, "g": 9.81}
    s = P.size_x_v2(scen, x7)                      # 每腿按 m/2 定尺
    l1 = float(x7[0])/1000.; l2 = float(x7[1])*l1; l3 = float(x7[2])*l1
    g = s["g"]; vx = float(v_x)
    m1, m2, m3 = s["seg_mass"]
    a1 = s["q1_0"]; a2 = a1 + (np.pi - s["thetaA"]); a3 = a2 - (np.pi - s["thetaK"])
    dvec = lambda a: np.array([np.cos(a), 0., np.sin(a)])
    Fp0 = np.array([0., 0., s["gap0"] + r_w])
    A0 = Fp0 + l1*dvec(a1); K0 = A0 + l2*dvec(a2); H0 = K0 + l3*dvec(a3)

    SC = exu.SystemContainer(); mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])
    # 机体：长方体惯量（跨轮距摆布，俯仰惯量比小球诚实）
    C = H0 + np.array([0.5*wb, 0., 0.])            # 前髋在 H0，机体中心后移 wb/2 → 髋对称
    body = mbs.CreateRigidBody(name="payload", referencePosition=list(C),
                               initialVelocity=[vx, 0, -v0],
                               inertia=InertiaCuboid(density=m/((wb+0.2)*0.25*0.12),
                                                     sideLengths=[wb+0.2, 0.25, 0.12]),
                               gravity=[0, 0, -g])
    mbs.CreateGenericJoint(bodyNumbers=[ground, body], position=list(C),
                           constrainedAxes=[0,1,0,1,0,1])      # x,z,pitch 自由

    def rod(P0, P1, ang, mass):
        L = np.linalg.norm(P1-P0); com = 0.5*(P0+P1)
        return mbs.CreateRigidBody(referencePosition=list(com),
                                   referenceRotationMatrix=rotY(ang),
                                   initialVelocity=[vx,0,-v0],
                                   inertia=InertiaCuboid(density=mass/(L*0.02*0.02),
                                                         sideLengths=[L,0.02,0.02]),
                                   gravity=[0,0,-g])

    legs = []
    # 双腿必须镜像对称（前腿原向、后腿绕竖直面镜像）：
    #   同向双腿的水平分力不抵消，合力作用在地面、质心在高处 → 恒定俯仰力矩驱动
    #   摇摆发散，Fr=0 也翻（实测 90°）。镜像后水平分力对消，才站得住。
    # 足端 x 对称于质心（±wb/2）；髋位随各自几何链平移（真机=支架前后安装位）。
    for tag, fx0, mir in (("f", C[0] + 0.5*wb, False), ("r", C[0] - 0.5*wb, True)):
        if mir:
            b1_, b2_, b3_ = np.pi-a1, np.pi-a2, np.pi-a3
        else:
            b1_, b2_, b3_ = a1, a2, a3
        Fpm = np.array([fx0, 0., Fp0[2]])
        Am = Fpm + l1*dvec(b1_); Km = Am + l2*dvec(b2_); Hm = Km + l3*dvec(b3_)
        Fp, A, K, H = Fpm, Am, Km, Hm
        tarso = rod(Fp, A, b1_, m1); tibio = rod(A, K, b2_, m2); femur = rod(K, H, b3_, m3)
        for (b0,b1,Pj,kj,cj) in [(body,femur,H,s["k_hip"],s["c_hip"]),
                                 (femur,tibio,K,s["k_knee"],s["c_knee"]),
                                 (tibio,tarso,A,s["k_ankle"],s["c_ankle"])]:
            mbs.CreateRevoluteJoint(bodyNumbers=[b0,b1], position=list(Pj), axis=[0,1,0])
            mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0,b1], position=list(Pj),
                                            axis=[0,1,0], stiffness=kj, damping=cj)
        wheel_b = mbs.CreateRigidBody(referencePosition=list(Fp),
                                      referenceRotationMatrix=rotY(b1_),
                                      initialVelocity=[vx,0,-v0],
                                      inertia=_wheel_inertia(w["m_w"], r_w),
                                      gravity=[0,0,-g])
        oLock = None
        if mode == "locked":
            mbs.CreateGenericJoint(bodyNumbers=[tarso, wheel_b], position=list(Fp),
                                   constrainedAxes=[1,1,1,1,1,1])
        else:
            mbs.CreateRevoluteJoint(bodyNumbers=[tarso, wheel_b], position=list(Fp), axis=[0,1,0])
            tau_max = float(w["tau_max"]); om_th = float(w["omega_th"])
            def _brake(mbs_, t_, item_, rot, rot_t, k_, d_, off_, _tm=tau_max, _oth=om_th):
                frac = max(0.0, 1.0 - abs(rot_t)/max(_oth,1e-6))
                return -np.tanh(rot_t/0.5) * _tm * frac
            con = mbs.CreateTorsionalSpringDamper(bodyNumbers=[tarso, wheel_b],
                                                  position=list(Fp), axis=[0,1,0],
                                                  stiffness=0.0, damping=0.0)
            mbs.SetObjectParameter(con, "springTorqueUserFunction", _brake)
            if mode == "bc":
                oLock = mbs.CreateTorsionalSpringDamper(bodyNumbers=[tarso, wheel_b],
                                                        position=list(Fp), axis=[0,1,0],
                                                        stiffness=0.0, damping=0.0)
        quad = [[-30,-30,0],[30,-30,0],[30,30,0],[-30,30,0]]
        mbs.CreateSphereQuadContact(bodyNumbers=[wheel_b, ground],
                                    localPosition0=[0,0,0], radiusSphere=r_w,
                                    quadPoints=quad,
                                    contactStiffness=s["kc"], contactDamping=s["cc"],
                                    dynamicFriction=s["mu"], frictionProportionalZone=1e-3)
        OV = exu.OutputVariableType
        legs.append(dict(tag=tag, wheel=wheel_b, tarso=tarso, oLock=oLock,
            sWAV=mbs.AddSensor(SensorBody(bodyNumber=wheel_b, storeInternal=True, outputVariableType=OV.AngularVelocity)),
            sTAV=mbs.AddSensor(SensorBody(bodyNumber=tarso, storeInternal=True, outputVariableType=OV.AngularVelocity)),
            sWP =mbs.AddSensor(SensorBody(bodyNumber=wheel_b, storeInternal=True, outputVariableType=OV.Position))))

    OV = exu.OutputVariableType
    sAcc = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OV.Acceleration))
    sPos = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OV.Position))
    sRot = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OV.Rotation))
    sVel = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True, outputVariableType=OV.Velocity))

    if mode == "bc":
        K_LATCH, C_LATCH = 40.0, 0.05
        _latch = {L["tag"]: None for L in legs}
        def _prestep(mbs_, t_):
            if t_ > 0.004:
                for L in legs:
                    if _latch[L["tag"]] is None:
                        rel = mbs_.GetSensorValues(L["sWAV"])[1] - mbs_.GetSensorValues(L["sTAV"])[1]
                        if rel < -0.5:
                            phi = float(np.atleast_1d(mbs_.GetObjectOutput(
                                L["oLock"], exu.OutputVariableType.Rotation))[0])
                            mbs_.SetObjectParameter(L["oLock"], "offset", phi)
                            mbs_.SetObjectParameter(L["oLock"], "stiffness", K_LATCH)
                            mbs_.SetObjectParameter(L["oLock"], "damping", C_LATCH)
                            _latch[L["tag"]] = t_
            return True
        mbs.SetPreStepUserFunction(_prestep)
    else:
        _latch = {}

    mbs.Assemble()
    ss = exu.SimulationSettings()
    ss.timeIntegration.endTime = T
    ss.timeIntegration.numberOfSteps = int(T/h)
    ss.timeIntegration.generalizedAlpha.spectralRadius = 0.7
    ss.timeIntegration.verboseMode = 0
    ss.solutionSettings.writeSolutionToFile = False
    ss.solutionSettings.sensorsWritePeriod = h
    try:
        from hf_exudyn import silence_solver
        with silence_solver():
            mbs.SolveDynamic(ss)
    except Exception as e:
        return dict(fail="solver", mode=mode, err=str(e)[:80])

    acc = mbs.GetSensorStoredData(sAcc); pos = mbs.GetSensorStoredData(sPos)
    rot = mbs.GetSensorStoredData(sRot); vel = mbs.GetSensorStoredData(sVel)
    t = acc[:,0]; ax = acc[:,1]; az = acc[:,3]; x = pos[:,1]; z = pos[:,3]
    if not np.all(np.isfinite(az)):
        return dict(fail="nonfinite", mode=mode)
    pitch = np.degrees(rot[:,2])
    out = dict(mode=mode, wb_m=wb,
               peak_g=float(np.max(np.abs(az)))/9.81,
               a_res_g=float(np.max(np.hypot(ax,az)))/9.81,
               pitch_max_deg=float(np.max(np.abs(pitch))),
               pitch_end_deg=float(pitch[-1]),
               roll_dist_mm=float(x[-1]-x[0])*1e3,
               vx_end=float(np.mean(vel[-max(5,int(0.1/h)):,1])),
               body_drop_mm=float(z[0]-np.min(z))*1e3)
    out["stopped"] = bool(abs(out["vx_end"]) < 0.05)
    sinks, strokes, spins = [], [], []
    for L in legs:
        wz = mbs.GetSensorStoredData(L["sWP"])[:,3]
        sinks.append(max(0.0, r_w - float(np.min(wz))))
        ow = mbs.GetSensorStoredData(L["sWAV"])[:,2]; ot = mbs.GetSensorStoredData(L["sTAV"])[:,2]
        n = min(len(ow), len(ot)); spins.append(float(np.max(np.abs(ow[:n]-ot[:n]))))
    stroke_tot = float(z[0]-np.min(z))             # 机体下沉，双腿共同
    out.update(sink_mm=max(sinks)*1e3, wheel_spin_max=max(spins),
               latch_t=dict(_latch) if mode == "bc" else None)
    fail = None
    if guards:
        if out["pitch_max_deg"] > TIP_DEG: fail = "tipover"
        elif max(sinks) > 0.9*r_w:         fail = "deep_sink"
        elif stroke_tot - max(sinks) > 0.6*(l1+l2+l3): fail = "collapse"
    out["fail"] = fail
    if __import__("os").environ.get("W2_DEBUG"):
        out["_t"]=t.tolist(); out["_pitch"]=pitch.tolist(); out["_az"]=az.tolist(); out["_rx"]=np.degrees(rot[:,1]).tolist(); out["_rz"]=np.degrees(rot[:,3]).tolist()
    return out

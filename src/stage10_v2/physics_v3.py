# -*- coding: utf-8 -*-
"""v3 · 轮足物理与先验：双镜像腿 + B+C 轮（反离心刹车 + 单向棘轮）。

继承 v2.5 的全部口径（尺寸化 size_x_v2、结构定尺 size_structure、地面模型、
μ(kc)、材料 al7075），只把「刚性足球 × 1 条腿 + 锁俯仰」换成
「B+C 轮 × 前后镜像双腿 + 俯仰自由」。三条 v2.5 没有的物理：

  1. 翻倒——单点支撑变双点支撑后，俯仰自由度必须放开（M2 教训：
     同向双腿水平分力不抵消会摇摆发散；镜像 = 真飞机/四足站姿）。
  2. 滚动阻力 τ_rr = C_RR·N·r_w·tanh(ω/0.5)——没有它停车距离恒为无穷（M3 教训）。
  3. 停车距离——窗口内停住取实测；没停住用末段减速度外推 v²/2a。

设计 14 维 = v2.5 的 10 维腿 + 4 维轮：
    x[10] = wb    轮距 (m)          [0.50, 0.90]   M3: 0.3 两模式全灭，0.5 起 bc 全 Fr 存活
    x[11] = τ_max 刹车峰值力矩 (N·m) [2.0, 6.0]    M3: 16 N·m 翻倒率高（咬合过猛）
    x[12] = r_w   轮半径 (m)        [0.02, 0.045]  M3: 不敏感，留给结构权衡
    x[13] = ω_th  刹车脱开转速 (rad/s) [5, 60] 对数  M3 未扫——停车行为由它主导（v_eng=ω_th·r_w）

条件 8 维 = v2.5 的 6 维 + 2：
    [log10 m, v0, log10 kc, gcap, smax, Fr, pitch_cap, log10 Lstop]
    pitch_cap ∈ {15°, 25°, 40°} 抽签（俯仰裕度：载荷/桨叶越金贵越严）
    Lstop     ∈ [3, 15] m 对数抽签（跑道预算）
    两者与 gcap/smax 同法：事后抽签的设计要求，不进仿真。

本工作设定（写论文必须标注）：
    C_RR = 0.015（橡胶胎对硬面的滚阻系数，工程手册值，未实测）
    轮+刹车+轴承质量模型：m_w = 44.4·r_w² kg；m_brake = 0.02 + 0.008·τ_max kg；
    轴承 0.015 kg —— 均为小型件目录级估计。
    起落架质量预算 MASS_FRAC_CAP_V3 = 0.08（轮式系统口径高于 v2.5 裸腿的 0.06；
    民机全起落架 3–6%，小型轮式取上限加轮组余量）。
    棘轮闩锁刚度 40 N·m/rad（数值稳定上界附近；咬合后的弹性回摆≈齿隙柔性）。
"""
from __future__ import annotations
import numpy as np
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import physics_v2 as P                                    # noqa: E402
from bioprior import BioPrior                             # noqa: E402
import exudyn as exu                                      # noqa: E402
from exudyn.utilities import InertiaCuboid, InertiaSphere, SensorBody  # noqa: E402
from hf_exudyn import silence_solver                      # noqa: E402

rotY = P.rotY

# ---- v3 盒与常数（依据：P9w-3 标定，见《足端方案备忘_轮足》M3 节）----
WB_RANGE   = (0.50, 0.90)      # m
TAU_RANGE  = (2.0, 6.0)        # N·m
RW_RANGE   = (0.020, 0.045)    # m
OMTH_RANGE = (5.0, 60.0)       # rad/s，对数均匀
C_RR       = 0.015             # 滚动阻力系数（本工作设定）
K_LATCH, C_LATCH = 40.0, 0.05  # 棘轮闩锁（M1 验证值）
TIP_KILL_DEG = 60.0            # 评价级硬杀（不可恢复）；15/25/40 由事后判据抽签
MASS_FRAC_CAP_V3 = 0.08
T_V3, H_V3 = 2.0, 2e-4         # 停车观测窗口；步长与 M1–M3 一致

PITCH_CAPS = (15.0, 25.0, 40.0)          # 事后抽签档
LSTOP_RANGE = (3.0, 15.0)                # m，对数抽签


def wheel_masses(r_w, tau_max):
    """轮组质量模型（目录级估计，本工作设定）。返回单腿的 (轮, 刹车+轴承)。"""
    return 44.4 * r_w * r_w, 0.02 + 0.008 * tau_max + 0.015


class WheelPrior:
    """14 维设计先验 = BioPrior(bio,v25) 的 10 维腿 × 4 维轮工程盒。

    与 BioPrior 同接口（ndim / expand / contract / describe），
    工厂与训练管线无需分支。轮维与质量无关（纯工程盒）。
    """

    def __init__(self, arm="bio", **kw):
        self.leg = BioPrior(arm, v21=True, v25=True, **kw)
        self.ndim = self.leg.ndim + 4
        self.arm = arm

    def expand(self, u01, m_kg):
        u = np.atleast_2d(np.asarray(u01, float))
        X = np.atleast_2d(self.leg.expand(u[:, :self.leg.ndim], m_kg))
        lin = lambda uu, rg: rg[0] + (rg[1] - rg[0]) * uu
        logu = lambda uu, rg: 10 ** (np.log10(rg[0]) + (np.log10(rg[1]) - np.log10(rg[0])) * uu)
        W = np.column_stack([lin(u[:, self.leg.ndim + 0], WB_RANGE),
                             lin(u[:, self.leg.ndim + 1], TAU_RANGE),
                             lin(u[:, self.leg.ndim + 2], RW_RANGE),
                             logu(u[:, self.leg.ndim + 3], OMTH_RANGE)])
        x = np.concatenate([X, W], 1)
        return x[0] if np.ndim(u01) == 1 else x

    def contract(self, x14, m_kg):
        x = np.atleast_2d(np.asarray(x14, float))
        U = np.atleast_2d(self.leg.contract(x[:, :10], m_kg))
        inv = lambda v, rg: (v - rg[0]) / (rg[1] - rg[0])
        loginv = lambda v, rg: ((np.log10(v) - np.log10(rg[0]))
                                / (np.log10(rg[1]) - np.log10(rg[0])))
        W = np.column_stack([inv(x[:, 10], WB_RANGE), inv(x[:, 11], TAU_RANGE),
                             inv(x[:, 12], RW_RANGE), loginv(x[:, 13], OMTH_RANGE)])
        u = np.concatenate([U, W], 1)
        return u[0] if np.ndim(x14) == 1 else u

    def describe(self):
        d = self.leg.describe()
        d.update(v3=True, ndim=self.ndim, wb_range=list(WB_RANGE),
                 tau_range=list(TAU_RANGE), rw_range=list(RW_RANGE),
                 omth_range=list(OMTH_RANGE), c_rr=C_RR,
                 mass_frac_cap=MASS_FRAC_CAP_V3)
        return d


# ------------------------------------------------------------------ 评价
def eval_v3(x14, m, v0, kc, zeta_c=0.15, base=None, v_x=0.0,
            mat="al7075", T=T_V3, h=H_V3):
    """完整 v3 评价：双镜像腿轮足落震+滑跑 → 结构定尺 → 判据所需全部量。

    始终返回 dict；硬失败带 fail ∈ {solver, nonfinite, collapse, deep_sink, tip_kill}。
    可判量（pitch_max / stop_dist 等）不在这里判死，留给 feasible_v3 按抽签判。
    """
    x14 = [float(v) for v in x14]
    x10, (wb, tau_max, r_w, om_th) = x14[:10], x14[10:]
    m = float(m)
    m_w, m_hub = wheel_masses(r_w, tau_max)

    scen = {**(base or P.SCEN_BIRD_X), "m": 0.5 * m, "v0": float(v0),
            "kc": float(kc), "zeta_c": float(zeta_c), "v_x": float(v_x),
            "planar": True, "foot_mode": "bearing", "hip_damp_unified": True,
            "mu_from_ground": True, "T": T, "h": h, "g": 9.81}
    s = P.size_x_v2(scen, x10)                 # 每腿按 m/2 定尺（v2.5 同式）
    l1 = x10[0] / 1000.0; l2 = x10[1] * l1; l3 = x10[2] * l1
    g = s["g"]; vx = float(v_x)
    m1, m2, m3 = s["seg_mass"]
    a1 = s["q1_0"]; a2 = a1 + (np.pi - s["thetaA"]); a3 = a2 - (np.pi - s["thetaK"])
    dvec = lambda a: np.array([np.cos(a), 0., np.sin(a)])
    Fp0 = np.array([0., 0., s["gap0"] + r_w])
    H0 = Fp0 + l1 * dvec(a1) + l2 * dvec(a2) + l3 * dvec(a3)

    SC = exu.SystemContainer(); mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])
    C = np.array([0.0, 0.0, H0[2]])
    Mtot = m + 2 * (m1 + m2 + m3 + m_w + m_hub)
    body = mbs.CreateRigidBody(referencePosition=list(C),
                               initialVelocity=[vx, 0, -v0],
                               inertia=InertiaCuboid(density=m / ((wb + 0.2) * 0.25 * 0.12),
                                                     sideLengths=[wb + 0.2, 0.25, 0.12]),
                               gravity=[0, 0, -g])
    mbs.CreateGenericJoint(bodyNumbers=[ground, body], position=list(C),
                           constrainedAxes=[0, 1, 0, 1, 0, 1])   # x,z,俯仰自由

    def rod(P0, P1, ang, mass):
        L = np.linalg.norm(P1 - P0); com = 0.5 * (P0 + P1)
        return mbs.CreateRigidBody(referencePosition=list(com),
                                   referenceRotationMatrix=rotY(ang),
                                   initialVelocity=[vx, 0, -v0],
                                   inertia=InertiaCuboid(density=mass / (L * 0.02 * 0.02),
                                                         sideLengths=[L, 0.02, 0.02]),
                                   gravity=[0, 0, -g])

    OV = exu.OutputVariableType
    N_st = 0.5 * Mtot * g                       # 单轮静载（滚阻用）
    legs = []
    for tag, fx0, mir in (("f", +0.5 * wb, False), ("r", -0.5 * wb, True)):
        b1_, b2_, b3_ = ((np.pi - a1, np.pi - a2, np.pi - a3) if mir
                         else (a1, a2, a3))
        Fp = np.array([fx0, 0., Fp0[2]])
        A = Fp + l1 * dvec(b1_); K = A + l2 * dvec(b2_); H = K + l3 * dvec(b3_)
        tarso = rod(Fp, A, b1_, m1); tibio = rod(A, K, b2_, m2); femur = rod(K, H, b3_, m3)
        for (b0, b1, Pj, kj, cj) in [(body, femur, H, s["k_hip"], s["c_hip"]),
                                     (femur, tibio, K, s["k_knee"], s["c_knee"]),
                                     (tibio, tarso, A, s["k_ankle"], s["c_ankle"])]:
            mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1], position=list(Pj), axis=[0, 1, 0])
            mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0, b1], position=list(Pj),
                                            axis=[0, 1, 0], stiffness=kj, damping=cj)
        wheel_b = mbs.CreateRigidBody(referencePosition=list(Fp),
                                      referenceRotationMatrix=rotY(b1_),
                                      initialVelocity=[vx, 0, -v0],
                                      inertia=InertiaCuboid(
                                          density=m_w / (np.pi * r_w**2 * 0.3 * r_w),
                                          sideLengths=[2 * r_w, 0.3 * r_w, 2 * r_w]),
                                      gravity=[0, 0, -g])
        mbs.CreateRevoluteJoint(bodyNumbers=[tarso, wheel_b], position=list(Fp),
                                axis=[0, 1, 0])

        def _brake(mbs_, t_, item_, rot, rot_t, k_, d_, off_,
                   _tm=tau_max, _oth=om_th, _nr=C_RR * N_st * r_w):
            om = rot_t
            frac = max(0.0, 1.0 - abs(om) / max(_oth, 1e-6))
            # ⚠ exudyn 符号约定（台架实测 2026-09-19）：springTorqueUserFunction
            # 的返回值代入内置公式 k·Δ+d·ω 的位置，随内置的负反馈符号一起施加——
            # 所以「阻力」必须返回 +|τ|·sign(ω)。写 − 号会变成电机（M1–M3 踩过的坑）。
            return (np.tanh(om / 0.5) * _tm * frac         # 反离心刹车
                    + np.tanh(om / 0.5) * _nr)             # 滚动阻力（v3 新增）
        con = mbs.CreateTorsionalSpringDamper(bodyNumbers=[tarso, wheel_b],
                                              position=list(Fp), axis=[0, 1, 0],
                                              stiffness=0.0, damping=0.0)
        mbs.SetObjectParameter(con, "springTorqueUserFunction", _brake)
        oLock = mbs.CreateTorsionalSpringDamper(bodyNumbers=[tarso, wheel_b],
                                                position=list(Fp), axis=[0, 1, 0],
                                                stiffness=0.0, damping=0.0)
        quad = [[-60, -60, 0], [60, -60, 0], [60, 60, 0], [-60, 60, 0]]
        mbs.CreateSphereQuadContact(bodyNumbers=[wheel_b, ground],
                                    localPosition0=[0, 0, 0], radiusSphere=r_w,
                                    quadPoints=quad,
                                    contactStiffness=s["kc"], contactDamping=s["cc"],
                                    dynamicFriction=s["mu"],
                                    frictionProportionalZone=1e-3)
        legs.append(dict(
            tag=tag, oLock=oLock,
            rods=(tarso, tibio, femur),
            sWAV=mbs.AddSensor(SensorBody(bodyNumber=wheel_b, storeInternal=True,
                                          outputVariableType=OV.AngularVelocity)),
            sTAV=mbs.AddSensor(SensorBody(bodyNumber=tarso, storeInternal=True,
                                          outputVariableType=OV.AngularVelocity)),
            sWP=mbs.AddSensor(SensorBody(bodyNumber=wheel_b, storeInternal=True,
                                         outputVariableType=OV.Position)),
            sRot=[mbs.AddSensor(SensorBody(bodyNumber=b, storeInternal=True,
                                           outputVariableType=OV.Rotation))
                  for b in (femur, tibio, tarso)]))

    sAcc = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                    outputVariableType=OV.Acceleration))
    sPos = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                    outputVariableType=OV.Position))
    sRotB = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                     outputVariableType=OV.Rotation))
    sVel = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                    outputVariableType=OV.Velocity))

    _latch = {"f": None, "r": None}
    def _prestep(mbs_, t_):
        if t_ > 0.004:
            for L in legs:
                if _latch[L["tag"]] is None:
                    rel = (mbs_.GetSensorValues(L["sWAV"])[1]
                           - mbs_.GetSensorValues(L["sTAV"])[1])
                    if rel < -0.5:
                        phi = float(np.atleast_1d(mbs_.GetObjectOutput(
                            L["oLock"], OV.Rotation))[0])
                        mbs_.SetObjectParameter(L["oLock"], "offset", phi)
                        mbs_.SetObjectParameter(L["oLock"], "stiffness", K_LATCH)
                        mbs_.SetObjectParameter(L["oLock"], "damping", C_LATCH)
                        _latch[L["tag"]] = t_
        return True
    mbs.SetPreStepUserFunction(_prestep)

    mbs.Assemble()
    ss = exu.SimulationSettings()
    ss.timeIntegration.endTime = T
    ss.timeIntegration.numberOfSteps = int(T / h)
    ss.timeIntegration.generalizedAlpha.spectralRadius = 0.7
    ss.timeIntegration.verboseMode = 0
    ss.solutionSettings.writeSolutionToFile = False
    ss.solutionSettings.sensorsWritePeriod = h
    try:
        with silence_solver():
            mbs.SolveDynamic(ss)
    except Exception:
        return dict(fail="solver")

    acc = mbs.GetSensorStoredData(sAcc); pos = mbs.GetSensorStoredData(sPos)
    rotB = mbs.GetSensorStoredData(sRotB); vel = mbs.GetSensorStoredData(sVel)
    t = acc[:, 0]; ax = acc[:, 1]; az = acc[:, 3]; x = pos[:, 1]; z = pos[:, 3]
    if not np.all(np.isfinite(az)):
        return dict(fail="nonfinite")
    pitch = np.degrees(rotB[:, 2])
    pitch_max = float(np.max(np.abs(pitch)))
    if pitch_max > TIP_KILL_DEG:
        return dict(fail="tip_kill", pitch_max_deg=pitch_max)

    # ---- 行程 / 下陷 / 塌陷 ----
    stroke = float(z[0] - np.min(z))
    sinks, spins = [], []
    for L in legs:
        wz = mbs.GetSensorStoredData(L["sWP"])[:, 3]
        sinks.append(max(0.0, r_w - float(np.min(wz))))
        ow = mbs.GetSensorStoredData(L["sWAV"])[:, 2]
        ot = mbs.GetSensorStoredData(L["sTAV"])[:, 2]
        n = min(len(ow), len(ot)); spins.append(float(np.max(np.abs(ow[:n] - ot[:n]))))
    sink = max(sinks)
    leg_stroke = float(max(0.0, stroke - sink))
    if leg_stroke > 0.6 * (l1 + l2 + l3):
        return dict(fail="collapse")
    if sink > 0.9 * r_w:
        return dict(fail="deep_sink")

    # ---- 停车距离 ----
    # 「停住」= 存在某时刻起速度一直低于 0.15 m/s（此后不再显著前进）。
    # 不能只看末段均值符号：棘轮/接触让机身在停下后小幅前后颤，
    # 末段均值可能落在 ±0.05 之外却早已不走了（会被误判没停 + 外推出 inf）。
    # 速度靠滚阻+刹车单调下降（不会自发再加速），所以全程最小速度点即停车点。
    # 用 argmin 而非阈值窗口：避免末段在 0.08–0.15 徘徊时判据落进 inf 毛刺。
    vx_t = vel[:, 1]; speed = np.abs(vx_t)
    roll = float(x[-1] - x[0])
    i_min = int(np.argmin(speed))
    if speed[i_min] < 0.10 and i_min < len(speed) - 3:
        stopped = True
        stop_dist = abs(float(x[i_min] - x[0])); extrap = 0.0
        vx_end = float(vx_t[i_min])
    else:
        stopped = False
        n_tail = max(5, int(0.30 / h))
        vx_end = float(np.mean(vx_t[-max(5, int(0.10 / h)):]))
        dec = -(np.polyfit(t[-n_tail:], speed[-n_tail:], 1)[0])   # 末段平均减速度
        if dec > 0.05:
            stop_dist = abs(roll) + speed[-1]**2 / (2.0 * dec); extrap = 1.0
        else:
            stop_dist = float("inf"); extrap = 1.0

    # ---- 回弹（v2.5 同式：最大压缩后机身回到触地高度以上的高度）----
    F = Mtot * (az + g); thr = 0.02 * Mtot * g
    idx = np.where(F > thr)[0]
    i0 = int(idx[0]) if len(idx) else 0
    imin = int(np.argmin(z))
    rebound = float(max(0.0, np.max(z[imin:]) - z[i0])) if imin > i0 else 0.0

    # ---- 结构定尺（v2.5 同式；每条模拟腿 = 左右一对，nlegs=2 分担对内载荷）----
    hgrid = t[1] - t[0]
    worst = None
    for L in legs:
        r_ = [mbs.GetSensorStoredData(si)[:, 2] for si in L["sRot"]]
        dth = dict(hip=r_[0] - r_[0][0],
                   knee=(r_[1] - r_[0]) - (r_[1][0] - r_[0][0]),
                   ankle=(r_[2] - r_[1]) - (r_[2][0] - r_[1][0]))
        Mj = {}
        for jn, kk_, cc_ in [("hip", s["k_hip"], s["c_hip"]),
                             ("knee", s["k_knee"], s["c_knee"]),
                             ("ankle", s["k_ankle"], s["c_ankle"])]:
            th = dth[jn]
            Mj[jn] = float(np.max(np.abs(kk_ * th + cc_ * np.gradient(th, hgrid))))
        if worst is None or sum(Mj.values()) > sum(worst.values()):
            worst = Mj
    met_sz = dict(M_ankle=worst["ankle"], M_knee=worst["knee"], M_hip=worst["hip"],
                  F_peak=0.7 * Mtot * (float(np.max(np.abs(az))) + g),  # 不均分担系数 1.4/2
                  seg_len=[l1, l2, l3])
    rows, pair_mass, over = P.size_structure(met_sz, mat=mat)   # nlegs=2 → 一对腿的质量
    leg_mass = 2.0 * pair_mass                                   # 前后两对
    gear_mass = leg_mass + 4.0 * (m_w + m_hub)                   # 4 轮（每条模拟腿=一对）
    frac = gear_mass / m

    a_res = float(np.max(np.hypot(ax, az)))
    return dict(
        peak_a=float(np.max(np.abs(az))), a_res=a_res,
        stroke=stroke, leg_stroke=leg_stroke, sink=sink,
        rebound=rebound, pitch_max_deg=pitch_max, pitch_end_deg=float(pitch[-1]),
        roll_dist_m=abs(roll), stop_dist_m=float(stop_dist), stopped=float(stopped),
        stop_extrap=extrap, vx_end=vx_end,
        wheel_spin_max=max(spins),
        latch_f=(-1.0 if _latch["f"] is None else float(_latch["f"])),
        latch_r=(-1.0 if _latch["r"] is None else float(_latch["r"])),
        leg_mass_kg=leg_mass, gear_mass_kg=gear_mass, mass_frac=frac,
        struct_over=bool(over), mass_over=bool(frac > MASS_FRAC_CAP_V3),
        D1_mm=rows[0]["D_mm"], D2_mm=rows[1]["D_mm"], D3_mm=rows[2]["D_mm"],
        L1_mm=l1 * 1e3, L2_mm=l2 * 1e3, L3_mm=l3 * 1e3,
        m=m, v0=float(v0), kc=float(kc), v_x=vx, mu_ground=float(s["mu"]),
        wb=wb, tau_max=tau_max, r_w=r_w, om_th=om_th)


# ------------------------------------------------------------------ 判据
def feasible_v3(r, gcap, smax, pitch_cap_deg, lstop_cap_m,
                reb_cap=0.05):
    """v3 可行性。返回 (是否可行, 全部违反项)。

    继承 v2.5：gcap(判 a_res)、smax(腿行程)、slenderness、massbudget、rebound。
    去掉 slip（轮子本来就滚）。新增：
      tipover : pitch_max > pitch_cap（抽签 15/25/40°）
      stop    : stop_dist > Lstop（抽签 [3,15] m 对数）——外推值同判。
    """
    if r is None:
        return False, ["none"]
    if r.get("fail"):
        return False, [r["fail"]]
    if not np.isfinite(r.get("peak_a", np.nan)):
        return False, ["nonfinite"]
    bad = []
    if r["a_res"] > gcap:
        bad.append("gcap")
    if r["leg_stroke"] > smax:
        bad.append("smax")
    if r["struct_over"]:
        bad.append("slenderness")
    if r["mass_over"]:
        bad.append("massbudget")
    v0 = r.get("v0", 0.0)
    if v0:
        h0 = v0 * v0 / (2 * 9.81)
        if h0 > 1e-12 and r.get("rebound", 0.0) / h0 > reb_cap:
            bad.append("rebound")
    if r["pitch_max_deg"] > pitch_cap_deg:
        bad.append("tipover")
    sd = r.get("stop_dist_m", float("inf"))
    if not np.isfinite(sd) or sd > lstop_cap_m:
        bad.append("stop")
    return (len(bad) == 0), bad


def draw_requirements(rng):
    """事后抽签的 v3 设计要求（gcap/smax 沿用 v2.5 的抽法，由调用方传入范围）。"""
    return dict(pitch_cap=float(rng.choice(PITCH_CAPS)),
                lstop=float(10 ** rng.uniform(np.log10(LSTOP_RANGE[0]),
                                              np.log10(LSTOP_RANGE[1]))))

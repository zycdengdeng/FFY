"""v2 物理:把 m 从"被人为消去"改回"真正参与"。

v1 的病根是每个力都写成了 m 的正比量,于是 a=F/m 里 m 约掉(见 E14 验尸)。
v2 只改三处**写错的物理**,不动良构的无量纲设计变量:

  P1.1 接触 = 绝对介质属性,并升为工况维
       v1: kc = 4000·m·g     ← 地面对更重的机体更硬,不是物理
       v2: kc ∈ [3e3, 2e6] N/m 由地形决定,与 m 无关;
           cc = 2·ζc·√(kc·m)(标准阻尼比参数化,含辐射阻尼的 √m)
  P1.2 杆件质量 = 结构定尺导出,不再是 5%·m
       每段按关节力矩定薄壁圆管(壁厚=0.1D),密度是绝对材料属性
  P1.3 结构判据进在线可行性,且**有几何上限**
       D ≤ 0.25×段长(再粗就不是杆件了,且与相邻段干涉)、D ≥ 4mm(可制造下限)
       —— 没有上限的话应力/屈曲永远不会卡住(粗一点就行),平方立方律就咬不到人

**刻意不改**:κ = k/(m·g·L_leg)、ζ、r2、r3 继续用无量纲形式。它们是良构的
无量纲群(簧扭矩比重力扭矩),改成绝对单位只会让 1kg 与 12kg 的取值范围失配。
不变性应当由**绝对量**打破,而不是把好的参数化拆掉。
"""
from __future__ import annotations

import os
import sys

import numpy as np
import exudyn as exu
from exudyn.utilities import *   # noqa: F401,F403

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6_surrogate"))
sys.path.insert(0, os.path.join(HERE, "..", "stage8_struct"))
from hf_exudyn import (SCEN_BIRD_X, NAN_METRICS, rotY, silence_solver,   # noqa: E402
                       _metrics)
from e8_struct import MATERIALS, size_segment                            # noqa: E402

# ---------------------------------------------------------------- 介质(工况维)
# 绝对刚度 [N/m] 与阻尼比;与飞行器质量无关——这正是打破不变性的地方。
TERRAIN = {
    "concrete": dict(kc=2.0e6, zeta_c=0.05),
    "asphalt":  dict(kc=8.0e5, zeta_c=0.08),
    "turf":     dict(kc=2.0e5, zeta_c=0.15),
    "wetsand":  dict(kc=5.0e4, zeta_c=0.30),
    "softmud":  dict(kc=2.0e4, zeta_c=0.35),   # 12 kg 端已接近模型边界,慎用
}
# 工况采样按对数均匀。软端由**模型有效性**决定而非物理:侵入深度超过足端球半径时
# 球-面罚接触失效(记 fail="deep_sink",绝不与真实塌陷混淆)。
# 实测有效边界(足端 = 0.20·L1):kc ≥ 5e4 在 m∈[1,12] 全程有效;
# kc ∈ [2e4, 5e4) 仅在 m ≲ 8 kg 有效。更软(泥/水)需 v²拖曳+附加质量模型 → v2.1。
# 上界从 2e6 收到 1e6:足端不是刚体,弹性足垫与地面**串联**,
# 有效接触刚度由较软者封顶;2e6 N/m 经一个 20mm 半径的足垫传递并不现实。
KC_RANGE = (5.0e4, 1.0e6)
ZETA_C_RANGE = (0.05, 0.35)

# ---------------------------------------------------------------- 结构(绝对量)
MAT_DEFAULT = "cfnylon"
SF = 2.0                 # 安全系数
D_MAX_RATIO = 0.25       # 外径 / 段长 上限:超过即判不可行(杆件假设失效+干涉)
D_MIN = 0.004            # 最小可制造外径 4mm(壁厚 0.4mm)
NLEGS = 2                # 双腿分担
MASS_FRAC_CAP = 0.06     # 腿总质量 / 机体质量 上限(6%,航空口径的结构质量预算)
# 足端等效半径按跗跖长缩放(蹼足)。v1 固定 8 mm,与 33–121 mm 的跗跖长不自洽,
# 且在软介质上会让侵入深度超过球半径,使球-面罚接触模型失效(被误判成"腿塌了")。
ZENER_SUPPORTED = True   # P7:关节 Zener 化已实现(见 exu_eval_v2)
FOOT_RATIO = 0.20
SEG_FRAC_GUESS = 0.02    # 第一遍的杆件质量猜测(占 m 的比例)


def size_x_v2(scen, x7, seg_mass=None):
    """把 7 维设计尺寸化成 Exudyn 场景。与 v1 的差别只在接触与杆件质量。"""
    xv = [float(v) for v in x7]
    L1, r2, r3, ka, kk, kh, z = xv[:7]
    l1 = L1 / 1000.0
    Lleg = l1 * (1.0 + r2 + r3)
    mgL = scen["m"] * scen["g"] * Lleg
    s = dict(scen)
    s["k_ankle"] = ka * mgL; s["k_knee"] = kk * mgL; s["k_hip"] = kh * mgL
    s["c_ankle"] = z * s["k_ankle"]; s["c_knee"] = z * s["k_knee"]
    # P1 对照开关:hip_damp_unified=True 时髋与踝/膝同式 c = τ·k;
    # 默认保持 v1 遗留特例式(等效松弛时间为踝/膝的 6.67 倍),不影响任何既有结果
    s["c_hip"] = (z * s["k_hip"] if scen.get("hip_damp_unified")
                  else (z / 0.03) * 0.2 * s["k_hip"])
    # —— 这两行是 v2 的全部要害 ——
    s["kc"] = float(scen["kc"])                                   # 绝对,不含 m
    s["cc"] = (0.01 * s["kc"] if scen.get("legacy_cc") else
               2.0 * float(scen.get("zeta_c", 0.15)) * np.sqrt(s["kc"] * scen["m"]))
    s["seg_mass"] = (list(seg_mass) if seg_mass is not None
                     else [SEG_FRAC_GUESS * scen["m"]] * 3)
    s["seg_len"] = [l1, r2 * l1, r3 * l1]
    s["r_foot"] = FOOT_RATIO * l1
    s["gap0"] = 0.3 * s["r_foot"]          # 初始离地间隙随足端一起缩放
    if len(xv) >= 9:                       # v2.1:姿态(度)是设计向量的第 8/9 维
        s["thetaA"] = np.radians(xv[7]); s["thetaK"] = np.radians(xv[8])
    return s


def exu_eval_v2(x7, s):
    """与 v1 同拓扑,但每段质量独立给定(结构定尺导出),接触参数为绝对量。"""
    L1, r2, r3 = float(x7[0]), float(x7[1]), float(x7[2])
    l1 = L1 / 1000.0; l2 = r2 * l1; l3 = r3 * l1
    m, g, v0 = s["m"], s["g"], s["v0"]
    m1, m2, m3 = s["seg_mass"]
    a1 = s["q1_0"]; a2 = a1 + (np.pi - s["thetaA"]); a3 = a2 - (np.pi - s["thetaK"])
    d = lambda a: np.array([np.cos(a), 0., np.sin(a)])
    Fp = np.array([0., 0., s["gap0"] + s["r_foot"]])
    A = Fp + l1 * d(a1); K = A + l2 * d(a2); H = K + l3 * d(a3)

    SC = exu.SystemContainer(); mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])

    def rod(name, P0, P1, ang, mass):
        L = np.linalg.norm(P1 - P0); com = 0.5 * (P0 + P1)
        inertia = InertiaCuboid(density=mass / (L * 0.02 * 0.02),
                                sideLengths=[L, 0.02, 0.02])
        return mbs.CreateRigidBody(name=name, referencePosition=list(com),
                                   referenceRotationMatrix=rotY(ang),
                                   initialVelocity=[0, 0, -v0],
                                   inertia=inertia, gravity=[0, 0, -g])
    tarso = rod("tarso", Fp, A, a1, m1)
    tibio = rod("tibio", A, K, a2, m2)
    femur = rod("femur", K, H, a3, m3)
    body = mbs.CreateRigidBody(name="payload", referencePosition=list(H),
                               initialVelocity=[0, 0, -v0],
                               inertia=InertiaSphere(mass=m, radius=0.12),
                               gravity=[0, 0, -g])
    mbs.CreatePrismaticJoint(bodyNumbers=[ground, body], position=list(H), axis=[0, 0, 1])
    # P7:zener = dict(ratio=k2/k1, joints=("ankle","knee","hip")) → 该关节改为
    # 标准线性固体 k1 ∥ (k2 串 c):加一个 ODE1 内部状态 y(Maxwell 阻尼器转角),
    # ẏ = k2(θ−y)/c,力矩 = k1·θ + k2(θ−y)。无附加惯量,故不引入伪高频模态。
    zen = s.get("zener"); zjoints = set(zen["joints"]) if zen else set()
    _zstate = {}
    for (jn, b0, b1, P, kj, cj) in [("hip", body, femur, H, s["k_hip"], s["c_hip"]),
                                    ("knee", femur, tibio, K, s["k_knee"], s["c_knee"]),
                                    ("ankle", tibio, tarso, A, s["k_ankle"], s["c_ankle"])]:
        mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1], position=list(P), axis=[0, 1, 0])
        if zen and jn in zjoints and cj > 0:
            k2 = float(zen["ratio"]) * kj
            nd = mbs.AddNode(NodeGenericODE1(referenceCoordinates=[0.],
                                             initialCoordinates=[0.],
                                             numberOfODE1Coordinates=1))
            idx = len(_zstate); _zstate[jn] = dict(th=0.0, idx=idx)
            st = _zstate[jn]
            def _rhs(mbs_, t_, item_, q_, _k2=k2, _c=cj, _st=st):
                return [_k2 * (_st["th"] - q_[0]) / _c]
            mbs.AddObject(ObjectGenericODE1(nodeNumbers=[nd], rhsUserFunction=_rhs))
            def _tq(mbs_, t_, item_, rot, rot_t, k_, d_, off_,
                    _k2=k2, _st=st):
                _st["th"] = rot
                y = mbs_.systemData.GetODE1Coordinates()[_st["idx"]]
                return k_ * rot + _k2 * (rot - y)
            con = mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0, b1], position=list(P),
                                                  axis=[0, 1, 0], stiffness=kj, damping=0.0)
            mbs.SetObjectParameter(con, "springTorqueUserFunction", _tq)
        else:
            mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0, b1], position=list(P),
                                            axis=[0, 1, 0], stiffness=kj, damping=cj)
    # ---- P4c:膝-髋耦合连杆(无质量两力杆 ≡ 两点间距离约束) ----
    # scen["couple_rod"] = dict(off_hip=米, off_knee=米);缺省不加,老路径完全不变。
    cr = s.get("couple_rod")
    if cr:
        e = lambda a: np.array([-np.sin(a), 0., np.cos(a)])   # 面内法向(局部 z)
        nf = e(a3); nf = -nf if nf[0] > 0 else nf             # 股骨法向,取同一侧
        ns = e(a2); ns = -ns if ns[0] > 0 else ns             # 胫跗骨法向
        Pb = H + nf * float(cr["off_hip"])                    # 机身(支座)端
        Qg = K + ns * float(cr["off_knee"])                   # 胫跗骨端
        com_t = 0.5 * (A + K)                                 # 胫跗骨质心
        q_loc = [float(np.dot(Qg - com_t, d(a2))), 0.,
                 float(np.dot(Qg - com_t, e(a2)))]
        mbs.CreateDistanceConstraint(bodyNumbers=[body, tibio],
                                     localPosition0=list(Pb - H),
                                     localPosition1=q_loc,
                                     distance=float(np.linalg.norm(Pb - Qg)))

    quad = [[-3., -3., 0.], [3., -3., 0.], [3., 3., 0.], [-3., 3., 0.]]
    mbs.CreateSphereQuadContact(bodyNumbers=[tarso, ground],
                                localPosition0=[-0.5 * l1, 0., 0.],
                                radiusSphere=s["r_foot"], quadPoints=quad,
                                contactStiffness=s["kc"], contactDamping=s["cc"],
                                dynamicFriction=s["mu"], frictionProportionalZone=1e-3)
    sAcc = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                    outputVariableType=exu.OutputVariableType.Acceleration))
    sPos = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                    outputVariableType=exu.OutputVariableType.Position))
    sRot = [mbs.AddSensor(SensorBody(bodyNumber=b, storeInternal=True,
                                     outputVariableType=exu.OutputVariableType.Rotation))
            for b in (femur, tibio, tarso)]
    sFoot = mbs.AddSensor(SensorBody(                     # 足端球心,用于分离地面下陷
        bodyNumber=tarso, storeInternal=True, localPosition=[-0.5 * l1, 0., 0.],
        outputVariableType=exu.OutputVariableType.Position))
    # 踝(A)与膝(K)的实测位置:动画复原骨架用,不参与任何判据
    sAnk = mbs.AddSensor(SensorBody(
        bodyNumber=tarso, storeInternal=True, localPosition=[0.5 * l1, 0., 0.],
        outputVariableType=exu.OutputVariableType.Position))
    sKne = mbs.AddSensor(SensorBody(
        bodyNumber=tibio, storeInternal=True, localPosition=[0.5 * l2, 0., 0.],
        outputVariableType=exu.OutputVariableType.Position))
    sHip = mbs.AddSensor(SensorBody(
        bodyNumber=femur, storeInternal=True, localPosition=[0.5 * l3, 0., 0.],
        outputVariableType=exu.OutputVariableType.Position))
    mbs.Assemble()
    ss = exu.SimulationSettings()
    ss.timeIntegration.endTime = s["T"]
    ss.timeIntegration.numberOfSteps = int(s["T"] / s["h"])
    ss.timeIntegration.generalizedAlpha.spectralRadius = 0.7
    ss.timeIntegration.verboseMode = 0
    ss.solutionSettings.writeSolutionToFile = False
    ss.solutionSettings.sensorsWritePeriod = s["h"]
    try:
        with silence_solver():
            mbs.SolveDynamic(ss)
    except Exception:
        return dict(fail="solver")
    acc = mbs.GetSensorStoredData(sAcc); pos = mbs.GetSensorStoredData(sPos)
    t = acc[:, 0]; az = acc[:, 3]; z = pos[:, 3]
    if not np.all(np.isfinite(az)):
        return dict(fail="nonfinite")
    stroke = float(z[0] - np.min(z))                     # 机体总下沉
    zf = mbs.GetSensorStoredData(sFoot)[:, 3]
    sink = float(max(0.0, s["r_foot"] - np.min(zf)))     # 地面下陷(足球心低于半径部分)
    leg_stroke = float(max(0.0, stroke - sink))          # 起落架自身行程
    if leg_stroke > 0.6 * (l1 + l2 + l3):
        return dict(fail="collapse")     # 腿真被压塌(已扣除地面下陷),设计不行
    if sink > 0.9 * s["r_foot"]:
        return dict(fail="deep_sink")    # 侵入超过足端球半径 → 罚接触模型失效,非物理
    rot = [mbs.GetSensorStoredData(si)[:, 2] for si in sRot]
    footxyz = mbs.GetSensorStoredData(sFoot)[:, 1:4]
    h = t[1] - t[0]
    dth = dict(hip=rot[0] - rot[0][0],
               knee=(rot[1] - rot[0]) - (rot[1][0] - rot[0][0]),
               ankle=(rot[2] - rot[1]) - (rot[2][0] - rot[1][0]))
    Mj = {}
    for jn, kk_, cc_ in [("hip", s["k_hip"], s["c_hip"]),
                         ("knee", s["k_knee"], s["c_knee"]),
                         ("ankle", s["k_ankle"], s["c_ankle"])]:
        th = dth[jn]
        Mj[jn] = float(np.max(np.abs(kk_ * th + cc_ * np.gradient(th, h))))
    met = dict(peak_a=float(np.max(np.abs(az))), stroke=stroke)
    met.update(_metrics(t, z, az, m, g, v0))
    met.update(M_hip=Mj["hip"], M_knee=Mj["knee"], M_ankle=Mj["ankle"],
               seg_len=[l1, l2, l3], sink=sink, leg_stroke=leg_stroke)
    if s.get("keep_history"):
        # 留下整条时程供动画重播。默认关闭:一条时程约 40 万个数,批量跑时不要开。
        def _M(jn, kk_, cc_):
            th = dth[jn]
            return kk_ * th + cc_ * np.gradient(th, h)
        met["hist"] = dict(
            t=t, az=az, z=z, foot=footxyz,
            ankle_p=mbs.GetSensorStoredData(sAnk)[:, 1:4],
            knee_p=mbs.GetSensorStoredData(sKne)[:, 1:4],
            hip_p=mbs.GetSensorStoredData(sHip)[:, 1:4],
            th_hip=dth["hip"], th_knee=dth["knee"], th_ankle=dth["ankle"],
            M_hip=_M("hip", s["k_hip"], s["c_hip"]),
            M_knee=_M("knee", s["k_knee"], s["c_knee"]),
            M_ankle=_M("ankle", s["k_ankle"], s["c_ankle"]),
            seg_len=[l1, l2, l3], r_foot=s["r_foot"], m=m, g=g, v0=v0)
    return met


def size_structure(met, mat=MAT_DEFAULT, sf=SF, nlegs=NLEGS):
    """由关节力矩定薄壁圆管;返回每段外径/质量/控制工况,以及几何上限是否被突破。

    与 e8_struct.size_leg 的唯一差别:**加了 D ≤ D_MAX_RATIO×段长 的几何上限**。
    这条上限是让平方立方律真正咬人的关键——没有它,应力和屈曲永远可以靠加粗化解。
    """
    M = MATERIALS[mat]
    Ma, Mk, Mh = (met["M_ankle"] / nlegs, met["M_knee"] / nlegs, met["M_hip"] / nlegs)
    Fax = met["F_peak"] / nlegs
    segs = [("tarso", met["seg_len"][0], Ma),
            ("tibio", met["seg_len"][1], max(Ma, Mk)),
            ("femur", met["seg_len"][2], max(Mk, Mh))]
    out, total, over = [], 0.0, False
    for name, L, Mb in segs:
        D, mass, gov = size_segment(Mb, Fax, L, M, sf, D_MIN)
        dmax = D_MAX_RATIO * L
        if D > dmax:
            over = True; gov = "OVER-SLENDERNESS"
        out.append(dict(seg=name, L_mm=L * 1e3, M_Nm=Mb, D_mm=D * 1e3,
                        D_max_mm=dmax * 1e3, mass_g=mass * 1e3, governs=gov))
        total += mass
    return out, total * nlegs, over


def eval_v2(x7, m, v0, kc, zeta_c=0.15, mat=MAT_DEFAULT, npass=2, base=None,
            legacy_kc=False, legacy_segmass=False, legacy_cc=False,
            keep_history=False):
    """完整 v2 评价:落震 → 结构定尺 → 质量回代重算 → 可行性所需的全部量。

    两遍定点:第一遍用 2%·m 的杆件质量猜测跑出力矩,定尺得到真实杆件质量,
    第二遍带真实质量重跑。npass=1 可关掉第二遍;`dpeak_pass_pct` 记录两遍差多少,
    用来实测"第二遍值不值",而不是拍脑袋。
    """
    base = dict(SCEN_BIRD_X if base is None else base)
    scen = {**base, "m": float(m), "v0": float(v0),
            "kc": float(kc), "zeta_c": float(zeta_c)}
    # legacy_* 用于通道归因:单独把某一处退回 v1 的写法,看不变性回来多少
    if legacy_kc:
        scen["kc"] = 4000.0 * float(m) * scen["g"]
        scen["legacy_cc"] = True
    if legacy_cc:
        scen["legacy_cc"] = True       # 刚度已绝对化,只把阻尼退回 v1 的 0.01·kc
    if legacy_segmass:
        npass = 1
    segm, res, hist = None, None, []
    for it in range(max(1, npass)):
        s = size_x_v2(scen, x7,
                      seg_mass=([0.05 * float(m)] * 3 if legacy_segmass else segm))
        # 只在最后一遍留时程:前一遍用的是猜测杆件质量,不是最终结果
        s["keep_history"] = bool(keep_history and it == max(1, npass) - 1)
        met = exu_eval_v2(x7, s)
        if met.get("fail"):
            return dict(met)
        rows, leg_mass, over = size_structure(met, mat=mat)
        hist.append(met["peak_a"])
        segm = [r["mass_g"] / 1e3 for r in rows]      # 单腿每段质量,回代
        res = (met, rows, leg_mass, over)
    met, rows, leg_mass, over = res
    frac = leg_mass / float(m)
    out = dict(met)
    out.update(leg_mass_kg=leg_mass, mass_frac=frac,
               sink_mm=1e3 * met.get("sink", 0.0),
               leg_stroke_mm=1e3 * met.get("leg_stroke", met["stroke"]),
               struct_over=bool(over), mass_over=bool(frac > MASS_FRAC_CAP),
               D_mm=[r["D_mm"] for r in rows],
               D_max_mm=[r["D_max_mm"] for r in rows],
               governs=[r["governs"] for r in rows],
               dpeak_pass_pct=(100.0 * abs(hist[-1] - hist[0]) / max(hist[0], 1e-9)
                               if len(hist) > 1 else 0.0))
    return out


def feasible_v2(r, gcap, smax):
    """v2 可行性。返回 (是否可行, 违反的判据列表)。

    **返回全部违反项而不是第一个** —— 只报第一个会让统计带上检查顺序的伪影:
    低质量端因为 g_cap 先失效,把它们的结构状态整个遮住,看上去像
    "高质量端才出现结构失效",其实只是高质量端峰值低、才轮得到结构判据被检查。
    实测教训,见《方案》实施记录 ④。
    """
    if r is None:
        return False, ["none"]
    if r.get("fail"):
        return False, [r["fail"]]
    if not np.isfinite(r.get("peak_a", np.nan)):
        return False, ["nonfinite"]
    bad = []
    if r["peak_a"] > gcap:
        bad.append("gcap")
    if r.get("leg_stroke", r["stroke"]) > smax:
        bad.append("smax")
    if r["struct_over"]:
        bad.append("slenderness")     # 应力/屈曲要求的管径超过几何上限
    if r["mass_over"]:
        bad.append("massbudget")
    return (not bad), (bad or ["ok"])

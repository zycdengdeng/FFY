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

# 足端摩擦系数随地面刚度变化。**这四个锚点是工程估计，不是实测**，
# 和 κ、τ 一样属于"本工作设定"，写论文时必须标出来：
#   混凝土 2e6 → 0.35（打印尼龙对混凝土的手册值 0.30–0.40）
#   沥青   8e5 → 0.40
#   草地   2e5 → 0.49（剪切主导，比硬地高）
#   湿沙   5e4 → 0.60
# 拟合式 mu = 2.916·k_c^(-0.1461)，夹到 [0.30, 0.65]。
# v2.3 及以前统一用 0.5，且因为机体被竖直滑轨锁住，这个值几乎不起作用。
MU_LEGACY = 0.5


def mu_of_kc(kc):
    """由地面刚度导出足端摩擦系数。见上面的出处说明 —— 是估计不是实测。"""
    return float(np.clip(2.916 * float(kc) ** (-0.1461), 0.30, 0.65))
ZETA_C_RANGE = (0.05, 0.35)

# ---------------------------------------------------------------- 结构(绝对量)
MAT_DEFAULT = "cfnylon"
SF = 2.0                 # 安全系数
D_MAX_RATIO = 0.25       # 外径 / 段长 上限:超过即判不可行(杆件假设失效+干涉)
D_MIN = 0.004            # 最小可制造外径 4mm(壁厚 0.4mm)
NLEGS = 2                # 双腿分担
# ---- v2.5 新增判据的默认参数（见《实验计划 v2.5》E24 / E26） ----
REB_CAP = 0.05           # 回弹软闸：回能比 = rebound / (v0^2/2g) 的上限
                         # 依据《回弹判据调研》v3；E24 实测按此打掉现有可行样本的 30.2%
MU_SF = 1.0              # 足端打滑判据的安全系数：要求 mu_demand <= MU_SF * mu_ground
MASS_FRAC_CAP = 0.06     # 腿总质量 / 机体质量 上限(6%,航空口径的结构质量预算)
# 足端等效半径按跗跖长缩放(蹼足)。v1 固定 8 mm,与 33–121 mm 的跗跖长不自洽,
# 且在软介质上会让侵入深度超过球半径,使球-面罚接触模型失效(被误判成"腿塌了")。
ZENER_SUPPORTED = True   # P7:关节 Zener 化已实现(见 exu_eval_v2)
FOOT_RATIO = 0.20        # v2.2 及以前:r_foot = 0.20 × L1(足端与腿长绑定)

# ---------------------------------------------------------------- 足端定尺(v2.3)
# 问题:把 r_foot 绑在 L1 上,使"腿长"与"接地面积"成为同一个变量。
# 于是四臂消融里,低 b 的臂(none:b=0,腿长不随质量涨)自动获得一只小脚,
# 在软地面上侵入超限 → deep_sink → 模型失效。**这个劣势是绑定造成的,
# 与被检验的异速律假设无关**,是消融的混杂因素。
#
# 解法:让 r_foot 由 (质量, 地面) 派生,四臂在同一工况下拿到完全一样的脚。
# 定尺依据选"保持接触模型有效",而不是岩土承载力 —— 理由见下。
#
# 为什么不用承载力公式:deep_sink 是**模型有效性**失效,不是物理失效。
# 而且我们这个尺度上承载力本身无法可靠取值:建筑规范的容许承载力
# (IBC 1806 / CABO:砂 3000 psf ≈ 144 kPa)是给**米级**基础的;
# Terzaghi 圆形基础 q_ult = 0.3·γ·B·N_γ 对 B = 30 mm 的无黏聚力砂只给约 3 kPa
# —— 相差两个数量级。在没有小尺寸压入实测数据之前,任何 q_allow 取值都是编的。
# 所以本实现只声明它做的事:**把脚做大到接触模型在设计载荷下仍然成立**。
GCAP_REF = 10.0 * 9.81   # 足端定尺用的名义过载(m/s²);与验收 g_cap 解耦,定尺时用固定值
FOOT_SF = 3.0            # 侵入裕度。**实测标定**,不是拍的:
                         # 用 GCAP_REF 估的是设计载荷下的静态侵入,而随机探针里
                         # 有大量峰值远超 10g 的坏设计,动态侵入约 3 倍于静态估计。
                         # SF=1.5 时草地 12kg 给出 10.0mm(撞下限),比旧口径的
                         # 23.7mm 还小,废题率反而升到 33–83%(见 P8 首轮);
                         # SF=3.0 给 19.6mm,与旧口径同量级。
FOOT_R_MIN = 0.012       # 最小足端半径(m):可制造 + 常识下限
FOOT_R_MAX = 0.060       # 最大足端半径(m):绝对上限,**不与 L1 挂钩**,否则混杂会从这里回来


def foot_radius(mode, l1, m, kc, nlegs=None):
    """足端等效半径。
    mode="leg"     : r = 0.20·L1        —— v2.2 及以前的口径,保留以复现旧结果
    mode="bearing" : r 由 (m, kc) 派生  —— v2.3;四臂在同工况下完全一致
    """
    if mode != "bearing":
        return FOOT_RATIO * l1
    n = NLEGS if nlegs is None else nlegs
    F = m * GCAP_REF / n                    # 设计载荷(单腿)
    delta = F / float(kc)                   # 该载荷下的侵入深度
    r = FOOT_SF * delta / 0.9               # 使 侵入 ≤ 0.9·r 成立,并留裕度
    return float(np.clip(r, FOOT_R_MIN, FOOT_R_MAX))
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
    s["v_x"] = float(scen.get("v_x", 0.0))       # 水平触地速度(m/s)，v2.5 新增
    # mu 的取法：显式给了就用给的；否则 v2.5 由地面导出，legacy 路径保持 0.5
    if "mu" not in scen or scen.get("mu") is None:
        s["mu"] = mu_of_kc(s["kc"]) if scen.get("mu_from_ground") else MU_LEGACY
    # planar=True: 机体在 x-z 平面内自由平动，水平力必须由足端摩擦承担（v2.5 的要害改动）
    # planar=False: 保留 v2.3 的竖直滑轨约束，水平力由约束反力免费承担（回归测试用）
    s["planar"] = bool(scen.get("planar", False))
    s["seg_mass"] = (list(seg_mass) if seg_mass is not None
                     else [SEG_FRAC_GUESS * scen["m"]] * 3)
    s["seg_len"] = [l1, r2 * l1, r3 * l1]
    s["r_foot"] = foot_radius(scen.get("foot_mode", "leg"), l1,
                              scen["m"], s["kc"])
    s["gap0"] = 0.3 * s["r_foot"]          # 初始离地间隙随足端一起缩放
    if len(xv) >= 9:                       # v2.1:姿态(度)是设计向量的第 8/9 维
        s["thetaA"] = np.radians(xv[7]); s["thetaK"] = np.radians(xv[8])
    if len(xv) >= 10:                      # v2.5:跗跖骨倾角 q1_0(度)是第 10 维
        # 原来它是 SCEN_BIRD_X 里的常数 50°。E26 实测它在生物学跨度内
        # 让峰值过载变化 69.7%(中位) —— 一阶因素,不能冻结。
        s["q1_0"] = np.radians(float(xv[9]))
    return s


def exu_eval_v2(x7, s):
    """与 v1 同拓扑,但每段质量独立给定(结构定尺导出),接触参数为绝对量。"""
    L1, r2, r3 = float(x7[0]), float(x7[1]), float(x7[2])
    l1 = L1 / 1000.0; l2 = r2 * l1; l3 = r3 * l1
    m, g, v0 = s["m"], s["g"], s["v0"]
    vx = float(s.get("v_x", 0.0))
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
                                   initialVelocity=[vx, 0, -v0],
                                   inertia=inertia, gravity=[0, 0, -g])
    tarso = rod("tarso", Fp, A, a1, m1)
    tibio = rod("tibio", A, K, a2, m2)
    femur = rod("femur", K, H, a3, m3)
    body = mbs.CreateRigidBody(name="payload", referencePosition=list(H),
                               initialVelocity=[vx, 0, -v0],
                               inertia=InertiaSphere(mass=m, radius=0.12),
                               gravity=[0, 0, -g])
    if s["planar"]:
        # 面内自由平动：放开 x 与 z，锁死 y 与全部转动。
        # 这一放开是 v2.5 的要害 —— 只有机体能水平走，足端摩擦才真正约束姿态；
        # v2.3 里水平力全被竖直滑轨的约束反力吃掉了，倾斜姿态是"免费"的。
        mbs.CreateGenericJoint(bodyNumbers=[ground, body], position=list(H),
                               constrainedAxes=[0, 1, 0, 1, 1, 1])
    else:
        mbs.CreatePrismaticJoint(bodyNumbers=[ground, body], position=list(H),
                                 axis=[0, 0, 1])
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
    # 三根杆的质心加速度：与机体一起做动量平衡，反推地面反力的水平/竖直分量。
    # 不直接读接触对象的力 —— 那个接口在不同 exudyn 版本里名字不一样，动量平衡更稳。
    sAccR = [mbs.AddSensor(SensorBody(bodyNumber=b, storeInternal=True,
                                      outputVariableType=exu.OutputVariableType.Acceleration))
             for b in (femur, tibio, tarso)]
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
    met.update(_lateral(mbs, acc, pos, footxyz, sAccR, s, m, g, v0, h))
    met["mu_used"] = float(s["mu"])
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


def _lateral(mbs, acc, pos, footxyz, sAccR, s, m, g, v0, h):
    """v2.5 新增的横向量。全部由已有传感器算出，不额外增加求解成本。

    地面反力用**整机动量平衡**反推，而不是去读接触对象的力：
        F_x = Σ mᵢ·a_x,ᵢ                 （x 方向没有其它外力：重力竖直，
                                          planar 约束只锁 y 与转动，不出 x 反力）
        F_z = Σ mᵢ·a_z,ᵢ + M_total·g
    planar=False 时竖直滑轨会出 x 反力，这时 F_x 不是地面给的，`mu_demand` 无意义，
    所以只在 planar=True 时报它。
    """
    ax, az = acc[:, 1], acc[:, 3]
    a_res = float(np.max(np.hypot(ax, az)))
    out = dict(a_res=a_res, peak_ax=float(np.max(np.abs(ax))))

    mr = list(s["seg_mass"])                       # 单腿三段
    M_tot = float(m) + float(np.sum(mr))
    Fx = float(m) * ax
    Fz = float(m) * az
    for j, sj in enumerate(sAccR):
        a = mbs.GetSensorStoredData(sj)
        n = min(len(a), len(ax))
        Fx[:n] += mr[j] * a[:n, 1]
        Fz[:n] += mr[j] * a[:n, 3]
    Fz = Fz + M_tot * g                            # 地面反力的竖直分量

    on = Fz > 0.2 * M_tot * g                      # 接触窗口
    if on.any():
        mu_dem = float(np.max(np.abs(Fx[on]) / np.maximum(Fz[on], 1e-9)))
        fx = footxyz[:len(on)][on, 0]
        slip = float(np.max(fx) - np.min(fx)) if len(fx) else 0.0
    else:
        mu_dem, slip = float("nan"), 0.0
    out.update(mu_demand=mu_dem if s.get("planar") else float("nan"),
               slip=slip, contact_frac=float(on.mean()))

    x = pos[:, 1]
    out["x_drift"] = float(x[-1] - x[0])

    # 数值健康体检：末态总能不应高于初态（接触与摩擦只该耗散）
    vz = np.gradient(pos[:, 3], h); vx_ = np.gradient(x, h)
    e0 = 0.5 * m * (vx_[0] ** 2 + vz[0] ** 2) + m * g * pos[0, 3]
    e1 = 0.5 * m * (vx_[-1] ** 2 + vz[-1] ** 2) + m * g * pos[-1, 3]
    ke0 = max(0.5 * m * (vx_[0] ** 2 + vz[0] ** 2), 1e-9)
    out["e_gain"] = float((e1 - e0) / ke0)
    return out


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
            keep_history=False, v_x=None, planar=None, mu=None):
    """完整 v2 评价:落震 → 结构定尺 → 质量回代重算 → 可行性所需的全部量。

    两遍定点:第一遍用 2%·m 的杆件质量猜测跑出力矩,定尺得到真实杆件质量,
    第二遍带真实质量重跑。npass=1 可关掉第二遍;`dpeak_pass_pct` 记录两遍差多少,
    用来实测"第二遍值不值",而不是拍脑袋。
    """
    base = dict(SCEN_BIRD_X if base is None else base)
    scen = {**base, "m": float(m), "v0": float(v0),
            "kc": float(kc), "zeta_c": float(zeta_c)}
    # v2.5：水平触地速度、面内自由平动、足端摩擦。三个都不给就完全退回 v2.3 行为。
    if v_x is not None:
        scen["v_x"] = float(v_x)
    if planar is not None:
        scen["planar"] = bool(planar)
    if mu is not None:
        scen["mu"] = float(mu)
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
    out.update(m=float(m), v0=float(v0), kc=float(kc),      # 回弹判据要用 v0，必须落到输出里
               v_x=scen.get("v_x", 0.0), planar=bool(scen.get("planar", False)),
               mu_ground=float(met.get("mu_used", scen.get("mu", MU_LEGACY))),
               leg_mass_kg=leg_mass, mass_frac=frac,
               sink_mm=1e3 * met.get("sink", 0.0),
               leg_stroke_mm=1e3 * met.get("leg_stroke", met["stroke"]),
               struct_over=bool(over), mass_over=bool(frac > MASS_FRAC_CAP),
               D_mm=[r["D_mm"] for r in rows],
               D_max_mm=[r["D_max_mm"] for r in rows],
               governs=[r["governs"] for r in rows],
               dpeak_pass_pct=(100.0 * abs(hist[-1] - hist[0]) / max(hist[0], 1e-9)
                               if len(hist) > 1 else 0.0))
    return out


def feasible_v2(r, gcap, smax, reb_cap=REB_CAP, mu_sf=MU_SF, use_a_res=None):
    """v2.5 可行性。返回 (是否可行, 违反的判据列表)。

    **返回全部违反项而不是第一个** —— 只报第一个会让统计带上检查顺序的伪影:
    低质量端因为 g_cap 先失效,把它们的结构状态整个遮住,看上去像
    "高质量端才出现结构失效",其实只是高质量端峰值低、才轮得到结构判据被检查。
    实测教训,见《方案》实施记录 ④。

    v2.5 相对 v2.3 的三处改动（依据见《实验计划 v2.5》）：

    ① **过载判据改判合加速度 a_res = max‖(a_x, a_z)‖**，而不是只判竖直分量。
       有水平速度以后竖直分量不再代表全部惯性载荷。
       没有 `a_res`（老结果重打分）时自动退回 `peak_a`，所以旧数据照样能过。

    ② **回弹软闸**（E24）：回能比 = rebound / (v0²/2g) ≤ reb_cap，默认 5%。
       《回弹判据调研》v3 定的判据，一直算了但从来没挂上。
       E24 实测：现有"可行"样本里 30.2% 会弹，进训练集的前沿点里 23.7% 会弹
       —— 也就是说 v2.3 的训练目标里混了近四分之一会弹起来的设计。
       足端离地 `n_bounce` 只记录不判死，同样依据 v3。

    ③ **足端打滑硬闸**（E26）：mu_demand ≤ mu_sf · mu_ground。
       E26 发现峰值过载的收益几乎全部落在需要 μ ≥ 0.47 的倾斜姿态上，
       而打印尼龙对混凝土只有 0.30–0.40。没有这条闸，q1_0 一旦升为设计变量，
       生成器会一路跑到 28°，产出一批在真实地面上会打滑的设计。
       只有 planar=True 的结果才有 mu_demand（竖直滑轨下水平力由约束反力承担，
       那个数没有物理意义），所以 NaN 时这条闸自动跳过。
    """
    if r is None:
        return False, ["none"]
    if r.get("fail"):
        return False, [r["fail"]]
    if not np.isfinite(r.get("peak_a", np.nan)):
        return False, ["nonfinite"]
    bad = []

    # ① 过载：有 a_res 就用 a_res，没有就退回 peak_a
    a_use = r.get("a_res")
    if use_a_res is False or a_use is None or not np.isfinite(a_use):
        a_use = r["peak_a"]
    if a_use > gcap:
        bad.append("gcap")

    if r.get("leg_stroke", r["stroke"]) > smax:
        bad.append("smax")
    if r["struct_over"]:
        bad.append("slenderness")     # 应力/屈曲要求的管径超过几何上限
    if r["mass_over"]:
        bad.append("massbudget")

    # ② 回弹软闸
    v0 = r.get("v0", None)
    if v0 is None:
        v0 = r.get("hist", {}).get("v0") if isinstance(r.get("hist"), dict) else None
    reb = r.get("rebound", None)
    if reb is not None and np.isfinite(reb) and v0:
        h0 = float(v0) ** 2 / (2.0 * 9.81)
        if h0 > 1e-12 and reb / h0 > reb_cap:
            bad.append("rebound")

    # ③ 足端打滑
    md = r.get("mu_demand", None)
    mg = r.get("mu_ground", None)
    if md is not None and np.isfinite(md) and mg:
        if md > mu_sf * float(mg):
            bad.append("slip")

    return (not bad), (bad or ["ok"])


def reb_ratio(r):
    """回能比。单独拎出来，方便重打分脚本和判据共用同一套定义。"""
    v0 = r.get("v0")
    reb = r.get("rebound")
    if not v0 or reb is None or not np.isfinite(reb):
        return float("nan")
    return float(reb) / max(float(v0) ** 2 / (2.0 * 9.81), 1e-12)

"""Exudyn 多体高保真模型:三段连杆起落架落震(真实接触)。

拓扑(落震试验台标准形态):
  地面 ←棱柱副(竖直滑轨)← 机身质量 ←髋转动副+扭簧← 股骨杆 ←膝副+扭簧←
  胫跗杆 ←踝副+扭簧← 跖骨杆 → 足端球 ↔ 地面四边形 penalty 接触(摩擦/恢复系数)

相比 2-DOF 解析模型多的物理:分布段惯量、精确多体运动学、单边接触
(可弹跳/滑移/再触地)、库仑摩擦。纯 CPU,单次 ~0.5s。

用法:from hf_exudyn import exu_eval;  exu_eval((L1_mm, r2, r3), scen_dict)
"""
from __future__ import annotations
import numpy as np
import exudyn as exu
from exudyn.utilities import *   # noqa: F401,F403

SCEN_X = dict(m=200.0, v0=3.0, g=9.81,
              k_ankle=4.5e3, k_knee=4.5e3, k_hip=2.0e4,
              c_ankle=4.0e2, c_knee=4.0e2, c_hip=8.0e2,
              q1_0=np.radians(50.0), thetaA=np.radians(120.0), thetaK=np.radians(90.0),
              seg_mass_frac=0.05,          # 每段杆质量 = 5% 机身质量
              kc=6.0e5, cc=1.0e4, mu=0.5,  # 接触刚度/阻尼/摩擦
              r_foot=0.03, gap0=0.005, T=0.8, h=2e-4)


def _rotY(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-c * 0 - s, 0, c]])  # R@[1,0,0]=[c,0,-s]? fix below


def rotY(a):
    """使局部 x 轴指向平面内与水平夹角 a(x-z 平面,z 向上)。"""
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0., -s], [0., 1., 0.], [s, 0., c]])


def exu_eval(x, s=SCEN_X):
    L1, r2, r3 = x
    l1 = L1 / 1000.0; l2 = r2 * l1; l3 = r3 * l1
    m, g, v0 = s["m"], s["g"], s["v0"]
    ms = s["seg_mass_frac"] * m
    # 初始姿态角(与解析模型一致的之字折叠)
    a1 = s["q1_0"]                                # 跖骨仰角
    a2 = a1 + (np.pi - s["thetaA"])               # 胫跗
    a3 = a2 - (np.pi - s["thetaK"])               # 股骨(反折)
    d = lambda a: np.array([np.cos(a), 0., np.sin(a)])
    F = np.array([0., 0., s["gap0"] + s["r_foot"]])          # 足端球心
    A = F + l1 * d(a1); K = A + l2 * d(a2); H = K + l3 * d(a3)

    SC = exu.SystemContainer(); mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])
    def rod(name, P0, P1, ang, mass):
        L = np.linalg.norm(P1 - P0); com = 0.5 * (P0 + P1)
        inertia = InertiaCuboid(density=mass / (L * 0.02 * 0.02), sideLengths=[L, 0.02, 0.02])
        return mbs.CreateRigidBody(name=name, referencePosition=list(com),
                                   referenceRotationMatrix=rotY(ang),
                                   initialVelocity=[0, 0, -v0],
                                   inertia=inertia, gravity=[0, 0, -g])
    tarso = rod("tarso", F, A, a1, ms)
    tibio = rod("tibio", A, K, a2, ms)
    femur = rod("femur", K, H, a3, ms)
    body = mbs.CreateRigidBody(name="payload", referencePosition=list(H),
                               initialVelocity=[0, 0, -v0],
                               inertia=InertiaSphere(mass=m, radius=0.12), gravity=[0, 0, -g])
    # 关节
    mbs.CreatePrismaticJoint(bodyNumbers=[ground, body], position=list(H), axis=[0, 0, 1])
    for (b0, b1, P, k, c) in [(body, femur, H, s["k_hip"], s["c_hip"]),
                              (femur, tibio, K, s["k_knee"], s["c_knee"]),
                              (tibio, tarso, A, s["k_ankle"], s["c_ankle"])]:
        mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1], position=list(P), axis=[0, 1, 0])
        mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0, b1], position=list(P),
                                        axis=[0, 1, 0], stiffness=k, damping=c)
    # 足端接触(球 vs 地面大四边形)
    quad = [[-3., -3., 0.], [3., -3., 0.], [3., 3., 0.], [-3., 3., 0.]]
    mbs.CreateSphereQuadContact(bodyNumbers=[tarso, ground],
                                localPosition0=[-0.5 * l1, 0., 0.],
                                radiusSphere=s["r_foot"],
                                quadPoints=quad,
                                contactStiffness=s["kc"], contactDamping=s["cc"],
                                dynamicFriction=s["mu"], frictionProportionalZone=1e-3)
    sAcc = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
                                    outputVariableType=exu.OutputVariableType.Acceleration))
    sPos = mbs.AddSensor(SensorBody(bodyNumber=body, storeInternal=True,
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
        mbs.SolveDynamic(ss)
    except Exception:
        return dict(peak_a=np.nan, stroke=np.nan)
    acc = mbs.GetSensorStoredData(sAcc); pos = mbs.GetSensorStoredData(sPos)
    az = acc[:, 3]; z = pos[:, 3]
    if not np.all(np.isfinite(az)):
        return dict(peak_a=np.nan, stroke=np.nan)
    peak = float(np.max(np.abs(az)))
    stroke = float(z[0] - np.min(z))
    return dict(peak_a=peak, stroke=stroke)


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = exu_eval((368.0, 2.11, 0.97))
    print(f"smoke: peak_a={r['peak_a']:.1f} m/s2 ({r['peak_a']/9.81:.1f}g) stroke={r['stroke']:.3f} m  [{time.time()-t0:.1f}s]")

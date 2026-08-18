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

SCEN_BIRD_X = dict(m=5.0, v0=1.2, g=9.81,
              k_ankle=35.0, k_knee=35.0, k_hip=160.0,
              c_ankle=1.2, c_knee=1.2, c_hip=5.0,
              q1_0=np.radians(50.0), thetaA=np.radians(120.0), thetaK=np.radians(90.0),
              seg_mass_frac=0.05,
              kc=8.0e4, cc=400.0, mu=0.5,
              r_foot=0.008, gap0=0.003, T=1.0, h=2e-4)

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


import contextlib
import os as _os


@contextlib.contextmanager
def silence_solver():
    """屏蔽 Exudyn C++ 内核直接写文件描述符的横幅(DYNAMIC SOLVER FAILED 等)。

    这类输出不经过 Python 的 sys.stdout,contextlib.redirect_stdout 拦不住,
    必须在 fd 层重定向。求解失败本就由 try/except 接住并记为 NaN,
    横幅只是噪声;设 EXU_VERBOSE=1 可恢复输出以便调试。
    """
    if _os.environ.get("EXU_VERBOSE"):
        yield
        return
    devnull = _os.open(_os.devnull, _os.O_WRONLY)
    saved = (_os.dup(1), _os.dup(2))
    try:
        _os.dup2(devnull, 1); _os.dup2(devnull, 2)
        yield
    finally:
        _os.dup2(saved[0], 1); _os.dup2(saved[1], 2)
        _os.close(devnull); _os.close(saved[0]); _os.close(saved[1])


NAN_METRICS = dict(peak_a=np.nan, stroke=np.nan, eta=np.nan, cfe=np.nan,
                   peak_jerk=np.nan, E_abs=np.nan, F_peak=np.nan,
                   rebound=np.nan, n_bounce=np.nan, t_settle=np.nan)


def _metrics(t, z, az, m, g, v0):
    """从机身时程提取落震多指标(文献标准指标,白话注释):

    - eta 缓冲效率: 吸收能量 / (峰值力×最大压缩行程)。力-行程曲线越接近
      "矩形"(全程恒力)越接近 1;油气式支柱典型 0.8-0.9。
    - cfe 压溃力效率: 压缩段平均力/峰值力,耐撞性文献同思想指标。
    - peak_jerk: 加速度变化率峰值(冲击"突兀感",舒适性指标)。
    - E_abs: 压缩段吸收能量 ∫F ds。
    - rebound: 最大压缩后机身回弹超过触地高度的高度(弹跳倾向)。
    - n_bounce: 足-地分离次数(bounced landing 事故模式计数)。
    - t_settle: 触地→速度降到 5%v0 以内的耗时(镇定快慢)。
    """
    h = t[1] - t[0]
    F = m * (az + g)                       # 腿传给机身的竖直力(自由落体段≈0)
    thr = 0.02 * m * g
    idx = np.where(F > thr)[0]
    if len(idx) == 0:
        return {}
    i0 = int(idx[0])                       # 触地时刻
    imin = int(np.argmin(z))               # 最大压缩时刻
    if imin <= i0:
        imin = min(i0 + 1, len(z) - 1)
    zc, Fc = z[i0:imin + 1], F[i0:imin + 1]
    s_max = float(z[i0] - z[imin])
    Fmax = float(np.max(Fc))
    trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))
    E = float(trapz(Fc, x=-zc))            # ∫F d(-z), 下压方向为正
    eta = E / (Fmax * s_max) if Fmax > 0 and s_max > 1e-6 else np.nan
    cfe = float(np.mean(Fc) / Fmax) if Fmax > 0 else np.nan
    w = max(3, int(0.005 / h))             # 5ms 滑窗平滑再差分,防数值毛刺
    az_s = np.convolve(az, np.ones(w) / w, mode="same")
    jerk = float(np.max(np.abs(np.diff(az_s) / h)))
    reb = float(max(0.0, np.max(z[imin:]) - z[i0]))
    # 离地判定加 5ms 时间门槛:真实弹跳腾空为几十 ms 量级,
    # 回程弹簧卸载导致的亚毫秒级力过零属数值抖动,不计(判据调研 v2,2026-08-17)
    off = F[i0:] < thr
    wmin = max(1, int(0.005 / h))
    n_bounce = 0
    run = 0
    for o in off:
        run = run + 1 if o else 0
        if run == wmin:                      # 每个 ≥5ms 的失载段计一次
            n_bounce += 1
    v = np.gradient(z, h)
    hot = np.where(np.abs(v) > 0.05 * abs(v0))[0]
    t_settle = float(t[hot[-1]] - t[i0]) if len(hot) else 0.0
    return dict(eta=eta, cfe=cfe, peak_jerk=jerk, E_abs=E, F_peak=Fmax,
                rebound=reb, n_bounce=n_bounce, t_settle=t_settle)


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
        with silence_solver():
            mbs.SolveDynamic(ss)
    except Exception:
        return dict(NAN_METRICS)
    acc = mbs.GetSensorStoredData(sAcc); pos = mbs.GetSensorStoredData(sPos)
    t = acc[:, 0]; az = acc[:, 3]; z = pos[:, 3]
    if not np.all(np.isfinite(az)):
        return dict(NAN_METRICS)
    peak = float(np.max(np.abs(az)))
    stroke = float(z[0] - np.min(z))
    Lleg = l1 + l2 + l3
    if stroke > 0.6 * Lleg:                      # 塌陷/屈曲 → 不可行设计
        return dict(NAN_METRICS)
    out = dict(NAN_METRICS)
    out.update(peak_a=peak, stroke=stroke)
    out.update(_metrics(t, z, az, m, g, s["v0"]))
    return out


if __name__ == "__main__":
    import time
    t0 = time.time()
    r = exu_eval((368.0, 2.11, 0.97))
    print(f"smoke: peak_a={r['peak_a']:.1f} m/s2 ({r['peak_a']/9.81:.1f}g) stroke={r['stroke']:.3f} m  [{time.time()-t0:.1f}s]")

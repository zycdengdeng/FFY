# -*- coding: utf-8 -*-
"""v4 符号台架 —— 任何用户力/力矩函数上线前必须过的闸。

立项背景（2026-09-19 发现，2026-09-22 制度化）：
  M1–M3 三周里，`springTorqueUserFunction` 里写的「刹车」其实是**电机**。
  症状早就摆在眼前——刹车臂末速大于初速——但没有任何机制强迫去看它。
  M3 的整张相图因此作废。

  根因是 exudyn 的一个反直觉约定：
      springTorqueUserFunction 的返回值，代入内置公式 k·Δ + d·ω 的**位置**，
      再随内置的**负反馈符号**一起施加。
      ⇒ 阻力必须返回  +|τ|·sign(ω)；写成负号就变成了正反馈（电机）。

本台架用**解析解**钉死这个约定，并把「写错符号会怎样」也做成一个必过用例——
以后谁改了刹车律，跑一遍就知道自己是在刹车还是在加速。

用法：
    python bench_v4.py            # 全部用例
    python bench_v4.py -v         # 打印每步数值
"""
from __future__ import annotations

import sys
import numpy as np
import exudyn as exu
from exudyn.utilities import InertiaCuboid
from exudyn.itemInterface import *          # noqa: F401,F403
import exudyn.graphics as graphics          # noqa: F401

OV = exu.OutputVariableType

# 台架常量：一个绕 y 轴自转的圆盘，除被测力矩外不受任何外力
I_DISC = 6.6667e-3          # kg·m²，见 _make_disc
OMEGA0 = 100.0              # rad/s 初始角速度
T_END = 0.02                # s
H_STEP = 1e-5               # s


def _silence():
    exu.config.suppressWarnings = True


def _make_disc(torque_fn=None, damping=0.0, stiffness=0.0):
    """地面 + 单个可自转圆盘，绕 y 轴的旋转副 + 一个扭簧阻尼器。

    torque_fn 为 None 时用内置 damping；否则挂 springTorqueUserFunction。
    返回 (mbs, sensor_omega)。
    """
    SC = exu.SystemContainer()
    mbs = SC.AddSystem()
    ground = mbs.CreateGround(referencePosition=[0, 0, 0])
    # 立方体惯量：边长 a 的正方体绕中心轴 I = m(a²+a²)/12
    a, m_d = 0.2, 1.0
    inertia = InertiaCuboid(density=m_d / (a ** 3), sideLengths=[a, a, a])
    disc = mbs.CreateRigidBody(inertia=inertia,
                               referencePosition=[0, 0, 0],
                               initialAngularVelocity=[0, OMEGA0, 0],
                               gravity=[0, 0, 0])          # 台架不加重力
    mbs.CreateRevoluteJoint(bodyNumbers=[ground, disc],
                            position=[0, 0, 0], axis=[0, 1, 0])
    con = mbs.CreateTorsionalSpringDamper(bodyNumbers=[ground, disc],
                                          position=[0, 0, 0], axis=[0, 1, 0],
                                          stiffness=stiffness, damping=damping)
    if torque_fn is not None:
        mbs.SetObjectParameter(con, "springTorqueUserFunction", torque_fn)
    sOm = mbs.AddSensor(SensorBody(bodyNumber=disc, storeInternal=True,
                                   outputVariableType=OV.AngularVelocity))
    mbs.Assemble()
    return SC, mbs, sOm


def _run(SC, mbs, sOm, t_end=T_END, h=H_STEP):
    ss = exu.SimulationSettings()
    ss.timeIntegration.endTime = t_end
    ss.timeIntegration.numberOfSteps = int(t_end / h)
    ss.timeIntegration.generalizedAlpha.spectralRadius = 0.7
    ss.timeIntegration.verboseMode = 0
    ss.solutionSettings.writeSolutionToFile = False
    ss.solutionSettings.sensorsWritePeriod = h
    mbs.SolveDynamic(ss)
    d = mbs.GetSensorStoredData(sOm)
    return d[:, 0], d[:, 2]          # t, omega_y


# ═══════════════════════════════════════════════════════ 用例
def case_inertia_identity(v=False):
    """0 · 标定圆盘转动惯量：无外力矩时角速度必须严格守恒。"""
    SC, mbs, sOm = _make_disc(damping=0.0)
    t, om = _run(SC, mbs, sOm)
    err = abs(om[-1] - OMEGA0) / OMEGA0
    if v:
        print(f"    ω(0)={om[0]:.4f}  ω(T)={om[-1]:.4f}  相对漂移={err:.2e}")
    return err < 1e-6, f"无力矩下角速度漂移 {err:.2e}"


def case_builtin_damping(v=False):
    """1 · 内置粘性阻尼 vs 解析解 ω(t)=ω0·exp(−c·t/I)。"""
    c = 1e-3
    SC, mbs, sOm = _make_disc(damping=c)
    t, om = _run(SC, mbs, sOm)
    ana = OMEGA0 * np.exp(-c * t[-1] / I_DISC)
    err = abs(om[-1] - ana) / ana
    if v:
        print(f"    仿真 ω(T)={om[-1]:.4f}   解析 {ana:.4f}   相对误差={err:.2e}")
    return err < 2e-3, f"内置阻尼与解析解差 {err:.2%}"


def case_userfn_positive_is_resistive(v=False):
    """2 · **核心约定**：用户函数返回 +c·ω 才是阻力，结果须与内置阻尼一致。"""
    c = 1e-3

    def torque(mbs_, t_, item_, rot, rot_t, k_, d_, off_):
        return +c * rot_t                       # ← 正号 = 阻力

    SC, mbs, sOm = _make_disc(torque_fn=torque)
    t, om = _run(SC, mbs, sOm)
    ana = OMEGA0 * np.exp(-c * t[-1] / I_DISC)
    err = abs(om[-1] - ana) / ana
    if v:
        print(f"    用户函数 +c·ω → ω(T)={om[-1]:.4f}   解析阻尼 {ana:.4f}   误差={err:.2e}")
    return err < 2e-3, f"用户函数 +c·ω 未复现阻尼（差 {err:.2%}）"


def case_userfn_negative_is_motor(v=False):
    """3 · **陷阱用例**：返回 −c·ω 会变成电机（末速 > 初速）。

    这是 M1–M3 的原始事故形态。它必须**稳定地失败**成加速，
    以便任何人改坏符号时立刻看见。
    """
    c = 1e-3

    def torque(mbs_, t_, item_, rot, rot_t, k_, d_, off_):
        return -c * rot_t                       # ← 负号 = 正反馈

    SC, mbs, sOm = _make_disc(torque_fn=torque)
    t, om = _run(SC, mbs, sOm)
    ana_motor = OMEGA0 * np.exp(+c * t[-1] / I_DISC)
    grew = om[-1] > OMEGA0 * 1.0001
    err = abs(om[-1] - ana_motor) / ana_motor
    if v:
        print(f"    用户函数 −c·ω → ω(T)={om[-1]:.4f}（初速 {OMEGA0}）"
              f"  解析电机 {ana_motor:.4f}  误差={err:.2e}")
    return grew and err < 2e-3, "负号未表现为电机 —— 约定可能已随版本改变，须重新标定"


def case_coulomb_brake(v=False):
    """4 · 恒力矩刹车：ω 线性衰减，斜率 = −τ/I。"""
    tau = 0.5

    def torque(mbs_, t_, item_, rot, rot_t, k_, d_, off_):
        return tau * np.tanh(rot_t / 0.5)       # 平滑符号函数，与 physics 同形

    SC, mbs, sOm = _make_disc(torque_fn=torque)
    t, om = _run(SC, mbs, sOm)
    slope = np.polyfit(t, om, 1)[0]
    ana = -tau / I_DISC
    err = abs(slope - ana) / abs(ana)
    if v:
        print(f"    拟合斜率={slope:.2f} rad/s²   解析 −τ/I={ana:.2f}   误差={err:.2e}")
    return err < 5e-3, f"恒力矩刹车斜率偏差 {err:.2%}"


def brake_torque(om, tau_max, om_th, c_rr_N_rw=0.0):
    """v4 刹车律（**唯一定义**，物理模型与台架共用此函数）。

    τ_b(ω) = τ_max·max(0, 1−|ω|/ω_th)·tanh(ω/0.5)  +  C_RR·N·r_w·tanh(ω/0.5)

    反离心：轮转得快（|ω| > ω_th）完全不刹；转慢下来才线性咬合，ω→0 时给满 τ_max。
    第二项是滚动阻力，恒在。返回值为 exudyn 约定下的**阻力**（正号，见用例 2/3）。
    """
    frac = max(0.0, 1.0 - abs(om) / max(om_th, 1e-9))
    sgn = np.tanh(om / 0.5)
    return tau_max * frac * sgn + c_rr_N_rw * sgn


def case_brake_law_shape(v=False):
    """5 · 刹车律形状（纯函数，无仿真）：门槛以上为零、ω→0 给满、符号随 ω 反号。"""
    tm, oth = 2.0, 40.0
    checks = {
        "门槛以上不刹":      abs(brake_torque(60.0, tm, oth)) < 1e-12,
        "恰在门槛处为零":    abs(brake_torque(40.0, tm, oth)) < 1e-12,
        "半门槛给半力矩":    abs(brake_torque(20.0, tm, oth) - tm * 0.5) < 1e-3,
        "ω→0⁺ 趋近满力矩":  abs(brake_torque(2.0, tm, oth) - tm * 0.95) < 5e-2,
        "反向转动力矩反号":  brake_torque(-20.0, tm, oth) < 0,
        "ω=0 时无力矩":      abs(brake_torque(0.0, tm, oth)) < 1e-12,
    }
    if v:
        for k, o in checks.items():
            print(f"    {'OK ' if o else 'BAD'} {k}")
        print(f"    τ(60)={brake_torque(60,tm,oth):.3f}  τ(20)={brake_torque(20,tm,oth):.3f}  "
              f"τ(2)={brake_torque(2,tm,oth):.3f}  τ(−20)={brake_torque(-20,tm,oth):.3f}")
    bad = [k for k, o in checks.items() if not o]
    return not bad, f"刹车律形状不符：{bad}"


def case_brake_below_threshold(v=False):
    """6 · 门槛以下的动力学：刹车须把轮子刹停，且与数值解析积分一致。"""
    tm, oth = 2.0, 40.0
    om0 = 30.0                                   # 起始即在门槛以下

    def torque(mbs_, t_, item_, rot, rot_t, k_, d_, off_):
        return brake_torque(rot_t, tm, oth)

    global OMEGA0
    keep, OMEGA0 = OMEGA0, om0
    try:
        SC, mbs, sOm = _make_disc(torque_fn=torque)
        t, om = _run(SC, mbs, sOm, t_end=0.2, h=1e-5)
    finally:
        OMEGA0 = keep
    # 同一律的独立 RK4 参考积分
    w, dt = om0, 1e-6
    for _ in range(int(0.2 / dt)):
        def f(x): return -brake_torque(x, tm, oth) / I_DISC
        k1 = f(w); k2 = f(w + .5 * dt * k1); k3 = f(w + .5 * dt * k2); k4 = f(w + dt * k3)
        w += dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)
    err = abs(om[-1] - w)
    if v:
        print(f"    exudyn ω(T)={om[-1]:.4f}   独立 RK4 参考={w:.4f}   绝对差={err:.2e}")
    return om[-1] < 1.0 and err < 0.15, \
        f"末速 {om[-1]:.3f}（应 <1），与参考积分差 {err:.3f}（应 <0.15）"


def case_rolling_resistance_alone(v=False):
    """7 · M3 教训量化：门槛以上只有滚阻时，轮子几乎不减速。

    v3 的 C_RR=0.015、单轮法向力 ≈11.6 N、r_w=0.03 ⇒ τ_rr ≈ 5.2e-3 N·m，
    对 I=6.67e-3 只给 0.78 rad/s²。从 100 rad/s 靠滚阻降到门槛 40 需要 ~77 s。
    **所以真实模型里轮速不是靠滚阻掉下来的，是被地面接触点绑在整机速度上**
    （纯滚动 ω = v_x/r_w），整机一减速轮速就跟着掉。
    这条用例把这件事钉死，免得再有人以为滚阻能停住轮子。
    """
    c_rr = 0.015 * 11.6 * 0.03

    def torque(mbs_, t_, item_, rot, rot_t, k_, d_, off_):
        return brake_torque(rot_t, 2.0, 40.0, c_rr_N_rw=c_rr)

    SC, mbs, sOm = _make_disc(torque_fn=torque)
    t, om = _run(SC, mbs, sOm, t_end=0.5, h=1e-4)
    drop = OMEGA0 - om[-1]
    ana = c_rr / I_DISC * 0.5
    t_to_th = (OMEGA0 - 40.0) / (c_rr / I_DISC)
    if v:
        print(f"    0.5 s 内掉速 {drop:.3f} rad/s（解析 {ana:.3f}）；"
              f"靠滚阻降到门槛需 {t_to_th:.1f} s")
    return abs(drop - ana) < 0.02 and t_to_th > 50, \
        f"滚阻衰减 {drop:.3f} vs 解析 {ana:.3f}；到门槛用时 {t_to_th:.1f} s"


CASES = [case_inertia_identity, case_builtin_damping,
         case_userfn_positive_is_resistive, case_userfn_negative_is_motor,
         case_coulomb_brake, case_brake_law_shape,
         case_brake_below_threshold, case_rolling_resistance_alone]


def main(verbose=False):
    _silence()
    print("=" * 74)
    print("v4 符号台架 —— exudyn springTorqueUserFunction 约定标定")
    print("=" * 74)
    npass = 0
    for fn in CASES:
        name = fn.__doc__.strip().splitlines()[0]
        try:
            ok, msg = fn(verbose)
        except Exception as e:
            ok, msg = False, f"{type(e).__name__}: {e}"
        print(f"  [{'通过' if ok else '失败'}] {name}")
        if not ok:
            print(f"         → {msg}")
        npass += bool(ok)
    print("-" * 74)
    print(f"  {npass}/{len(CASES)} 通过")
    if npass == len(CASES):
        print("\n  约定确认：阻力必须返回 +|τ|·sign(ω)。写负号即为电机。")
    return 0 if npass == len(CASES) else 1


if __name__ == "__main__":
    sys.exit(main("-v" in sys.argv))

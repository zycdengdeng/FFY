# -*- coding: utf-8 -*-
"""v4 Phase 1-1 · 脚掌运动学分析（纯几何 + 准静力，无需仿真）。

要回答三个问题，全都是 v3 事后才撞见的坑：

  Q1 机体下沉时，脚掌会不会被腿的几何逼着水平滑移？滑多少？
     —— v3 的教训：Fr=0 纯垂直落地，镜像双腿张开使足端外滑，轮子转到 64.7 rad/s，
        刹车必然咬合、阻止外滑、把腿行程从 110 mm 压到 72 mm、过载从 1.69 g 升到 2.52 g。
        这个耦合是跑完 36k 数据、做完相图、写完 PPT 之后才发现的。

  Q2 由此产生的轮速有多大？会不会越过刹车咬合门槛 ω_th？
     —— τ 盒缩到 [0.1, 1.0] N·m 之后（地面摩擦钳制，见 mass_model），
        这个耦合还剩多少强度？

  Q3 脚掌要多长才不翻？
     —— v4 的抗俯仰机制从「机体轮距」换成「踝扭簧 + 脚掌轮距」，
        必须先知道脚掌轮距的量级，才知道这个构型可不可行。

关键构型差别（v4 vs v3）：
    v3：每条腿一个轮 → 足端姿态自由，接触点运动复杂
    v4：每只脚掌前后两轮 → **两轮同时着地时脚掌被地面锁成水平**，
        脚掌不转，只平移；前后两轮转速相同，都等于脚掌平移速度 / 轮半径。
        这比 v3 干净得多，也更容易分析。

自检 / 运行：  python foot_kinematics_v4.py
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize

G = 9.81

# ── chosen 5 kg 设计（v2.5 定案，v3/v4 沿用作为分析基准）──────────
CHOSEN = dict(
    L1_mm=113.1004,      # 跗跖骨
    r2=2.0272,           # 胫跗骨 / 跗跖骨
    r3=1.2446,           # 股骨 / 跗跖骨
    kap_ankle=1.2371,    # 无量纲扭簧刚度
    kap_knee=4.6829,
    kap_hip=16.2987,
    thetaA_deg=125.6335,  # 着地踝角
    thetaK_deg=129.8789,  # 着地膝角
    q1_0_deg=29.1107,     # 跗跖骨着地倾角
)
M_KG, V0_MPS = 4.7305, 1.3718
R_W_M = 0.030


def geom(des=CHOSEN):
    """三段长度（m）与着地时三段相对水平面的倾角（rad）。

    与 e8_struct.standing_height 同一套角度约定：
        a1 = q1_0                       跗跖骨
        a2 = a1 + (π − θ_踝)            胫跗骨
        a3 = a2 − (π − θ_膝)            股骨
    """
    L1 = des["L1_mm"] / 1e3
    L2, L3 = des["r2"] * L1, des["r3"] * L1
    a1 = np.radians(des["q1_0_deg"])
    a2 = a1 + (np.pi - np.radians(des["thetaA_deg"]))
    a3 = a2 - (np.pi - np.radians(des["thetaK_deg"]))
    return np.array([L1, L2, L3]), np.array([a1, a2, a3])


def stiffness_Nm(des=CHOSEN, m_kg=M_KG):
    """无量纲刚度 → 实际扭簧刚度 k = κ · m·g·L_total（N·m/rad）。

    标定：chosen 5 kg 设计的 κ(踝/膝/髋)=1.237/4.683/16.30 对应
    k = 28/105/365 N·m/rad，比值恒为 22.4 = m·g·(L1+L2+L3)。
    """
    L, _ = geom(des)
    scale = m_kg * G * L.sum()
    return np.array([des["kap_ankle"], des["kap_knee"], des["kap_hip"]]) * scale


def hip_from_angles(L, a):
    """由三段倾角给出髋点相对「趾关节」的位置（m）。"""
    return np.array([np.sum(L * np.cos(a)), np.sum(L * np.sin(a))])


def solve_sink(dz_m, des=CHOSEN, m_kg=M_KG, k_toe_Nm=None, foot_flat=True):
    """准静力：机体下沉 dz 时，三段倾角如何重分布。

    约束：髋点竖直下降 dz；两轮同时着地 ⇒ 脚掌水平（foot_flat）。
    目标：弹性能最小 ½Σ k_j·Δq_j²，Δq_j 为各关节相对着地姿态的转角。

    关节转角（相对着地姿态）：
        趾  Δq0 = Δa1 − Δa_foot        （脚掌水平 ⇒ Δa_foot = 0）
        踝  Δq1 = Δa2 − Δa1
        膝  Δq2 = Δa3 − Δa2
    髋关节转角由机体姿态定，纯垂直落地机体不俯仰 ⇒ 不计入。

    返回 dict：脚掌水平位移（正 = 向前/外滑）、各关节转角、轮速等。
    """
    L, a0 = geom(des)
    k = stiffness_Nm(des, m_kg)                 # [踝, 膝, 髋]
    k_toe = k_toe_Nm if k_toe_Nm is not None else k[0]   # 趾默认与踝同量级
    kk = np.array([k_toe, k[0], k[1]])          # [趾, 踝, 膝]
    p0 = hip_from_angles(L, a0)
    z_target = p0[1] - dz_m

    def unpack(u):
        return a0 + u

    def energy(u):
        a = unpack(u)
        da = a - a0
        dq = np.array([da[0], da[1] - da[0], da[2] - da[1]])
        return 0.5 * np.sum(kk * dq ** 2)

    def con_z(u):
        return hip_from_angles(L, unpack(u))[1] - z_target

    res = minimize(energy, np.zeros(3), constraints=[
        dict(type="eq", fun=con_z)], method="SLSQP",
        options=dict(maxiter=500, ftol=1e-14))
    a = unpack(res.x)
    p = hip_from_angles(L, a)
    # 髋点水平不动（纯垂直落地）⇒ 趾关节反向平移，脚掌随之滑移
    foot_dx = -(p[0] - p0[0])
    da = a - a0
    dq = np.array([da[0], da[1] - da[0], da[2] - da[1]])
    return dict(ok=res.success, dz_m=dz_m, foot_dx_m=foot_dx,
                dq_toe_deg=np.degrees(dq[0]), dq_ankle_deg=np.degrees(dq[1]),
                dq_knee_deg=np.degrees(dq[2]),
                hip_h0_m=p0[1], energy_J=res.fun)


def splay_rate(des=CHOSEN, m_kg=M_KG, k_toe_Nm=None, dz_probe=0.005):
    """外滑率 dx_脚掌 / dz_下沉（无量纲）。小位移差分。"""
    r = solve_sink(dz_probe, des, m_kg, k_toe_Nm)
    return r["foot_dx_m"] / dz_probe


def wheel_omega(v_sink_mps, rate, r_w_m=R_W_M):
    """两轮同时着地 ⇒ 脚掌不转只平移 ⇒ 两轮转速相同。"""
    return abs(rate) * v_sink_mps / r_w_m


# ── Q3 翻倒判据 ───────────────────────────────────────────────
def tipover_min_foot_len(h_cm_m, mu=0.40):
    """不翻倒所需的最小脚掌前后轮距（m）。

    刹车时水平力 F ≤ μ·m·g 作用在地面，对接触点产生俯仰力矩 F·h；
    地面通过前后轮法向力之差提供的反力矩上限为 m·g·b_f/2。
        μ·m·g·h ≤ m·g·b_f/2   ⇒   b_f ≥ 2·μ·h
    与质量无关 —— 这是一个纯几何判据。
    """
    return 2.0 * mu * h_cm_m


def usable_decel(b_f_m, h_cm_m, mu=0.40):
    """可用减速度（m/s²）= min(摩擦上限, 翻倒上限)。

    翻倒上限 a_tip = g·b_f/(2h) —— 脚掌越短越早起翻，刹车就必须越温柔。
    **v4 真正卡刹车力矩的是翻倒，不是摩擦**（v3 恰好相反，因为机体轮距够长）。
    """
    return min(mu * G, G * b_f_m / (2.0 * h_cm_m))


def tau_cap_Nm(b_f_m, h_cm_m, m_kg=M_KG, r_w_m=R_W_M, n_wheels=4, mu=0.40):
    """由可用减速度反推的单轮刹车力矩上限（N·m）。超过它只会把机器掀翻。"""
    return usable_decel(b_f_m, h_cm_m, mu) * m_kg * r_w_m / n_wheels


def stop_distance_m(vx_mps, b_f_m, h_cm_m, mu=0.40):
    a = usable_decel(b_f_m, h_cm_m, mu)
    return vx_mps ** 2 / (2.0 * a)


def _report():
    L, a0 = geom()
    k = stiffness_Nm()
    p0 = hip_from_angles(L, a0)
    print("=" * 76)
    print("v4 脚掌运动学分析 · chosen 5 kg 设计")
    print("=" * 76)
    print(f"三段长度  跗跖骨 {L[0]*1e3:.1f} / 胫跗骨 {L[1]*1e3:.1f} / 股骨 {L[2]*1e3:.1f} mm"
          f"   总长 {L.sum()*1e3:.1f} mm")
    print(f"着地倾角  {np.degrees(a0[0]):.1f}° / {np.degrees(a0[1]):.1f}° / {np.degrees(a0[2]):.1f}°"
          f"   髋高 {p0[1]*1e3:.0f} mm，水平伸出 {p0[0]*1e3:.0f} mm")
    print(f"扭簧刚度  踝 {k[0]:.1f} / 膝 {k[1]:.1f} / 髋 {k[2]:.1f} N·m/rad")

    print("\n── Q1 · 下沉 → 脚掌水平滑移 ──────────────────────────────")
    print(f"{'下沉mm':>8}{'脚掌滑移mm':>12}{'外滑率':>9}{'趾转角°':>9}{'踝转角°':>9}{'膝转角°':>9}")
    for dz in (0.005, 0.02, 0.05, 0.08, 0.11):
        r = solve_sink(dz)
        print(f"{dz*1e3:>8.0f}{r['foot_dx_m']*1e3:>12.1f}{r['foot_dx_m']/dz:>9.2f}"
              f"{r['dq_toe_deg']:>9.2f}{r['dq_ankle_deg']:>9.2f}{r['dq_knee_deg']:>9.2f}")

    rate = splay_rate()
    print(f"\n小位移外滑率 dx/dz = {rate:.2f}")

    print("\n── Q2 · 轮速与刹车咬合 ────────────────────────────────────")
    om = wheel_omega(V0_MPS, rate)
    print(f"着地瞬间下沉速度 {V0_MPS:.3f} m/s → 脚掌滑移速度 {abs(rate)*V0_MPS:.3f} m/s")
    print(f"  ⇒ 两轮转速 ω = {om:.1f} rad/s（轮缘线速度 {om*R_W_M:.2f} m/s）")
    print(f"  对照 v3 同工况实测：64.7 rad/s")
    for oth in (15, 25, 40, 60):
        state = "不刹（门槛以上）" if om > oth else f"咬合，给 {100*(1-om/oth):.0f}% 力矩"
        print(f"  ω_th = {oth:>2} rad/s → {state}")

    print("\n  对趾关节刚度的敏感度（趾越软，腿越容易靠屈趾吸收下沉，外滑越小）：")
    print(f"{'k_趾 N·m/rad':>14}{'外滑率':>9}{'ω rad/s':>10}")
    for kt in (2, 7, 14, 28, 56, 112):
        rr = splay_rate(k_toe_Nm=kt)
        print(f"{kt:>14}{rr:>9.2f}{wheel_omega(V0_MPS, rr):>10.1f}")

    print("\n── Q3 · 脚掌要多长才不翻 ─────────────────────────────────")
    h = p0[1]
    print(f"判据：b_f ≥ 2·μ·h（与质量无关，纯几何）")
    print(f"重心高度按髋高取 {h*1e3:.0f} mm（保守，实际重心略低）")
    print(f"{'μ':>6}{'最小脚掌轮距mm':>16}{'占腿总长':>10}")
    for mu in (0.30, 0.40, 0.50, 0.70):
        b = tipover_min_foot_len(h, mu)
        print(f"{mu:>6.2f}{b*1e3:>16.0f}{b/L.sum():>10.1%}")
    b40 = tipover_min_foot_len(h, 0.40)
    print(f"\n  μ=0.4 时要用满摩擦需要 {b40*1e3:.0f} mm 脚掌轮距，"
          f"是跗跖骨长度的 {b40/L[0]:.1f} 倍 —— 鸟不长这样的脚。")
    print("  但不必用满摩擦：脚掌短一点、刹轻一点、滑远一点，照样在跑道预算内。")

    print("\n── go/no-go · 脚掌长度 ↔ 停车距离权衡 ──────────────────────")
    import froude as FR
    h_brk = p0[1] - 0.11                      # 压到底后的有效重心高
    print(f"  有效重心高（压到底）{h_brk*1e3:.0f} mm · 跑道预算 3–15 m")
    hdr = f"{'脚掌轮距':>9}{'×跗跖骨':>9}{'起翻减速g':>10}{'可用减速g':>10}"
    frs = (1, 2, 3, 5)
    for fr in frs:
        hdr += f"{'Fr='+str(fr):>9}"
    print(hdr + f"{'单轮τ上限':>12}")
    for f in (0.5, 0.75, 1.0, 1.5, 2.0, 2.5):
        b = f * L[0]
        a_tip = G * b / (2 * h_brk)
        a = usable_decel(b, h_brk)
        row = f"{b*1e3:>8.0f}mm{f:>9.2f}{a_tip/G:>10.3f}{a/G:>10.3f}"
        for fr in frs:
            vx = float(FR.fr_to_vx(fr, M_KG))
            Ls = stop_distance_m(vx, b, h_brk)
            row += f"{Ls:>8.1f}" + ("✓" if Ls <= 15 else "✗")
        print(row + f"{tau_cap_Nm(b, h_brk):>10.3f} N·m")
    print("  ✓ = 在 15 m 跑道预算内")
    print("\n  判决：GO。脚掌轮距 ≥1.5×跗跖骨（170 mm）可覆盖到 Fr=5；")
    print("        1.0×（113 mm）可覆盖到 Fr=3。脚掌长度直接买停车距离。")
    print("  ⚠ 由此得到的单轮 τ 上限只有 0.04–0.14 N·m —— 比摩擦钳制的 0.139 还小，")
    print("    说明 v4 真正卡刹车力矩的是**翻倒**。v3 恰好相反（机体轮距 0.8 m →")
    print(f"    起翻 {G*0.8/(2*h_brk)/G:.2f} g，远高于摩擦上限，所以从不吃紧）。")


if __name__ == "__main__":
    _report()

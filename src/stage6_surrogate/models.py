"""两级保真度的冲击模型(多保真代理实验的地基)。

LF(低保真): 1-DOF 线性弹簧-阻尼(现有 Stage4 模型),解析、微秒级。
HF(高保真): 2-DOF 平面连杆落震——接触点铰接,广义坐标 q1(下段绕触点转角)、
             q2(踝弯角),髋部集中质量,拉格朗日动力学 + 关节扭簧/阻尼,RK4。
             非线性:大转角、位形相关惯量、几何刚化——LF 完全没有的物理。

共同输入 x=(L1_mm, r2, r3);共同场景(m, v0, 关节 k/c, 初始姿态)固定。
输出:peak_a(m/s²), stroke(m)。
"""
from __future__ import annotations
import numpy as np

SCEN = dict(m=200.0, v0=3.0, g=9.81,
            k1=4.5e3, k2=4.5e3,      # 关节扭簧 N·m/rad
            c1=4.0e2, c2=4.0e2,      # 线性阻尼(LF 用;HF 只保留小线性项)
            cq=90.0,                 # HF:液压式平方阻尼 N·m·(s/rad)²
            dq_stop=np.radians(55.), # HF:膝弯行程限位(超过后渐进撞块)
            k_stop=1.5e4,            # 撞块刚度
            q1_0=np.radians(50.0),   # 下段初始仰角(触地 ψ≈50°)
            thetaA=np.radians(120.0),
            thetaK=np.radians(90.0))

BOUNDS = dict(L1=(250.0, 490.0), r2=(1.3, 2.5), r3=(0.9, 2.0))


def _geom(x, s=SCEN):
    """x=(L1_mm,r2,r3) → 等效两连杆长度 l1, l2(米)。
    l2 = 踝→髋 虚拟杆:L2、L3 夹 thetaK 的对边(余弦定理)。"""
    L1, r2, r3 = x
    l1 = L1 / 1000.0
    L2, L3 = r2 * l1, r3 * l1
    l2 = np.sqrt(L2 ** 2 + L3 ** 2 - 2 * L2 * L3 * np.cos(s["thetaK"]))
    return l1, l2


def lf_eval(x, s=SCEN, dt=5e-4, T=1.2):
    """LF:力臂折算串联刚度的 1-DOF 线性落震(同 Stage4 思路)。"""
    l1, l2 = _geom(x, s)
    q1, q2 = s["q1_0"], np.pi - s["thetaA"]          # 初始位形
    # 触点→踝、触点→髋 的水平力臂
    ankle = np.array([l1 * np.cos(q1), l1 * np.sin(q1)])
    hip = ankle + l2 * np.array([np.cos(q1 + q2), np.sin(q1 + q2)])
    Ja, Jh = max(abs(ankle[0]), 1e-3), max(abs(hip[0]), 1e-3)
    K = 1.0 / (Ja ** 2 / s["k1"] + Jh ** 2 / s["k2"])
    C = 1.0 / (Ja ** 2 / s["c1"] + Jh ** 2 / s["c2"])
    m, g, v = s["m"], s["g"], s["v0"]
    d, vv = 0.0, v
    peak, dmax = 0.0, 0.0
    for _ in range(int(T / dt)):
        a = g - (C * vv + K * d) / m
        peak = max(peak, abs(a)); 
        vv += a * dt; d += vv * dt
        dmax = max(dmax, d)
        if vv < 0: break
    return dict(peak_a=peak, stroke=dmax)


def hf_eval(x, s=SCEN, dt=2e-4, T=1.5):
    """HF:2-DOF 平面连杆(触点铰接 + 髋部点质量)拉格朗日动力学。
    M(q) qdd + m J^T Jd qd = J^T F_g + tau_spring;小关节惯量正则化防奇异。"""
    l1, l2 = _geom(x, s)
    m, g = s["m"], s["g"]
    q = np.array([s["q1_0"], np.pi - s["thetaA"]])   # q2: 踝处外角(0=伸直)
    q0 = q.copy()
    # 初速:髋竖直 -v0 → 关节速率(最小范数逆解)
    def jac(q):
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        return np.array([[-l1 * s1 - l2 * s12, -l2 * s12],
                         [ l1 * c1 + l2 * c12,  l2 * c12]])
    J = jac(q)
    qd = np.linalg.lstsq(J, np.array([0.0, -s["v0"]]), rcond=None)[0]
    Ireg = np.diag([0.06 * m * l1 ** 2, 0.06 * m * l2 ** 2])   # 段惯量近似(正则)
    kvec = np.array([s["k1"], s["k2"]])
    clin = np.array([s["c1"], s["c2"]]) * 0.25          # HF 线性阻尼只留小项
    cq, dstop, kstop = s["cq"], s["dq_stop"], s["k_stop"]

    def qdd_f(q, qd):
        s1, c1 = np.sin(q[0]), np.cos(q[0])
        s12, c12 = np.sin(q[0] + q[1]), np.cos(q[0] + q[1])
        J = np.array([[-l1 * s1 - l2 * s12, -l2 * s12],
                      [ l1 * c1 + l2 * c12,  l2 * c12]])
        # dJ/dt
        w1, w12 = qd[0], qd[0] + qd[1]
        Jd = np.array([[-l1 * c1 * w1 - l2 * c12 * w12, -l2 * c12 * w12],
                       [-l1 * s1 * w1 - l2 * s12 * w12, -l2 * s12 * w12]])
        M = m * J.T @ J + Ireg
        tau = -kvec * (q - q0) - clin * qd - cq * qd * np.abs(qd)   # 液压平方阻尼
        over = np.abs(q - q0) - dstop                                # 行程限位撞块
        tau -= np.where(over > 0, kstop * over * np.sign(q - q0), 0.0)
        rhs = J.T @ np.array([0.0, -m * g]) - m * J.T @ (Jd @ qd) + tau
        return np.linalg.solve(M, rhs), J, Jd

    hip_y = lambda q: l1 * np.sin(q[0]) + l2 * np.sin(q[0] + q[1])
    y0 = hip_y(q)
    peak, dmax = 0.0, 0.0
    n = int(T / dt)
    for i in range(n):
        qdd, J, Jd = qdd_f(q, qd)
        acc_y = (Jd @ qd + J @ qdd)[1]                 # 髋竖直加速度
        peak = max(peak, abs(acc_y))
        # RK4
        k1a, _, _ = qdd_f(q, qd)
        k2a, _, _ = qdd_f(q + 0.5 * dt * qd, qd + 0.5 * dt * k1a)
        k3a, _, _ = qdd_f(q + 0.5 * dt * (qd + 0.5 * dt * k1a), qd + 0.5 * dt * k2a)
        k4a, _, _ = qdd_f(q + dt * (qd + 0.5 * dt * k2a), qd + dt * k3a)
        qd = qd + dt * (k1a + 2 * k2a + 2 * k3a + k4a) / 6.0
        q = q + dt * qd
        drop = y0 - hip_y(q); dmax = max(dmax, drop)
        vy = (jac(q) @ qd)[1]
        if vy > 0 and i > 20: break                    # 髋开始回升 → 压缩相结束
        if hip_y(q) < 0.05 * (l1 + l2) or not np.all(np.isfinite(q)):
            return dict(peak_a=np.nan, stroke=np.nan)  # 塌陷/发散
    return dict(peak_a=peak, stroke=dmax)


def lhs(n, seed=0):
    rng = np.random.default_rng(seed)
    dims = [BOUNDS["L1"], BOUNDS["r2"], BOUNDS["r3"]]
    out = np.empty((n, 3))
    for j, (a, b) in enumerate(dims):
        edges = a + (b - a) * (np.arange(n) + rng.random(n)) / n
        out[:, j] = rng.permutation(edges)
    return out

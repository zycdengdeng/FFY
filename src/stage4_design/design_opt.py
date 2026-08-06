"""Stage 4 · 对接蓝本第二步:MotionChain → 参数化连杆设计空间 → 冲击评估 → Pareto 选优。

作用:把 Stage 3 产出的 motionchain.json(生物运动学:初始关节角 + 骨长比先验)真正
喂进「生成式设计」流程,端到端跑通一次,验证第 1 步的输出能驱动设计。

对齐蓝本(AESCTE §2.3 / §2.5):
  设计变量 P=(L1,L2,L3);L1∈[250,490]mm;r2=L2/L1∈[1.3,2.5](eq.12);r3=L3/L1∈[0.9,2.0](eq.13)
  采样:拉丁超立方(LHS,蓝本 eq.15)
  目标:min Peak_a、min Peak_jerk、max EA(能量吸收)→ 多目标 Pareto(蓝本 eq.22)

★重要诚实声明:冲击评估用的是**简化解析代理模型**,不是 ANSYS 瞬态有限元。
  它把三连杆在竖直落震下等效为 1-DOF 质量-弹簧-阻尼系统,几何(L,初始角)通过运动学
  雅可比(力臂)调制等效刚度/阻尼——足以体现「几何→性能」的耦合、驱动设计排序与选优,
  用于打通 pipeline 与验证接口,不用于替代蓝本的高保真 FE 结论(数值量级为示意)。
  motionchain 的初始角决定触地姿态(力臂),骨长比作为参考点标注在设计空间里。

纯 numpy + matplotlib。用法:
  python src/stage4_design/design_opt.py --mc outputs/swan01/motionchain.json --clip_id swan01 --n 60
产出:outputs/<clip>/design_results.csv + design_opt.json + stage4_pareto.png + stage4_response.png
"""
from __future__ import annotations
import argparse, csv, json, os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

# 设计空间(蓝本 §2.3.2 eq.12/13)
L1_RANGE = (250.0, 490.0)     # mm, tarsometatarsus
R2_RANGE = (1.3, 2.5)         # L2/L1  tibiotarsus/tarsometatarsus
R3_RANGE = (0.9, 2.0)         # L3/L1  femur/tarsometatarsus

# 冲击场景 + 铰接名义参数(示意量级,固定;设计变量只有几何 L,同蓝本)
# 踝、膝取不同阻尼比(生物上柔顺不同):几何改变两关节的力臂权重 → 独立调等效频率 ω 与
# 阻尼比 ζ → 目标空间成二维前沿(否则单一 ω 自由度会让 Pareto 退化为全体)。
SCEN = dict(m=200.0, v0=3.0, g=9.81,          # 每腿等效簧上质量kg / 触地竖直速度m/s / 重力
            k_ankle=8.0e3, k_knee=8.0e3,      # 关节转动刚度 N·m/rad
            c_ankle=1.5e2, c_knee=6.0e2)      # 关节转动阻尼 N·m·s/rad(膝更阻尼)


def _rot(vec, deg):
    r = np.radians(deg); c, s = np.cos(r), np.sin(r)
    return np.array([c * vec[0] - s * vec[1], s * vec[0] + c * vec[1]])


def leg_geometry(L1, L2, L3, phi, theta_A, theta_K):
    """由连杆长度 + 初始关节角建触地姿态的 2D 关节坐标(mm)。
    接地点=MTP(足);向上依次 ankle→knee→hip。返回各关节 xy 与对接地点的水平力臂。"""
    M = np.array([0.0, 0.0])                      # MTP 接地
    A = M + L1 * np.array([np.cos(np.radians(phi)), np.sin(np.radians(phi))])   # tarsometatarsus 抬起
    # 踝处内角 theta_A(knee-ankle-mtp):把 (M-A) 方向转 theta_A 得 (K-A) 方向
    dirMA = (M - A) / (np.linalg.norm(M - A) + 1e-9)
    dAK = _rot(dirMA, theta_A)
    K = A + L2 * dAK
    # 膝处内角 theta_K(hip-knee-ankle):把 (A-K) 方向转 theta_K 得 (H-K) 方向
    dirAK = (A - K) / (np.linalg.norm(A - K) + 1e-9)
    dKH = _rot(dirAK, theta_K)
    H = K + L3 * dKH
    # 竖直力(过接地点)对各关节的力臂 = 关节到接地点的水平距离
    lever_ankle = abs(A[0] - M[0]) / 1000.0       # mm→m
    lever_knee = abs(K[0] - M[0]) / 1000.0
    return dict(M=M, A=A, K=K, H=H, hip_h=H[1] / 1000.0,
                lever_ankle=max(lever_ankle, 1e-3), lever_knee=max(lever_knee, 1e-3))


def impact_response(geo, scen=SCEN, dt=5e-4, T=1.5):
    """1-DOF 竖直落震:m δ'' + C_eff δ' + K_eff δ = m g;δ(0)=0,δ'(0)=v0。
    关节转动刚度/阻尼经力臂投影成竖直等效量(串联柔度相加)。RK4 积分。"""
    Ja, Jk = geo["lever_ankle"], geo["lever_knee"]
    # 竖直等效:柔度 = Σ J²/k(串联),K_eff=1/柔度;阻尼同理
    K_eff = 1.0 / (Ja ** 2 / scen["k_ankle"] + Jk ** 2 / scen["k_knee"])
    C_eff = 1.0 / (Ja ** 2 / scen["c_ankle"] + Jk ** 2 / scen["c_knee"])
    m, g, v0 = scen["m"], scen["g"], scen["v0"]
    n = int(T / dt)
    d, v = 0.0, v0
    acc = np.empty(n); ts = np.empty(n); dmax = 0.0
    def accel(d, v): return g - (C_eff * v + K_eff * d) / m
    ea = 0.0
    for i in range(n):
        a = accel(d, v); acc[i] = a; ts[i] = i * dt
        ea += C_eff * v * v * dt                    # 阻尼耗散功率积分(诊断:被动落震近守恒)
        dmax = max(dmax, d)
        # RK4
        k1v = accel(d, v);             k1d = v
        k2v = accel(d + .5*dt*k1d, v + .5*dt*k1v); k2d = v + .5*dt*k1v
        k3v = accel(d + .5*dt*k2d, v + .5*dt*k2v); k3d = v + .5*dt*k2v
        k4v = accel(d + dt*k3d, v + dt*k3v);       k4d = v + dt*k3v
        v += dt*(k1v+2*k2v+2*k3v+k4v)/6.0
        d += dt*(k1d+2*k2d+2*k3d+k4d)/6.0
        if d < 0 and v < 0:                          # 回弹离地,停止
            acc = acc[:i+1]; ts = ts[:i+1]; break
    peak_a = float(np.max(np.abs(acc)))
    jerk = np.gradient(acc, ts)
    peak_jerk = float(np.max(np.abs(jerk)))
    return dict(t=ts, a=acc, peak_a=peak_a, peak_jerk=peak_jerk, stroke=float(dmax),
                EA=float(ea), K_eff=K_eff, C_eff=C_eff)


def lhs(n, seed=0):
    """拉丁超立方(蓝本 eq.15):每维分 n 层,层内均匀取样,各维独立乱序。"""
    rng = np.random.default_rng(seed)
    dims = [L1_RANGE, R2_RANGE, R3_RANGE]
    out = np.empty((n, 3))
    for j, (a, b) in enumerate(dims):
        edges = a + (b - a) * (np.arange(n) + rng.random(n)) / n
        out[:, j] = rng.permutation(edges)
    return out   # 列:L1, r2, r3


def evaluate(samples, ia):
    rows = []
    for L1, r2, r3 in samples:
        L2, L3 = r2 * L1, r3 * L1
        geo = leg_geometry(L1, L2, L3, ia["phi"], ia["theta_A"], ia["theta_K"])
        r = impact_response(geo)
        rows.append(dict(L1=L1, L2=L2, L3=L3, r2=r2, r3=r3,
                         peak_a=r["peak_a"], peak_jerk=r["peak_jerk"],
                         stroke=r["stroke"], EA=r["EA"]))
    return rows


def pareto_mask(objs):
    """objs: (N,3) 全部「越小越好」(EA 取负)。返回非支配布尔掩码。"""
    n = len(objs); dom = np.zeros(n, bool)
    for i in range(n):
        if dom[i]:
            continue
        for k in range(n):
            if k != i and np.all(objs[k] <= objs[i]) and np.any(objs[k] < objs[i]):
                dom[i] = True; break
    return ~dom


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", required=True, help="Stage3 motionchain.json")
    ap.add_argument("--clip_id", default="swan01"); ap.add_argument("--n", type=int, default=60)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    mc = json.load(open(args.mc, encoding="utf-8"))
    ia = mc["init_angles"]; r2_bio, r3_bio = mc["link_ratio"]
    outdir = os.path.join(args.out, args.clip_id); os.makedirs(outdir, exist_ok=True)

    samples = lhs(args.n)
    rows = evaluate(samples, ia)
    # 三目标(均越小越好):峰值加速度、峰值 jerk、缓冲行程。EA 近守恒仅作诊断,不进优化。
    objs = np.array([[r["peak_a"], r["peak_jerk"], r["stroke"]] for r in rows])
    pm = pareto_mask(objs)
    # 折中解:三目标各自归一化后到理想点欧氏距离最小
    norm = (objs - objs.min(0)) / (np.ptp(objs, axis=0) + 1e-9)
    sel = int(np.where(pm)[0][np.argmin(np.linalg.norm(norm[pm], axis=1))])
    best = rows[sel]

    # 生物参考设计(用 motionchain 骨长比 + 中位 L1),看它落在设计空间何处
    L1_bio = np.mean(L1_RANGE)
    geo_bio = leg_geometry(L1_bio, r2_bio * L1_bio, r3_bio * L1_bio, ia["phi"], ia["theta_A"], ia["theta_K"])
    r_bio = impact_response(geo_bio)
    r3_out = not (R3_RANGE[0] <= r3_bio <= R3_RANGE[1])

    print(f"[stage4] LHS N={args.n} · 简化解析冲击代理(非FE)")
    print(f"  init posture from motionchain: phi={ia['phi']:.1f} thetaA={ia['theta_A']:.0f} thetaK={ia['theta_K']:.0f}")
    print(f"  bio link_ratio r2={r2_bio} r3={r3_bio}  → r3 在设计范围[{R3_RANGE[0]},{R3_RANGE[1]}] {'之外!' if r3_out else '之内'}")
    print(f"  Pareto 非支配集: {int(pm.sum())}/{args.n}")
    print(f"  选中折中解 P=(L1={best['L1']:.0f},L2={best['L2']:.0f},L3={best['L3']:.0f})mm "
          f"r2={best['r2']:.2f} r3={best['r3']:.2f}")
    print(f"    Peak_a={best['peak_a']:.1f} m/s² ({best['peak_a']/9.81:.1f}g) "
          f"Peak_jerk={best['peak_jerk']:.0f} stroke={best['stroke']*1000:.0f}mm (EA≈{best['EA']:.0f} 诊断)")
    print(f"  生物比例设计(L1={L1_bio:.0f}): Peak_a={r_bio['peak_a']:.1f} m/s² ({r_bio['peak_a']/9.81:.1f}g) "
          f"stroke={r_bio['stroke']*1000:.0f}mm")

    # CSV
    with open(os.path.join(outdir, "design_results.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["L1", "L2", "L3", "r2", "r3", "peak_a", "peak_jerk", "stroke", "EA", "pareto"])
        w.writeheader()
        for i, r in enumerate(rows):
            w.writerow({**{k: round(r[k], 3) for k in r}, "pareto": int(pm[i])})

    json.dump({
        "surrogate": "1-DOF analytical impact (NOT ANSYS FE); for end-to-end wiring + design ranking only",
        "design_space": {"L1_mm": L1_RANGE, "r2": R2_RANGE, "r3": R3_RANGE, "sampling": "LHS", "n": args.n},
        "objectives": ["min peak_a", "min peak_jerk", "min stroke"],
        "EA_note": "被动1-DOF落震耗散能量≈初始动能(守恒),对几何近不敏感(变化~4%),故降为诊断量,以缓冲行程 stroke 作第三目标(与峰值加速度真实权衡)。蓝本用高保真FE,EA可判据。",
        "jerk_note": "线性代理里 peak_jerk 与 peak_a 近共线(corr≈1.0),有效权衡实为 peak_a↔stroke;蓝本非线性瞬态FE中三者解耦,故仍保留三目标接口。",
        "init_from_motionchain": ia, "bio_link_ratio": {"r2": r2_bio, "r3": r3_bio,
            "r3_outside_design_range": r3_out, "note": "biological femur ratio below engineering lower bound 0.9"},
        "selected": best, "pareto_count": int(pm.sum()),
    }, open(os.path.join(outdir, "design_opt.json"), "w"), ensure_ascii=False, indent=2)

    _plots(rows, objs, pm, sel, best, r2_bio, r3_bio, r3_out, ia, outdir)
    print(f"[stage4] wrote design_results.csv + design_opt.json + stage4_pareto.png + stage4_response.png → {outdir}")


def _plots(rows, objs, pm, sel, best, r2_bio, r3_bio, r3_out, ia, outdir):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    pa = np.array([r["peak_a"] for r in rows]); st = np.array([r["stroke"]*1000 for r in rows])
    pj = np.array([r["peak_jerk"] for r in rows])
    r2s = np.array([r["r2"] for r in rows]); r3s = np.array([r["r3"] for r in rows])

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    # 左:目标空间 Peak_a vs 缓冲行程(色=jerk),标 Pareto + 选中。软→低加速度但行程更长
    sc = ax[0].scatter(st, pa / 9.81, c=pj, cmap="viridis", s=35, alpha=.7)
    ax[0].scatter(st[pm], pa[pm] / 9.81, edgecolor="crimson", facecolor="none", s=90, lw=1.6, label="Pareto set")
    ax[0].scatter(st[sel], pa[sel] / 9.81, marker="*", color="red", s=320, edgecolor="k", zorder=5, label="selected")
    ax[0].set_xlabel("cushioning stroke (mm) (↓ better)"); ax[0].set_ylabel("Peak acceleration (g) (↓ better)")
    ax[0].set_title("Objective space: peak-a vs stroke trade (surrogate)"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    plt.colorbar(sc, ax=ax[0], label="Peak jerk")
    # 右:设计空间 r2-r3,标蓝本可行域 + 生物点(可能越界)
    ax[1].add_patch(plt.Rectangle((R2_RANGE[0], R3_RANGE[0]), R2_RANGE[1]-R2_RANGE[0], R3_RANGE[1]-R3_RANGE[0],
                                  fill=False, ls="--", ec="gray", label="blueprint feasible box"))
    ax[1].scatter(r2s, r3s, c="C0", s=25, alpha=.5, label="LHS samples")
    ax[1].scatter(r2s[pm], r3s[pm], edgecolor="crimson", facecolor="none", s=80, lw=1.4, label="Pareto")
    ax[1].scatter(best["r2"], best["r3"], marker="*", color="red", s=320, edgecolor="k", zorder=5, label="selected")
    ax[1].scatter(r2_bio, r3_bio, marker="D", color="darkgreen", s=90, zorder=6,
                  label=f"biological (r3={r3_bio}{' OUTSIDE' if r3_out else ''})")
    ax[1].axhline(R3_RANGE[0], color="crimson", ls=":", lw=1, alpha=.6)
    ax[1].set_xlabel("r2 = L2/L1"); ax[1].set_ylabel("r3 = L3/L1")
    ax[1].set_title("Design space: biology vs engineering box"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "stage4_pareto.png"), dpi=110); plt.close()

    # 响应时程:选中最优 vs 生物比例设计
    L1b = np.mean(L1_RANGE)
    g_sel = leg_geometry(best["L1"], best["L2"], best["L3"], ia["phi"], ia["theta_A"], ia["theta_K"])
    g_bio = leg_geometry(L1b, r2_bio*L1b, r3_bio*L1b, ia["phi"], ia["theta_A"], ia["theta_K"])
    rs, rb = impact_response(g_sel), impact_response(g_bio)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(rs["t"]*1000, rs["a"]/9.81, color="red", lw=2, label=f"selected  peak {rs['peak_a']/9.81:.1f}g")
    ax.plot(rb["t"]*1000, rb["a"]/9.81, color="darkgreen", lw=2, ls="--",
            label=f"biological ratio  peak {rb['peak_a']/9.81:.1f}g")
    ax.axhline(0, color="gray", lw=.6)
    ax.set_xlabel("time (ms)"); ax.set_ylabel("vertical acceleration (g)")
    ax.set_title("Touchdown impact response (1-DOF surrogate)"); ax.legend(fontsize=9); ax.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(outdir, "stage4_response.png"), dpi=110); plt.close()


if __name__ == "__main__":
    main()

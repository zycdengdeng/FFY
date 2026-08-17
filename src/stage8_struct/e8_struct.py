"""E8 · 结构落地包:从落震载荷反算最小截面 → 衍生质量,兑现"轻量化"。

链条(每个设计零新增假设):
  Exudyn 落震(加转角传感器)→ 三关节扭簧力矩时程 M_j(t)=k·Δθ+c·Δθ̇ + 腿传力 F(t)
  → 每段杆设计弯矩 = 相邻关节峰值力矩(保守),轴力 = F_peak(保守全额)
  → 薄壁圆管(壁厚=0.1D)按 弯曲+轴压 ≤ 许用应力 反算最小外径 D
  → Euler 屈曲复核 + 打印最小径限制 → 段质量 = ρ·A·L → 腿结构质量。

材料预设(数据表典型值;全尺寸对应 MMPDS/CMH-17 口径):
  cfnylon 短碳纤尼龙(3D 打印,对接 E10 样机) | al7075 铝合金 | ti64 钛合金
安全系数默认 2.0(<25kg 无人机工程惯例;适航级为 1.5,FAR 25.303)。

实验:取代表性工况(冻结考卷抽 6 题或内置网格),cVAE 出 40 候选 → 实摔选优
→ 结构定尺寸;天鹅几何+规则配簧作对照。输出每题:最优设计、关节峰值力矩、
各段外径、腿质量、质量占比;并与 5% 段质量规则对比(规则蕴含单腿=15% 体重)。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage8_struct/e8_struct.py \
    --model outputs/gen_e5c_r85/cvae_s1.pt --refs outputs/gen_e5/refs.json \
    --out outputs/gen_e8 --workers 64
仿真量:~6 题 × 41 次 ≈ 250 次,分钟级。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage6_surrogate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage7_generative"))
import models as M                                    # noqa: E402
from hf_exudyn import SCEN_BIRD_X, _metrics, rotY     # noqa: E402
from train_cvae import CVAE, norm                     # noqa: E402

import exudyn as exu                                  # noqa: E402
from exudyn.utilities import *                        # noqa: E402,F401,F403

MATERIALS = {   # (密度 kg/m³, 屈服/强度极限 Pa, 弹性模量 Pa)
    "cfnylon": (1180.0, 70e6, 6e9),
    "al7075": (2810.0, 503e6, 71.7e9),
    "ti64": (4430.0, 880e6, 113.8e9),
}
WALL = 0.10          # 薄壁圆管 壁厚/外径
SWAN_X3 = (111.2, 1.764, 0.951)


# ---------------------------------------------------------------- 带传感器的落震
def exu_eval_struct(x7, sc):
    """与 hf_exudyn.exu_eval 同模型,外加三杆转角传感器 → 关节力矩时程。"""
    L1, r2, r3 = x7[0], x7[1], x7[2]
    s = sc
    l1 = L1 / 1000.0; l2 = r2 * l1; l3 = r3 * l1
    m, g, v0 = s["m"], s["g"], s["v0"]
    ms = s["seg_mass_frac"] * m
    a1 = s["q1_0"]; a2 = a1 + (np.pi - s["thetaA"]); a3 = a2 - (np.pi - s["thetaK"])
    d = lambda a: np.array([np.cos(a), 0., np.sin(a)])
    F = np.array([0., 0., s["gap0"] + s["r_foot"]])
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
                               inertia=InertiaSphere(mass=m, radius=0.12),
                               gravity=[0, 0, -g])
    mbs.CreatePrismaticJoint(bodyNumbers=[ground, body], position=list(H), axis=[0, 0, 1])
    for (b0, b1, P, k, c) in [(body, femur, H, s["k_hip"], s["c_hip"]),
                              (femur, tibio, K, s["k_knee"], s["c_knee"]),
                              (tibio, tarso, A, s["k_ankle"], s["c_ankle"])]:
        mbs.CreateRevoluteJoint(bodyNumbers=[b0, b1], position=list(P), axis=[0, 1, 0])
        mbs.CreateTorsionalSpringDamper(bodyNumbers=[b0, b1], position=list(P),
                                        axis=[0, 1, 0], stiffness=k, damping=c)
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
        return None
    acc = mbs.GetSensorStoredData(sAcc); pos = mbs.GetSensorStoredData(sPos)
    t = acc[:, 0]; az = acc[:, 3]; z = pos[:, 3]
    if not np.all(np.isfinite(az)):
        return None
    stroke = float(z[0] - np.min(z))
    if stroke > 0.6 * (l1 + l2 + l3):
        return None
    rot = [mbs.GetSensorStoredData(si)[:, 2] for si in sRot]   # 绕 Y 角(femur/tibio/tarso)
    h = t[1] - t[0]
    dth = dict(hip=rot[0] - rot[0][0],
               knee=(rot[1] - rot[0]) - (rot[1][0] - rot[0][0]),
               ankle=(rot[2] - rot[1]) - (rot[2][0] - rot[1][0]))
    Mj = {}
    for jn, kk, cc_ in [("hip", s["k_hip"], s["c_hip"]),
                        ("knee", s["k_knee"], s["c_knee"]),
                        ("ankle", s["k_ankle"], s["c_ankle"])]:
        th = dth[jn]
        Mj[jn] = float(np.max(np.abs(kk * th + cc_ * np.gradient(th, h))))
    met = dict(peak_a=float(np.max(np.abs(az))), stroke=stroke)
    met.update(_metrics(t, z, az, m, g, v0))
    met.update(M_hip=Mj["hip"], M_knee=Mj["knee"], M_ankle=Mj["ankle"],
               seg_len=[l1, l2, l3])
    return met


def _job(args):
    x7, m, v0 = args
    sc = M.bird_size_x({**SCEN_BIRD_X, "m": m, "v0": v0, "kappa": 4.0}, np.asarray(x7))
    return exu_eval_struct(np.asarray(x7), sc)


# ---------------------------------------------------------------- 结构定尺寸
def size_segment(Mb, Fax, L, mat, sf, dmin):
    """薄壁圆管:弯曲+轴压组合应力 + Euler 屈曲 → 最小外径与质量。"""
    rho, sig_y, E = mat
    sig = sig_y / sf
    D = max((Mb / (0.05796 * sig)) ** (1 / 3) if Mb > 0 else dmin, dmin)
    for _ in range(30):                                   # 组合应力定点迭代
        st = Mb / (0.05796 * D ** 3) + Fax / (0.2827 * D ** 2)
        if st <= sig:
            break
        D *= (st / sig) ** (1 / 3)
    governs = "bending+axial"
    Dbuck = (sf * Fax * L ** 2 / (np.pi ** 2 * E * 0.02898)) ** 0.25 if Fax > 0 else 0.0
    if Dbuck > D:
        D, governs = Dbuck, "buckling"
    if D <= dmin + 1e-12:
        governs = "min-print-size"
    mass = rho * 0.2827 * D ** 2 * L
    return D, mass, governs


def size_leg(met, mat, sf, nlegs, dmin):
    """段设计弯矩=相邻关节峰值力矩的较大者(保守);轴力=F_peak 全额(保守)。"""
    Mh, Mk, Ma = met["M_hip"] / nlegs, met["M_knee"] / nlegs, met["M_ankle"] / nlegs
    Fax = met["F_peak"] / nlegs
    segs = [("tarso(跖)", met["seg_len"][0], Ma),            # 足-踝段:踝力矩
            ("tibio(胫)", met["seg_len"][1], max(Ma, Mk)),
            ("femur(股)", met["seg_len"][2], max(Mk, Mh))]
    out, total = [], 0.0
    for name, L, Mb in segs:
        D, mass, gov = size_segment(Mb, Fax, L, mat, sf, dmin)
        out.append(dict(seg=name, L_mm=L * 1e3, M_Nm=Mb, D_mm=D * 1e3,
                        mass_g=mass * 1e3, governs=gov))
        total += mass
    return out, total


# ---------------------------------------------------------------- 主实验
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/gen_e5c_r85/cvae_s1.pt")
    ap.add_argument("--refs", default="outputs/gen_e5/refs.json")
    ap.add_argument("--out", default="outputs/gen_e8")
    ap.add_argument("--material", default="cfnylon", choices=list(MATERIALS))
    ap.add_argument("--sf", type=float, default=2.0, help="安全系数")
    ap.add_argument("--nlegs", type=int, default=1,
                    help="分载腿数(1=单腿扛全重,最保守)")
    ap.add_argument("--dmin", type=float, default=0.004, help="最小外径 m(打印工艺限)")
    ap.add_argument("--ngen", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=808_001)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    mat = MATERIALS[args.material]

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    model = CVAE(xd=ck["xd"], cd=len(c_lo), z=ck["zdim"])
    model.load_state_dict(ck["state"]); model.eval()

    # 代表性工况:考卷里抽 6 题铺满 (m, v0) 角落;无考卷则用内置网格
    if os.path.exists(args.refs):
        refs = json.load(open(args.refs))
        ms = np.array([r["m"] for r in refs]); vs = np.array([r["v0"] for r in refs])
        picks, chosen = [], set()
        for tm, tv in [(ms.min(), vs.min()), (ms.min(), vs.max()),
                       (np.median(ms), np.median(vs)),
                       (ms.max(), vs.min()), (ms.max(), vs.max()),
                       (np.median(ms), vs.max())]:
            i = int(np.argmin((ms - tm) ** 2 + ((vs - tv) * 5) ** 2))
            if i not in chosen:
                chosen.add(i); picks.append(refs[i])
    else:
        picks = [dict(m=m_, v0=v_, gcap=12 * 9.81, smax=0.035, ref=np.nan)
                 for m_, v_ in [(1.5, 0.8), (1.5, 1.6), (5, 1.2),
                                (10, 0.8), (10, 1.6), (7, 1.4)]]
    print(f"[e8] material={args.material} SF={args.sf} nlegs={args.nlegs} "
          f"| 工况 {len(picks)} 个")

    rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for pi, r in enumerate(picks):
            torch.manual_seed(args.seed + pi)
            cvec = [r["m"], r["v0"], r["gcap"], r["smax"]][:len(c_lo)]
            cn = torch.tensor(norm(np.array(cvec), c_lo, c_hi), dtype=torch.float32)
            Xg = x_lo + (x_hi - x_lo) * model.sample(cn, args.ngen).numpy()
            jobs = [(x, r["m"], r["v0"]) for x in Xg]
            # 天鹅对照:骨架几何 + 规则配簧(κ=4/4/16, ζ=0.03)
            xs = np.array(list(SWAN_X3) + [4.0, 4.0, 16.0, 0.03])[:len(x_lo)]
            jobs.append((xs, r["m"], r["v0"]))
            res = list(ex.map(_job, jobs, chunksize=2))

            feas = [(x, mt) for x, mt in zip(Xg, res[:-1])
                    if mt and mt["peak_a"] <= r["gcap"] and mt["stroke"] <= r["smax"]]
            entry = dict(m=r["m"], v0=r["v0"], gcap_g=r["gcap"] / 9.81,
                         smax_mm=r["smax"] * 1e3, n_feas=len(feas))
            if feas:
                xb, mb = min(feas, key=lambda p: p[1]["peak_a"])
                segs, mtot = size_leg(mb, mat, args.sf, args.nlegs, args.dmin)
                entry.update(best=dict(
                    x=np.round(xb, 3).tolist(), peak_g=mb["peak_a"] / 9.81,
                    stroke_mm=mb["stroke"] * 1e3,
                    M_hip=mb["M_hip"], M_knee=mb["M_knee"], M_ankle=mb["M_ankle"],
                    segments=segs, leg_mass_g=mtot * 1e3,
                    mass_frac_pct=100 * mtot / r["m"]))
            sw = res[-1]
            if sw:
                ssw, msw = size_leg(sw, mat, args.sf, args.nlegs, args.dmin)
                entry.update(swan=dict(peak_g=sw["peak_a"] / 9.81,
                                       leg_mass_g=msw * 1e3,
                                       mass_frac_pct=100 * msw / r["m"]))
            rows.append(entry)
            b = entry.get("best"); sname = entry.get("swan")
            print(f"  工况 m={r['m']:.1f}kg v0={r['v0']:.2f}: "
                  + (f"生成腿 {b['leg_mass_g']:.0f}g({b['mass_frac_pct']:.1f}%体重) "
                     f"峰值{b['peak_g']:.1f}g" if b else "无可行")
                  + (f" | 天鹅对照 {sname['leg_mass_g']:.0f}g"
                     f"({sname['mass_frac_pct']:.1f}%)" if sname else ""))

    # 5% 规则对比:规则质量 = 3 段 × 5% 体重
    for e in rows:
        e["rule5pct_leg_mass_g"] = 1e3 * 0.15 * e["m"]
    json.dump(dict(material=args.material, sf=args.sf, nlegs=args.nlegs,
                   wall_ratio=WALL, dmin_mm=args.dmin * 1e3, rows=rows),
              open(os.path.join(args.out, "e8_struct.json"), "w"),
              indent=2, ensure_ascii=False)
    print("\n== E8 结构落地包 ==")
    for e in rows:
        b = e.get("best")
        if not b:
            continue
        rr = e["rule5pct_leg_mass_g"]
        secs = ", ".join("{}:{:.1f}mm/{}".format(sg["seg"], sg["D_mm"], sg["governs"])
                         for sg in b["segments"])
        print(f"  m={e['m']:.1f}kg: 衍生腿质量 {b['leg_mass_g']:.0f}g "
              f"= {b['mass_frac_pct']:.1f}% 体重(5% 规则蕴含 {rr:.0f}g → "
              f"轻 {rr / max(b['leg_mass_g'], 1e-9):.1f}×); 控制截面 {secs}")
    print(f"[e8] done → {args.out}/e8_struct.json")


if __name__ == "__main__":
    main()

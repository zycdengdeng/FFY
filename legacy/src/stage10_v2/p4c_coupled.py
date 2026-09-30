# -*- coding: utf-8 -*-
"""P4c · 锁自由度值不值:独立三关节 vs 膝-髋连杆耦合。

背景:P4/P4b 说踝是唯一真通道、膝是平坦方向、髋近乎刚性;加上股骨只有 111 mm
装不下两套共面减震器 —— 于是提出用连杆把膝锁给髋,只留踝可调。
但"刚度不随工况变"≠"自由度可以锁掉":本实验就验证这一步到底掉多少性能。

两条臂,同几何、同工况、同判官、同搜索预算:
  A 独立  3 关节自由 —— 搜 (κ踝, κ膝, τ),κ髋 固定        [现状,对应"左右错开"的实机]
  B 耦合  连杆锁膝   —— 搜 (κ踝, κ髋, τ),κ膝 = 0         [提案,一套减震器在髋]
连杆 = 无质量两力杆 ≡ 距离约束(physics_v2 的 couple_rod),机身端与胫跗骨端各外偏 50 mm。

公平性口径:两条臂各有 **3 个可调旋钮**(旋钮数 = 可调零件数,是设计上真正的成本)。
  A 的折叠刚度由 κ膝(可调)+κ髋(固定)共同决定;B 的由 κ髋(可调)单独决定。
  若担心 A 被 κ髋 固定所限,可加 --fair-hip 让 A 也搜 κ髋(变成 4 旋钮,对 A 有利)。

每条臂各做两问:
  ① 按工况整定的上界(每个工况各自搜最优)
  ② 鲁棒被动(一组固定设定打全部工况) —— 这才是产品路线关心的
用法(A100,单行):
  OMP_NUM_THREADS=1 python src/stage10_v2/p4c_coupled.py --workers 128 --out outputs/v2_p4c
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P
from bioprior import BioPrior, KAP_RANGE_V21, ZETA_RANGE_V21
from factory_v2 import zeta_of_kc
from e17_emergent_b import load as load_cvae
from train_cvae import norm

GCAP, SMAX = 10 * 9.81, 0.024
KA_RG, KK_RG, KH_RG = KAP_RANGE_V21
TAU_RG = ZETA_RANGE_V21
OFF_HIP = OFF_KNEE = 0.050          # 连杆两端外偏(米);50/50 由间隙+耦合比扫描选定
ROD = dict(off_hip=OFF_HIP, off_knee=OFF_KNEE)
BIG = 1e6
KCs = [5e4, 1e5, 2.5e5, 1e6]; V0s = [0.6, 1.0, 1.4, 1.8]
CONDS = [(kc, v0) for kc in KCs for v0 in V0s]


def _ev(a):
    x9, m, v0, kc, coupled = a
    base = {**P.SCEN_BIRD_X, "hip_damp_unified": True}
    if coupled:
        base["couple_rod"] = ROD
    r = P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=base)
    if r is None or r.get("fail"):
        return (BIG * 2, None, False, [(r or {}).get("fail", "?")])
    ok, why = P.feasible_v2(r, GCAP, SMAX)
    g = r["peak_a"] / 9.81
    return (g + (0 if ok else BIG), g, bool(ok), list(why))


def assemble(geom, arm, p):
    """geom=(L1,r2,r3,θA,θK);A 臂 p=(κ踝,κ膝,τ)+κ髋固定;B 臂 p=(κ踝,κ髋,τ),κ膝=0。"""
    L1, r2, r3, tA, tK = geom
    if arm == "A":
        ka, kk, tau, kh = p[0], p[1], p[2], p[3]
    else:
        ka, kk, tau, kh = p[0], 1e-6, p[2], p[1]
    return [L1, r2, r3, float(ka), float(kk), float(kh), float(tau), tA, tK]


def loguni(rng, rg, n):
    lo, hi = np.log10(rg[0]), np.log10(rg[1])
    return 10 ** (lo + (hi - lo) * rng.random(n))


def rod_geometry(geom, kap_hip):
    """报连杆的几何健康度:与股骨间隙、耦合比 dθ膝/dθ髋(平面,触地位形)。"""
    from scipy.optimize import brentq
    L1, r2, r3, tA, tK = geom
    l1 = L1; l2 = r2 * L1; l3 = r3 * L1
    a1 = np.radians(50.); a2 = a1 + np.radians(180 - tA); a3 = a2 - np.radians(180 - tK)
    dv = lambda a: np.array([np.cos(a), np.sin(a)])
    Pf = np.array([0., 0.]); A = Pf + l1 * dv(a1); K = A + l2 * dv(a2); H = K + l3 * dv(a3)
    e = lambda a: np.array([-np.sin(a), np.cos(a)])
    nf = e(a3); nf = -nf if nf[0] > 0 else nf
    ns = e(a2); ns = -ns if ns[0] > 0 else ns
    Pb = H + nf * OFF_HIP * 1000; Q0 = K + ns * OFF_KNEE * 1000
    def seg(p1, p2, p3, p4):
        f = lambda p, a, b: (lambda t: np.hypot(*(p - (a + t * (b - a)))))(
            np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1))
        return min(f(p1, p3, p4), f(p2, p3, p4), f(p3, p1, p2), f(p4, p1, p2))
    L0 = np.hypot(*(Q0 - Pb))
    R = lambda t: np.array([[np.cos(t), -np.sin(t)], [np.sin(t), np.cos(t)]])
    def Qof(th, tk):
        Kn = H + R(th) @ (K - H); sd = R(th + tk) @ (A - K)
        n = e(np.arctan2(sd[1], sd[0])); n = -n if n[0] > 0 else n
        return Kn + n * OFF_KNEE * 1000
    d = np.radians(1.0)
    try:
        tk = brentq(lambda t: np.hypot(*(Qof(d, t) - Pb)) - L0, -np.radians(12), np.radians(12))
        ratio = tk / d
    except Exception:
        ratio = float("nan")
    return dict(rod_mm=float(L0), clear_femur_mm=float(seg(K, H, Pb, Q0)),
                clear_shank_mm=float(seg(A, K, Pb, Q0)), ratio=float(ratio))


def search(ex, geom, kap_hip, arm, m, rng, n1, n2, fair_hip=False):
    """两阶段搜索:每工况返回最优 (score, g, ok, why, params)。"""
    RG = (KA_RG, KK_RG, TAU_RG) if arm == "A" else (KA_RG, KH_RG, TAU_RG)
    jobs, tag = [], []
    for ci, (kc, v0) in enumerate(CONDS):
        for p0, p1, p2 in zip(loguni(rng, RG[0], n1), loguni(rng, RG[1], n1),
                              loguni(rng, RG[2], n1)):
            kh = (float(loguni(rng, KH_RG, 1)[0])
                  if (fair_hip and arm == "A") else kap_hip)
            pp = (float(p0), float(p1), float(p2), kh)
            jobs.append((assemble(geom, arm, pp), m, v0, kc, arm == "B"))
            tag.append((ci, pp))
    r1 = list(ex.map(_ev, jobs, chunksize=1))
    jobs2, tag2 = [], []
    for ci, (kc, v0) in enumerate(CONDS):
        cand = sorted([(r1[i][0], tag[i][1]) for i in range(len(tag)) if tag[i][0] == ci],
                      key=lambda t: t[0])[:3]
        for _, pp in cand:
            for _ in range(max(1, n2 // 3)):
                f = lambda v, rg: float(np.clip(v * 10 ** rng.normal(0, 0.12), rg[0], rg[1]))
                q = (f(pp[0], RG[0]), f(pp[1], RG[1]), f(pp[2], RG[2]),
                     f(pp[3], KH_RG) if (fair_hip and arm == "A") else kap_hip)
                jobs2.append((assemble(geom, arm, q), m, v0, kc, arm == "B"))
                tag2.append((ci, q))
    r2 = list(ex.map(_ev, jobs2, chunksize=1))
    best = []
    for ci in range(len(CONDS)):
        allc = ([(r1[i], tag[i][1]) for i in range(len(tag)) if tag[i][0] == ci] +
                [(r2[i], tag2[i][1]) for i in range(len(tag2)) if tag2[i][0] == ci])
        r, pp = min(allc, key=lambda t: t[0][0])
        best.append(dict(kc=CONDS[ci][0], v0=CONDS[ci][1], g=r[1], ok=r[2],
                         why=r[3], params=list(pp)))
    return best


def robust(ex, geom, arm, m, cand_params, good_ci):
    """把每工况最优当候选,逐个跨全部工况互评 → 找一组打天下的固定设定。"""
    jobs = [(assemble(geom, arm, pp), m, v0, kc, arm == "B")
            for pp in cand_params for (kc, v0) in CONDS]
    res = list(ex.map(_ev, jobs, chunksize=1))
    rows = []
    for i, pp in enumerate(cand_params):
        rr = res[i * len(CONDS):(i + 1) * len(CONDS)]
        okg = [rr[j][1] for j in good_ci if rr[j][2] and rr[j][1]]
        rows.append(dict(params=list(pp), cover=sum(1 for j in good_ci if rr[j][2]),
                         n=len(good_ci), g_med=(float(np.median(okg)) if okg else None),
                         g_worst=(float(np.max(okg)) if okg else None)))
    rows.sort(key=lambda r: (-r["cover"], r["g_worst"] or 99))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v21_e5_bio/cvae_r39.pt")
    ap.add_argument("--masses", default="2,12")
    ap.add_argument("--n1", type=int, default=32)
    ap.add_argument("--n2", type=int, default=24)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--fair-hip", action="store_true",
                    help="让 A 臂也搜 κ髋(4 旋钮),作为对 A 有利的稳健性检查")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="outputs/v2_p4c")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import torch
    model, meta = load_cvae(a.ckpt); pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])

    def cvae_x(m, v0, kc):
        c = np.array([np.log10(m), v0, np.log10(kc), GCAP, SMAX]); torch.manual_seed(7)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32), 64).numpy()
        return prior.expand(np.clip(u, 0, 1), m).mean(0)

    rng = np.random.default_rng(a.seed); OUT = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for m in [float(v) for v in a.masses.split(",")]:
            xr = cvae_x(m, 1.2, 1e5)
            geom = (xr[0], xr[1], xr[2], xr[7], xr[8]); kap_hip = float(xr[5])
            rg = rod_geometry(geom, kap_hip)
            print(f"\n===== m = {m:g} kg  L1={geom[0]:.1f} θ={geom[3]:.0f}/{geom[4]:.0f} =====")
            print(f"  连杆几何: 长 {rg['rod_mm']:.0f} mm  股骨间隙 {rg['clear_femur_mm']:.0f} mm  "
                  f"小腿间隙 {rg['clear_shank_mm']:.0f} mm  耦合比 {rg['ratio']:+.2f}", flush=True)
            if rg["clear_femur_mm"] < 12:
                print("  ⚠ 间隙不足 12 mm,几何不可造,结果仅供参考", flush=True)
            res = {}
            for arm, nm in (("A", "独立(3自由度)"), ("B", "耦合(2自由度)")):
                res[arm] = search(ex, geom, kap_hip, arm, m, rng, a.n1, a.n2, a.fair_hip)
                print(f"  [{nm}] 逐工况搜索完成", flush=True)
            good = [i for i, c in enumerate(CONDS)
                    if not (m == 12.0 and c == (1e6, 1.8))]     # 要求物理不可能的那格
            rob = {}
            for arm in ("A", "B"):
                cand = [b["params"] for b in res[arm]]
                rob[arm] = robust(ex, geom, arm, m, cand, good)
                print(f"  [{arm}] 鲁棒被动互评完成", flush=True)

            print(f"\n  {'kc':>8}{'v0':>5} │{'A独立 g':>9}{'B耦合 g':>9}{'Δ':>8} │ A/B 判定")
            for i in good:
                A, B = res["A"][i], res["B"][i]
                dv = (f"{(B['g']-A['g'])/A['g']*100:+6.1f}%"
                      if (A["g"] and B["g"] and A["ok"] and B["ok"]) else "    —  ")
                f = lambda d: "   —  " if d["g"] is None else f"{d['g']:6.2f}"
                print(f"  {CONDS[i][0]:>8.0e}{CONDS[i][1]:>5.1f} │{f(A):>9}{f(B):>9}{dv:>8} │  "
                      f"{'✓' if A['ok'] else '✗'}/{'✓' if B['ok'] else '✗'}")
            cA = sum(1 for i in good if res["A"][i]["ok"]); cB = sum(1 for i in good if res["B"][i]["ok"])
            print(f"\n  ① 按工况整定的覆盖:  A 独立 {cA}/{len(good)}   B 耦合 {cB}/{len(good)}")
            gA = [res["A"][i]["g"] for i in good if res["A"][i]["ok"] and res["B"][i]["ok"]]
            gB = [res["B"][i]["g"] for i in good if res["A"][i]["ok"] and res["B"][i]["ok"]]
            if gA:
                print(f"     双可行处峰值中位: A {np.median(gA):.2f}g  B {np.median(gB):.2f}g  "
                      f"→ 耦合代价 {(np.median(gB)-np.median(gA))/np.median(gA)*100:+.1f}%")
            print(f"  ② 鲁棒被动(一组设定打天下):")
            for arm, nm in (("A", "独立"), ("B", "耦合")):
                r0 = rob[arm][0]
                print(f"     {nm}  覆盖 {r0['cover']}/{r0['n']}  "
                      f"g中位 {r0['g_med'] if r0['g_med'] is None else f'{r0[chr(103)+chr(95)+chr(109)+chr(101)+chr(100)]:.2f}'}  "
                      f"g最坏 {r0['g_worst'] if r0['g_worst'] is None else f'{r0[chr(103)+chr(95)+chr(119)+chr(111)+chr(114)+chr(115)+chr(116)]:.2f}'}")
            OUT[m] = dict(geom=list(geom), kap_hip=kap_hip, rod=rg,
                          per_cond=res, robust=rob, good_ci=good)
    json.dump(OUT, open(os.path.join(a.out, "p4c_coupled.json"), "w"), indent=1,
              ensure_ascii=False)
    print(f"\n[p4c] → {a.out}/p4c_coupled.json")


if __name__ == "__main__":
    main()

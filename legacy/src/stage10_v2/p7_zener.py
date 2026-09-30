# -*- coding: utf-8 -*-
"""P7 · 关节 Zener 化:给阻尼支路串一根弹簧,量它值多少。

现状每个关节是 Kelvin-Voigt(k ∥ c)。真实硬件里阻尼器的安装座有柔度,
等效于给阻尼支路串一根弹簧 → 标准线性固体(Zener):  k1 ∥ (k2 串 c)。
理论上(见 fig_拓扑对照):Zener 用约 1% 的峰值代价换掉触地瞬间的力跳变;
k2 → ∞ 时精确退化为并联。本实验在三自由度整腿上量这个代价到底多大。

实现:每个 Zener 关节加一个 ODE1 内部状态 y(Maxwell 阻尼器转角),
  ẏ = k2·(θ − y)/c ,  关节力矩 = k1·θ + k2·(θ − y)
用 Exudyn 的 NodeGenericODE1 + 关节的 springTorqueUserFunction 实现,
**不引入附加惯量**,因此没有伪高频模态。

自校验:脚本先在单自由度上跑 Zener,与解析/数值参考比对;
不通过则直接报错退出,绝不产出未经校验的结果。

用法:  python src/stage10_v2/p7_zener.py --out outputs/v2_p7
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np
from scipy.integrate import solve_ivp

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P
from bioprior import BioPrior
from factory_v2 import zeta_of_kc
from e17_emergent_b import load as load_cvae
from train_cvae import norm

GCAP, SMAX = 10 * 9.81, 0.024


# ---------------------------------------------------------------- 自校验(单自由度)
def sdof(model, k1, c, k2, m=12.0, v0=1.2, T=0.12):
    g = 9.81
    if model == "KV":
        f = lambda t, s: [s[1], g - (k1 * s[0] + c * s[1]) / m]
        F = lambda S: k1 * S[0] + c * S[1]; s0 = [0.0, v0]
    else:                                     # Zener: k1 ∥ (k2 串 c)
        def f(t, s):
            Fm = k2 * (s[0] - s[2])
            return [s[1], g - (k1 * s[0] + Fm) / m, Fm / c]
        F = lambda S: k1 * S[0] + k2 * (S[0] - S[2]); s0 = [0.0, v0, 0.0]
    r = solve_ivp(f, [0, T], s0, max_step=2e-5, rtol=1e-9, atol=1e-11, dense_output=True)
    t = np.linspace(0, T, 4000); S = r.sol(t)
    a = np.abs(g - F(S) / m) / g
    return float(a.max()), float(a[0]), float(S[0].max() * 1000)


def selfcheck():
    """三条必须成立的性质,任一不成立即判实现/理论不符。"""
    k1, c = 2.0e4, 293.0
    pk_kv, a0_kv, st_kv = sdof("KV", k1, c, None)
    ok = []
    # ① Zener 触地瞬间加速度 = 1.0 g(自由下落),并联则 > 1
    _, a0_z, _ = sdof("ZE", k1, c, 1.0e5)
    ok.append(("触地瞬间无跳变", abs(a0_z - 1.0) < 1e-6 and a0_kv > 1.5))
    # ② k2 → ∞ 时 Zener 峰值收敛到并联
    pk_big, _, _ = sdof("ZE", k1, c, 1.0e9)
    ok.append(("k2→∞ 退化为并联", abs(pk_big - pk_kv) / pk_kv < 0.005))
    # ③ 峰值随 k2 单调下降
    pks = [sdof("ZE", k1, c, k2)[0] for k2 in (3e4, 1e5, 3e5, 1e6, 1e7)]
    ok.append(("峰值随 k2 单调下降", all(pks[i] >= pks[i + 1] - 1e-9 for i in range(len(pks) - 1))))
    print("=== 自校验(单自由度 Zener) ===")
    for nm, v in ok:
        print(f"  {'✓' if v else '✗'} {nm}")
    print(f"  参考: 并联 峰值 {pk_kv:.3f} g,触地瞬间 {a0_kv:.3f} g;"
          f" Zener(k2=1e5) 触地瞬间 {a0_z:.3f} g")
    if not all(v for _, v in ok):
        raise SystemExit("[p7] 自校验未通过,拒绝产出结果。")
    return dict(kv_peak=pk_kv, kv_a0=a0_kv)


# ---------------------------------------------------------------- 三自由度整腿
def eval_zener(x9, m, v0, kc, k2_ratio, joints=("ankle", "knee", "hip")):
    """k2_ratio = k2/k1(每个 Zener 关节);None 表示纯并联(基线)。"""
    import exudyn as exu
    base = {**P.SCEN_BIRD_X, "hip_damp_unified": True}
    if k2_ratio is None:
        return P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=base)
    base = {**base, "zener": dict(ratio=float(k2_ratio), joints=tuple(joints))}
    return P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=base)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v22_e5_bio/cvae_r39.pt")
    ap.add_argument("--masses", default="5,12,30")
    ap.add_argument("--ratios", default="3,10,30,100,1000",
                    help="k2/k1;越大越接近并联")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_p7")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    ref = selfcheck()

    import torch
    model, meta = load_cvae(a.ckpt); pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
    lo, hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])

    def gen(m, v0, kc):
        c = np.array([np.log10(m), v0, np.log10(kc), GCAP, SMAX]); torch.manual_seed(7)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, lo, hi), dtype=torch.float32), 64).numpy()
        return prior.expand(np.clip(u, 0, 1), m).mean(0)

    if not hasattr(P, "ZENER_SUPPORTED"):
        print("\n⚠ physics_v2 尚未实现 zener 场景键 —— 只输出单自由度部分。")
        json.dump(dict(selfcheck=ref, note="3-DOF 部分待 physics_v2 支持"),
                  open(os.path.join(a.out, "p7_zener.json"), "w"), indent=1, ensure_ascii=False)
        return

    OUT = []
    print(f"\n=== 三自由度整腿:并联基线 vs Zener(k2/k1 扫描) ===")
    print(f"{'m':>5}{'地面':>6}{'k2/k1':>8}{'峰值g':>9}{'Δ vs 并联':>11}{'行程mm':>9}  判定")
    for m in [float(v) for v in a.masses.split(",")]:
        for tn, kc in (("草地", 1e5), ("硬地", 1e6)):
            x = gen(m, 1.2, kc)
            r0 = eval_zener(x, m, 1.2, kc, None)
            if r0 is None or r0.get("fail"):
                print(f"{m:>5.0f}{tn:>6}   基线失败"); continue
            p0 = r0["peak_a"] / 9.81
            ok0, _ = P.feasible_v2(r0, GCAP, SMAX)
            print(f"{m:>5.0f}{tn:>6}{'并联':>8}{p0:>9.2f}{'—':>11}{r0['leg_stroke_mm']:>9.1f}"
                  f"  {'✓' if ok0 else '✗'}")
            for rt in [float(v) for v in a.ratios.split(",")]:
                r = eval_zener(x, m, 1.2, kc, rt)
                if r is None or r.get("fail"):
                    print(f"{'':>11}{rt:>8.0f}   失败 {r.get('fail') if r else '?'}"); continue
                pk = r["peak_a"] / 9.81; ok, _ = P.feasible_v2(r, GCAP, SMAX)
                print(f"{'':>11}{rt:>8.0f}{pk:>9.2f}{(pk-p0)/p0*100:>+10.1f}%"
                      f"{r['leg_stroke_mm']:>9.1f}  {'✓' if ok else '✗'}")
                OUT.append(dict(m=m, terr=tn, ratio=rt, peak_g=pk, peak_base=p0,
                                stroke=r["leg_stroke_mm"], ok=bool(ok)))
    json.dump(dict(selfcheck=ref, rows=OUT),
              open(os.path.join(a.out, "p7_zener.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n[p7] → {a.out}/p7_zener.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""两种子比对:轨迹 + 通道使用 + 姿态收敛点 + b_eff,一次出可进汇报的口径。

纪律:训练类指标必须 ≥2 种子才报。本脚本把"哪些数转正、哪些仍需保留"直接算出来:
判据 = 臂间/版本间差是否大于种子间散布(噪声地板)。

用法:
  python src/stage10_v2/seed_compare.py --dirs outputs/v22_e5_bio outputs/v22_e5_bio_s1
  # 可选 --ref outputs/v21_e5_bio 加一列 v2.1 对照
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
from bioprior import BioPrior
from e17_emergent_b import load as load_cvae
from train_cvae import norm

GCAP, SMAX = 10 * 9.81, 0.024


def probe(ckpt):
    import torch
    model, meta = load_cvae(ckpt); pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
    lo, hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    mlo, mhi = 10 ** lo[0], 10 ** hi[0]

    def gen(m, v0, kc):
        c = np.array([np.log10(m), v0, np.log10(kc), GCAP, SMAX]); torch.manual_seed(7)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, lo, hi), dtype=torch.float32), 64).numpy()
        return prior.expand(np.clip(u, 0, 1), m).mean(0)

    mid = float(np.sqrt(mlo * mhi))
    X = np.array([gen(mid, v0, kc) for v0 in (1.2, 2.0) for kc in (1e6, 1e5, 5e4)])
    ms = np.geomspace(mlo * 1.15, mhi * 0.85, 6)
    b = float(np.polyfit(np.log10(ms), np.log10([gen(m, 1.2, 1e5)[0] for m in ms]), 1)[0])
    rr = lambda j: float(X[:, j].max() / X[:, j].min())
    return dict(m_range=(mlo, mhi), m_mid=mid, b_eff=b,
                thA=float(X[:, 7].mean()), thK=float(X[:, 8].mean()),
                thA_ptp=float(np.ptp(X[:, 7])), thK_ptp=float(np.ptp(X[:, 8])),
                r_ka=rr(3), r_kk=rr(4), r_kh=rr(5), r_tau=rr(6))


def traj(d, k=5):
    T = json.load(open(os.path.join(d, "trajectory.json")))
    t = T[-k:]
    return dict(n=len(T), gap=float(np.median([x["median_gap"] for x in t]) * 100),
                feas=float(np.mean([x["feas_rate"] for x in t]) * 100),
                cov=float(np.mean([x["coverage"] for x in t]) * 100),
                fail=int(sum(x["fail"] for x in t)),
                cross=next((x["round"] for x in T if x["median_gap"] < 0), None),
                last=float(T[-1]["median_gap"] * 100),
                trend=float((T[-1]["median_gap"] - T[-6]["median_gap"]) * 100))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True, help="同一版本的多个种子目录")
    ap.add_argument("--ref", default=None, help="可选:上一版本目录作对照")
    ap.add_argument("--ckpt", default="cvae_r39.pt")
    a = ap.parse_args()

    S = [(d, traj(d), probe(os.path.join(d, a.ckpt))) for d in a.dirs]
    names = [f"seed{i}" for i in range(len(S))]
    print(f"质量范围 {S[0][2]['m_range'][0]:.1f}–{S[0][2]['m_range'][1]:.1f} kg   "
          f"通道探针在 {S[0][2]['m_mid']:.1f} kg\n")

    rows = [("gap 中位 %", lambda t, p: t["gap"]), ("可行率 %", lambda t, p: t["feas"]),
            ("覆盖 %", lambda t, p: t["cov"]), ("越前沿轮", lambda t, p: t["cross"]),
            ("末5轮趋势 %", lambda t, p: t["trend"]),
            ("θ踝 °", lambda t, p: p["thA"]), ("θ膝 °", lambda t, p: p["thK"]),
            ("κ踝 跨工况×", lambda t, p: p["r_ka"]), ("τ 跨工况×", lambda t, p: p["r_tau"]),
            ("κ膝 跨工况×", lambda t, p: p["r_kk"]), ("κ髋 跨工况×", lambda t, p: p["r_kh"]),
            ("b_eff", lambda t, p: p["b_eff"])]
    hdr = "".join(f"{n:>10}" for n in names)
    print(f"{'指标':<16}{hdr}{'散布':>10}{'判定':>8}")
    verdict = {}
    for nm, f in rows:
        vs = [f(t, p) for _, t, p in S]
        if any(v is None for v in vs):
            print(f"{nm:<16}" + "".join(f"{str(v):>10}" for v in vs)); continue
        sp = max(vs) - min(vs)
        rel = abs(sp) / max(1e-9, abs(np.mean(vs))) * 100
        ok = rel < 15
        verdict[nm] = (float(np.mean(vs)), float(sp), ok)
        print(f"{nm:<16}" + "".join(f"{v:>10.3g}" for v in vs) +
              f"{sp:>10.3g}{'✓稳' if ok else '✗散':>8}")

    if a.ref:
        rt, rp = traj(a.ref), probe(os.path.join(a.ref, a.ckpt))
        print(f"\n对照 {a.ref}(质量 {rp['m_range'][0]:.1f}–{rp['m_range'][1]:.1f} kg):")
        print(f"  gap {rt['gap']:+.1f}%  覆盖 {rt['cov']:.1f}%  "
              f"θ {rp['thA']:.0f}/{rp['thK']:.0f}°  κ踝 {rp['r_ka']:.2f}×  τ {rp['r_tau']:.2f}×  "
              f"b_eff {rp['b_eff']:.3f}")
        print("  ⚠ gap/覆盖不可跨版本直接比(条件分布不同);可比的是姿态、通道比、b_eff。")

    print("\n=== 可进汇报的口径 ===")
    for nm, (mu, sp, ok) in verdict.items():
        print(f"  {nm:<16} {mu:.3g} ± {sp/2:.2g}" + ("" if ok else "   ← 散布过大,需第三个种子或更多轮"))
    t0 = S[0][1]
    if t0["trend"] < -0.5:
        print(f"\n⚠ 末5轮仍在下行({t0['trend']:+.1f}%/5轮) → 未收敛,gap 的绝对值偏保守,建议续跑 20 轮。")


if __name__ == "__main__":
    main()

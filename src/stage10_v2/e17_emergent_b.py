"""E17 · 涌现标度指数 b_eff:训练后的生成器实际产出什么样的"腿长-体重律"?

零仿真。对每个消融臂:
  沿 m 扫条件 → 生成设计 → 物理腿长 L1 = expand(u, m) → log-log 回归 → b_eff
读数:
  b_eff ≈ b_prior            → 循环没动腿长这维,先验被原样继承
  四臂 b_eff → 同一个 b*      → 物理自己选斜率;b* vs 生物 0.391 直接可比
  |b_eff − b_prior| 大        → 循环在用 u 硬扛先验的错;看 u 是否"贴墙"
表达上限:u∈±2.5σ 摊到 1–12 kg,斜率最多能掰 ±0.363。

用法: python src/stage10_v2/e17_emergent_b.py --dir /path/to/ckpts --out /tmp/e17
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
from bioprior import BioPrior, ARMS          # noqa: E402
from train_cvae import CVAE, norm            # noqa: E402

B_PRIOR = {"bio": 0.391, "geo": 1/3, "elastic": 0.25, "none": 0.0}


def load(fp):
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    m = CVAE(xd=ck["xd"], cd=len(meta["c_lo"]), z=ck["zdim"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, meta


def sweep(model, meta, prior, n_m=25, n_ctx=8, ngen=24, seed=0):
    """沿 m 对数扫,其它条件随机抽 n_ctx 组;返回 (m, L1, uL) 展平数组。"""
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    ms = 10 ** np.linspace(np.log10(1.0), np.log10(12.0), n_m)
    M, L1, UL = [], [], []
    for m in ms:
        for _ in range(n_ctx):
            c = np.array([np.log10(m),
                          rng.uniform(0.5, 2.0),                       # v0
                          rng.uniform(np.log10(5e4), np.log10(1e6)),   # log kc
                          rng.uniform(4, 25) * 9.81,                   # gcap
                          rng.uniform(0.008, 0.040)])                  # smax
            cn = torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32)
            U = model.sample(cn, ngen).numpy()
            X = prior.expand(np.clip(U, 0, 1), m)
            M += [m] * len(X); L1 += list(X[:, 0])
            UL += list((2 * np.clip(U[:, 0], 0, 1) - 1) * prior.u_max)
    return np.array(M), np.array(L1), np.array(UL)


def fit_b(m, L1):
    x, y = np.log10(m), np.log10(L1)
    b, a = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1]
    # bootstrap 置信
    rng = np.random.default_rng(1)
    bs = [np.polyfit(x[i], y[i], 1)[0]
          for i in (rng.integers(0, len(x), len(x)) for _ in range(400))]
    return float(b), float(a), float(r * r), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="outputs", help="含 v2_e5_{arm}/cvae_r20.pt 的根目录")
    ap.add_argument("--round", default="r20")
    ap.add_argument("--out", default="outputs/v2_e17")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    res = {}
    print(f"{'臂':<9}{'b_prior':>9}{'b_eff':>9}{'95%CI':>18}{'修正量':>9}"
          f"{'R²':>7}{'u贴墙率%':>10}")
    for arm in ("bio", "geo", "elastic", "none"):
        fp = os.path.join(args.dir, f"v2_e5_{arm}", f"cvae_{args.round}.pt")
        if not os.path.exists(fp):
            print(f"{arm:<9}  (缺 {fp})"); continue
        model, meta = load(fp)
        prior = BioPrior(arm, sigma=meta["prior"]["sigma"], u_max=meta["prior"]["u_max"])
        M, L1, UL = sweep(model, meta, prior)
        b, a, r2, lo, hi = fit_b(M, L1)
        # u 贴墙:|u_L| > 2.25 (=0.9×2.5σ) 的占比;再按体重两端分别看
        wall = float(np.mean(np.abs(UL) > 0.9 * prior.u_max))
        light = M < 2.0; heavy = M > 8.0
        wl = float(np.mean(np.abs(UL[light]) > 0.9 * prior.u_max))
        wh = float(np.mean(np.abs(UL[heavy]) > 0.9 * prior.u_max))
        # u 随 log m 的斜率(直接量"模型在用 u 补多少斜率")
        du = float(np.polyfit(np.log10(M), UL, 1)[0] * prior.sigma)
        res[arm] = dict(b_prior=B_PRIOR[arm], b_eff=b, ci=[lo, hi], r2=r2,
                        correction=b - B_PRIOR[arm], du_slope_logm=du,
                        wall_rate=wall, wall_light=wl, wall_heavy=wh,
                        uL_mean=float(UL.mean()), uL_std=float(UL.std()))
        print(f"{arm:<9}{B_PRIOR[arm]:>9.3f}{b:>9.3f}"
              f"{f'[{lo:.3f},{hi:.3f}]':>18}{b - B_PRIOR[arm]:>+9.3f}"
              f"{r2:>7.3f}{wall*100:>10.1f}")
        res[arm]["samples"] = dict(m=np.round(M, 3).tolist()[::12],
                                   L1=np.round(L1, 2).tolist()[::12])
    json.dump(res, open(os.path.join(args.out, "e17_emergent_b.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"\n[e17] 表达上限:|修正量| ≤ 0.363(u=±2.5σ 摊到 1–12 kg)")
    print(f"[e17] 生物实测 b = 0.391;各臂 b_eff 与它的距离:"
          + "  ".join(f"{a}:{abs(v['b_eff']-0.391):.3f}" for a, v in res.items()))
    print(f"[e17] → {os.path.join(args.out, 'e17_emergent_b.json')}")


if __name__ == "__main__":
    main()

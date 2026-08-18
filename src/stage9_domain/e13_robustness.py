"""E13 · 环境不确定性下的稳健性:生成设计 vs 逐工况优化设计,谁更抗扰?

动机(回应外部评审):此前的 ±0.9pp 只是**训练随机性**(算法层),
不足以支撑"under environmental uncertainty"的主张。本实验做真正的环境扰动:
腿按名义工况制造,实际世界与名义不符,看名义最优解还站不站得住。

扰动维度(6 维,各自独立;取值范围为本工作设定,已在下方声明依据):
  载荷 m      ±5%    机体配置/载荷变化
  下沉速度 v0 ±10%   着陆控制误差
  地面刚度 kc ±30%   沥青 vs 草地 vs 沙土(最大的未建模不确定性)
  摩擦 μ      ±0.2   绝对偏移(干/湿表面)
  关节刚度 k  ±10%   弹簧制造公差
  关节阻尼 c  ±20%   阻尼器公差(比弹簧差)

两臂对比(同考题、同扰动样本):
  gen  = 部署模型零在线仿真生成 40 个 → 名义下最优可行者
  bo9  = 逐工况贝叶斯优化 9 次仿真 → 名义下最优可行者
关注点:优化器常把解停在约束边界上,理论上更脆;生成模型来自"前沿样本族",
可能更靠内。这条若成立,是面向工业的实打实论据;若不成立,如实报负结果。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage9_domain/e13_robustness.py \
    --model outputs/gen_e5c_r85/cvae_s1.pt --refs outputs/gen_e5/refs.json \
    --out outputs/gen_e13 --npert 32 --workers 128
成本: 题数 ×(40 生成 + 9 BO + 2×npert 扰动)≈ 8.6 千次仿真,分钟级。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage6_surrogate"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage7_generative"))
import models as M                                       # noqa: E402
from hf_exudyn import exu_eval, SCEN_BIRD_X              # noqa: E402
from data_factory import KEYS                            # noqa: E402
from e5_loop import lhs                                  # noqa: E402
from eval_gen import bo9                                 # noqa: E402
from train_cvae import CVAE, norm                        # noqa: E402

iP, iS = KEYS.index("peak_a"), KEYS.index("stroke")

# 扰动半幅(相对,μ 为绝对偏移)
PERT = dict(m=0.05, v0=0.10, kc=0.30, mu=0.20, k=0.10, c=0.20)
PKEYS = ("m", "v0", "kc", "mu", "k", "c")


def perturbed_eval(args):
    """腿按名义制造(刚度阻尼由名义 m 定),环境按扰动实际值。"""
    x, m_nom, v0_nom, d = args
    x = np.asarray(x, float)
    sc = dict(M.bird_size_x({**SCEN_BIRD_X, "m": m_nom, "v0": v0_nom, "kappa": 4.0}, x))
    sc["m"] = m_nom * (1 + d[0])          # 实际载荷
    sc["v0"] = v0_nom * (1 + d[1])        # 实际下沉速度
    sc["kc"] = sc["kc"] * (1 + d[2])      # 地面刚度
    sc["cc"] = 0.01 * sc["kc"]
    sc["mu"] = float(np.clip(sc["mu"] + d[3], 0.1, 1.2))
    for kk in ("k_ankle", "k_knee", "k_hip"):
        sc[kk] *= (1 + d[4])
    for cc in ("c_ankle", "c_knee", "c_hip"):
        sc[cc] *= (1 + d[5])
    sc["k1"], sc["k2"] = sc["k_ankle"], sc["k_knee"]
    sc["c1"], sc["c2"] = sc["c_ankle"], sc["c_knee"]
    r = exu_eval(tuple(x[:3]), sc)
    return [None if not np.isfinite(r[k]) else float(r[k]) for k in KEYS]


def nominal_eval(args):
    x, m, v0 = args
    sc = M.bird_size_x({**SCEN_BIRD_X, "m": m, "v0": v0, "kappa": 4.0}, np.asarray(x, float))
    r = exu_eval(tuple(np.asarray(x, float)[:3]), sc)
    return [None if not np.isfinite(r[k]) else float(r[k]) for k in KEYS]


def arr(rows):
    return np.array([[np.nan if v is None else v for v in r] for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/gen_e5c_r85/cvae_s1.pt")
    ap.add_argument("--refs", default="outputs/gen_e5/refs.json")
    ap.add_argument("--out", default="outputs/gen_e13")
    ap.add_argument("--ngen", type=int, default=40)
    ap.add_argument("--npert", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=1301)
    ap.add_argument("--skip-bo", action="store_true")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    model = CVAE(xd=ck["xd"], cd=len(c_lo), z=ck["zdim"])
    model.load_state_dict(ck["state"]); model.eval()
    refs = json.load(open(args.refs))
    print(f"[e13] 考题 {len(refs)}  扰动样本 {args.npert}/设计  "
          f"扰动幅度 {PERT}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # ---- ① 生成臂:每题 40 候选 → 名义最优可行 ----
        jobs, spans = [], []
        for si, r in enumerate(refs):
            torch.manual_seed(args.seed + si)
            cn = torch.tensor(norm(np.array([r["m"], r["v0"], r["gcap"], r["smax"]]),
                                   c_lo, c_hi), dtype=torch.float32)
            Xg = x_lo + (x_hi - x_lo) * model.sample(cn, args.ngen).numpy()
            spans.append((len(jobs), len(jobs) + len(Xg), Xg))
            jobs += [(x, r["m"], r["v0"]) for x in Xg]
        Y = arr(list(ex.map(nominal_eval, jobs, chunksize=8)))
        nominal = {}
        for si, (a, b, Xg) in enumerate(spans):
            r = refs[si]; Ys = Y[a:b]
            ok = np.isfinite(Ys[:, iP]) & (Ys[:, iP] <= r["gcap"]) & (Ys[:, iS] <= r["smax"])
            if ok.any():
                j = np.where(ok)[0][np.argmin(Ys[ok, iP])]
                nominal[("gen", si)] = (Xg[j], float(Ys[j, iP]))
        print(f"[e13] 生成臂名义解 {len(nominal)}/{len(refs)} ({time.time() - t0:.0f}s)")

        # ---- ② BO-9 臂 ----
        if not args.skip_bo:
            for si, r in enumerate(refs):
                rng = np.random.default_rng(990_000 + si)
                Xb, Yb = bo9(ex, r["m"], r["v0"], r["gcap"], r["smax"], x_lo, x_hi, rng)
                Yb = np.asarray(Yb, float)
                ok = (np.isfinite(Yb[:, 0]) & (Yb[:, 0] <= r["gcap"])
                      & (Yb[:, 1] <= r["smax"]))
                if ok.any():
                    j = np.where(ok)[0][np.argmin(Yb[ok, 0])]
                    nominal[("bo9", si)] = (np.asarray(Xb)[j], float(Yb[j, 0]))
            print(f"[e13] BO-9 臂完成 ({time.time() - t0:.0f}s)")

        # ---- ③ 同一批扰动环境作用于两臂 ----
        prng = np.random.default_rng(args.seed + 77)
        Dp = 2 * lhs(args.npert, len(PKEYS), prng) - 1.0          # [-1,1]^6 LHS
        Dp = Dp * np.array([PERT[k] for k in PKEYS])
        pj, ptag = [], []
        for (arm, si), (x, _) in nominal.items():
            r = refs[si]
            for q in range(args.npert):
                pj.append((x, r["m"], r["v0"], Dp[q]))
                ptag.append((arm, si, q))
        Yp = arr(list(ex.map(perturbed_eval, pj, chunksize=8)))
        print(f"[e13] 扰动仿真 {len(pj)} 次完成 ({time.time() - t0:.0f}s)")

    bucket = {}
    for (arm, si, q), y in zip(ptag, Yp):
        bucket.setdefault((arm, si), []).append(y)

    rows = []
    for (arm, si), ys in bucket.items():
        r = refs[si]; Ys = np.array(ys)
        ok = (np.isfinite(Ys[:, iP]) & (Ys[:, iP] <= r["gcap"])
              & (Ys[:, iS] <= r["smax"]))
        pk = Ys[np.isfinite(Ys[:, iP]), iP]
        nom = nominal[(arm, si)][1]
        rows.append(dict(arm=arm, sc=si, m=r["m"], v0=r["v0"],
                         nominal_g=nom / 9.81,
                         retention=float(ok.mean()),
                         p50_g=float(np.median(pk)) / 9.81 if len(pk) else None,
                         p90_g=float(np.quantile(pk, 0.9)) / 9.81 if len(pk) else None,
                         max_g=float(pk.max()) / 9.81 if len(pk) else None,
                         degrade_p90=float(np.quantile(pk, 0.9) / nom - 1) if len(pk) else None))

    summ = {}
    for arm in sorted({r["arm"] for r in rows}):
        v = [r for r in rows if r["arm"] == arm]
        summ[arm] = dict(
            n=len(v),
            retention_mean=float(np.mean([r["retention"] for r in v])),
            retention_p10=float(np.quantile([r["retention"] for r in v], 0.1)),
            n_all_feasible=int(sum(1 for r in v if r["retention"] >= 0.999)),
            nominal_g_median=float(np.median([r["nominal_g"] for r in v])),
            p90_g_median=float(np.median([r["p90_g"] for r in v if r["p90_g"]])),
            degrade_p90_median=float(np.median([r["degrade_p90"] for r in v
                                                if r["degrade_p90"] is not None])))
    json.dump(dict(perturbation=PERT, npert=args.npert,
                   summary=summ, rows=rows),
              open(os.path.join(args.out, "e13_results.json"), "w"),
              indent=2, ensure_ascii=False)

    print("\n== E13 环境扰动稳健性 ==")
    print(f"{'臂':<6}{'题数':>5}{'保持可行率':>12}{'全扰动通过':>12}"
          f"{'名义峰值':>10}{'P90峰值':>10}{'P90劣化':>10}")
    for arm, s in summ.items():
        print(f"{arm:<6}{s['n']:>5}{s['retention_mean'] * 100:>11.1f}%"
              f"{s['n_all_feasible']:>8}/{s['n']:<3}"
              f"{s['nominal_g_median']:>9.2f}g{s['p90_g_median']:>9.2f}g"
              f"{s['degrade_p90_median'] * 100:>9.1f}%")
    print(f"[e13] done → {args.out}/e13_results.json")


if __name__ == "__main__":
    main()

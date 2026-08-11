"""E1 评测:留出工况上,cVAE 生成 vs 三个基线,Exudyn 实判。

四个选手(部署时每工况仿真预算):
  gen    cVAE 生成 40 个设计,0 次训练外仿真(摊销)——验证用仿真不计入预算
  lhs9   随机 LHS 9 个设计取最好(蓝本预算的下界基线)
  warm   标度律暖启动:L1=111.2·(m/10)^0.45 + 比例中位,9 个扰动取最好
  bo9    GPR+EI 贝叶斯优化 9 次调用(蓝本式)
裁判:每工况新采 150 设计 LHS 得参考前沿(仅评测用,不给选手看)。

指标:可行率 | 最优差距(可行内最小 peak_a 相对参考前沿最优) | 族覆盖度(生成可行解
行程跨度 / 参考前沿行程跨度)。

用法(A100): python src/stage7_generative/eval_gen.py \
  --model outputs/gen_model/cvae.pt --data outputs/gen_data/gen_dataset.npz \
  --out outputs/gen_eval --workers 32 [--nsc 20]
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage6_surrogate"))
import models as M                      # noqa: E402
from hf_exudyn import exu_eval, SCEN_BIRD_X  # noqa: E402
from train_cvae import CVAE, norm       # noqa: E402

KAPPA = 4.0


def _eval_one(args):
    x, m, v0 = args
    sc = M.bird_size({**SCEN_BIRD_X, "m": m, "v0": v0, "kappa": KAPPA}, x)
    r = exu_eval(tuple(x), sc)
    return r["peak_a"], r["stroke"]


def lhs(n, d, rng):
    X = np.empty((n, d))
    for j in range(d):
        e = (np.arange(n) + rng.random(n)) / n
        X[:, j] = rng.permutation(e)
    return X


def evals(ex, X, m, v0):
    return np.array(list(ex.map(_eval_one, [(x, m, v0) for x in X], chunksize=1)))


def best_feasible(Y, gcap, smax):
    ok = np.isfinite(Y[:, 0]) & (Y[:, 0] <= gcap) & (Y[:, 1] <= smax)
    return (float(Y[ok, 0].min()) if ok.any() else np.nan), ok


def bo9(ex, m, v0, gcap, smax, lo, hi, rng):
    """GPR+EI,9 次仿真:4 初始 + 5 迭代;目标 = peak_a + 约束罚。"""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel
    X = lo + (hi - lo) * lhs(4, 3, rng)
    Y = evals(ex, X, m, v0)
    def pen(y):
        if not np.isfinite(y[0]): return 1e4
        return y[0] + 1e3 * max(0, y[0] - gcap) / gcap + 1e3 * max(0, y[1] - smax) / smax
    for _ in range(5):
        t = np.array([pen(y) for y in Y])
        g = GaussianProcessRegressor(Ck(1.0) * RBF([0.3] * 3) + WhiteKernel(1e-4),
                                     normalize_y=True).fit(norm(X, lo, hi), t)
        cand = norm(lo + (hi - lo) * lhs(400, 3, rng), lo, hi)
        mu, sd = g.predict(cand, return_std=True)
        best = t.min()
        from scipy.stats import norm as N
        z = (best - mu) / (sd + 1e-9)
        ei = (best - mu) * N.cdf(z) + sd * N.pdf(z)
        xn = cand[np.argmax(ei)]
        xnew = lo + (hi - lo) * xn
        X = np.vstack([X, xnew]); Y = np.vstack([Y, evals(ex, [xnew], m, v0)])
    return X, Y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/gen_model/cvae.pt")
    ap.add_argument("--data", default="outputs/gen_data/gen_dataset.npz")
    ap.add_argument("--out", default="outputs/gen_eval")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--ngen", type=int, default=40)
    ap.add_argument("--nref", type=int, default=150)
    ap.add_argument("--nsc", type=int, default=20, help="评测场景数上限")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = CVAE(); model.load_state_dict(ck["state"]); model.eval()
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    lo, hi = np.array(M.LO_BIRD), np.array(M.HI_BIRD)

    C_te = np.load(args.data)["C_te"][:args.nsc]
    print(f"[eval] scenarios: {len(C_te)}")
    R = dict(gen=[], lhs9=[], warm=[], bo9=[])
    cov = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for si, row in enumerate(C_te):
            m, v0, gcap, smax = row[:4]
            rng = np.random.default_rng(500_000 + si)
            # 裁判:参考前沿
            Yr = evals(ex, lo + (hi - lo) * lhs(args.nref, 3, rng), m, v0)
            ref, okr = best_feasible(Yr, gcap, smax)
            if not np.isfinite(ref):
                print(f"  sc{si}: 参考集无可行解,跳过"); continue
            ref_span = np.ptp(Yr[okr, 1]) if okr.sum() > 1 else 0.0
            # ① 生成
            cn = torch.tensor(norm(np.array([m, v0, gcap, smax]), c_lo, c_hi),
                              dtype=torch.float32)
            Xg = x_lo + (x_hi - x_lo) * model.sample(cn, args.ngen).numpy()
            Yg = evals(ex, Xg, m, v0)
            bg, okg = best_feasible(Yg, gcap, smax)
            feas_rate = float(okg.mean())
            span = np.ptp(Yg[okg, 1]) if okg.sum() > 1 else 0.0
            cov.append(span / ref_span if ref_span > 0 else np.nan)
            R["gen"].append(dict(sc=si, best=bg, ref=ref, feas=feas_rate))
            # ② LHS-9
            Y9 = evals(ex, lo + (hi - lo) * lhs(9, 3, rng), m, v0)
            b9, _ = best_feasible(Y9, gcap, smax)
            R["lhs9"].append(dict(sc=si, best=b9, ref=ref))
            # ③ 标度律暖启动 + 8 扰动
            xw = np.array([np.clip(111.2 * (m / 10) ** 0.45, lo[0], hi[0]), 1.77, 1.08])
            Xw = np.clip(xw + rng.normal(0, [8, 0.1, 0.08], (9, 3)), lo, hi); Xw[0] = xw
            Yw = evals(ex, Xw, m, v0)
            bw, _ = best_feasible(Yw, gcap, smax)
            R["warm"].append(dict(sc=si, best=bw, ref=ref))
            # ④ BO-9
            _, Yb = bo9(ex, m, v0, gcap, smax, lo, hi, rng)
            bb, _ = best_feasible(Yb, gcap, smax)
            R["bo9"].append(dict(sc=si, best=bb, ref=ref))
            print(f"  sc{si} m={m:.1f} v0={v0:.2f} gcap={gcap/9.81:.1f}g smax={smax*1e3:.0f}mm | "
                  f"ref={ref/9.81:.2f}g gen={bg/9.81 if np.isfinite(bg) else -1:.2f} "
                  f"lhs9={b9/9.81 if np.isfinite(b9) else -1:.2f} "
                  f"warm={bw/9.81 if np.isfinite(bw) else -1:.2f} "
                  f"bo9={bb/9.81 if np.isfinite(bb) else -1:.2f}  ({time.time()-t0:.0f}s)")

    # 汇总:最优差距 =(best−ref)/ref;失败(无可行)计 100% 差距
    summ = {}
    for k, lst in R.items():
        gaps = [((r["best"] - r["ref"]) / r["ref"]) if np.isfinite(r["best"]) else 1.0
                for r in lst]
        summ[k] = dict(median_gap=float(np.median(gaps)), mean_gap=float(np.mean(gaps)),
                       fail=int(sum(1 for r in lst if not np.isfinite(r["best"]))),
                       n=len(lst))
    summ["gen"]["feas_rate_mean"] = float(np.mean([r["feas"] for r in R["gen"]]))
    summ["gen"]["coverage_mean"] = float(np.nanmean(cov))
    print("\n== 汇总(最优差距,越小越好)==")
    for k, v in summ.items():
        print(f"  {k:5s} median {v['median_gap']*100:5.1f}%  mean {v['mean_gap']*100:5.1f}%  "
              f"fail {v['fail']}/{v['n']}")
    print(f"  gen 可行率 {summ['gen']['feas_rate_mean']*100:.0f}%  "
          f"前沿覆盖 {summ['gen']['coverage_mean']*100:.0f}%")
    json.dump(dict(summary=summ, detail=R), open(os.path.join(args.out, "eval_results.json"), "w"),
              indent=2)
    print(f"[eval] wrote eval_results.json → {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

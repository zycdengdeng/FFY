"""多保真残差代理 · 样本效率实验(生成式代理框架第一块基石)。

问题:高保真(HF)评估昂贵(蓝本=9 次 ANSYS FE),黑盒 GPR 在小样本下学不好。
主张:用仿生连杆的解析动力学(LF,免费)当物理先验,代理只学「HF−线性校准LF」的残差,
      小样本精度大幅提升。

方法对比(同一 HF 训练预算 n):
  A. LF-linear   : y ≈ a·yLF+b(纯物理先验+2参数校准,基线)
  B. GPR-direct  : 黑盒 GPR 直接学 x→yHF(蓝本做法)
  C. MF-residual : y = a·yLF+b + GP(x) 学残差(本方案)
  D. LF-feature  : GPR 学 [x, yLF]→yHF(LF 当特征,另一常见 MF 变体)

指标:NRMSE(测试集 200 点,RMSE/std)。n ∈ {4,6,9,12,16,24,32}(9=蓝本预算),30 次重复。
用法:python src/stage6_surrogate/mf_experiment.py --out outputs/surrogate
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(__file__))
import models as M
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as Ck, WhiteKernel

NTEST, NS, REPS = 200, [4, 6, 9, 12, 16, 24, 32], 30
LO3 = np.array([M.BOUNDS["L1"][0], M.BOUNDS["r2"][0], M.BOUNDS["r3"][0]])
HI3 = np.array([M.BOUNDS["L1"][1], M.BOUNDS["r2"][1], M.BOUNDS["r3"][1]])
LO5 = np.concatenate([LO3, [2.0, 150.0]])     # v0 m/s, m kg
HI5 = np.concatenate([HI3, [4.0, 260.0]])
# 7 维:+触地姿态(θA, θK),范围取 Duong 12 段 4 物种实测(113-160°, 118-158°)
LO7 = np.concatenate([LO5, [113.0, 118.0]])
HI7 = np.concatenate([HI5, [160.0, 158.0]])
DIM = 3
def _bounds():
    return {3: (LO3, HI3), 5: (LO5, HI5), 7: (LO7, HI7)}[DIM]
def unit(X):
    lo, hi = _bounds()
    return (X - lo) / (hi - lo)
def lhs_d(n, seed):
    rng = np.random.default_rng(seed)
    lo, hi = _bounds()
    out = np.empty((n, len(lo)))
    for j in range(len(lo)):
        e = lo[j] + (hi[j] - lo[j]) * (np.arange(n) + rng.random(n)) / n
        out[:, j] = rng.permutation(e)
    return out


def gp():
    k = Ck(1.0, (1e-3, 1e3)) * RBF([0.3] * DIM, (1e-2, 1e2)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2, random_state=0)


def gp4():
    k = Ck(1.0, (1e-3, 1e3)) * RBF([0.3] * (DIM + 1), (1e-2, 1e2)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    return GaussianProcessRegressor(kernel=k, normalize_y=True, n_restarts_optimizer=2, random_state=0)


HFMODE = "2dof"
LFMODE = "1dof"

def _exu_one(args):
    x, key = args
    from hf_exudyn import exu_eval, SCEN_X
    sc = dict(SCEN_X)
    if len(x) >= 5:
        sc["v0"], sc["m"] = x[3], x[4]
    if len(x) == 7:
        sc["thetaA"], sc["thetaK"] = np.radians(x[5]), np.radians(x[6])
    return exu_eval(tuple(x[:3]), sc)[key]

def evals(X, key):
    lf = []
    lf_fn = M.lf_eval if LFMODE == "1dof" else M.hf_eval   # 2dof 作 LF = 更像的物理先验
    for x in X:
        sc = dict(M.SCEN)
        if DIM >= 5:
            sc["v0"], sc["m"] = x[3], x[4]
        if DIM == 7:
            sc["thetaA"], sc["thetaK"] = np.radians(x[5]), np.radians(x[6])
        lf.append(lf_fn(x[:3], sc)[key])
    if HFMODE == "exudyn":
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor(max_workers=8) as ex:
            hf = list(ex.map(_exu_one, [(x, key) for x in X], chunksize=2))
    else:
        hf = []
        for x in X:
            sc = dict(M.SCEN)
            if DIM >= 5:
                sc["v0"], sc["m"] = x[3], x[4]
            if DIM == 7:
                sc["thetaA"], sc["thetaK"] = np.radians(x[5]), np.radians(x[6])
            hf.append(M.hf_eval(x[:3], sc)[key])
    return np.array(lf), np.array(hf)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="outputs/surrogate")
    ap.add_argument("--key", default="peak_a"); ap.add_argument("--dim", type=int, default=3)
    ap.add_argument("--hf", default="2dof", choices=["2dof", "exudyn"])
    ap.add_argument("--lf", default="1dof", choices=["1dof", "2dof"])
    ap.add_argument("--ntest", type=int, default=200); ap.add_argument("--reps", type=int, default=30)
    args = ap.parse_args()
    global DIM, HFMODE, LFMODE, NTEST, REPS
    DIM = args.dim; HFMODE = args.hf; LFMODE = args.lf
    NTEST = args.ntest; REPS = args.reps
    os.makedirs(args.out, exist_ok=True); t0 = time.time()

    Xte = lhs_d(NTEST, seed=999)
    lf_te, hf_te = evals(Xte, args.key)
    ok = np.isfinite(hf_te); Xte, lf_te, hf_te = Xte[ok], lf_te[ok], hf_te[ok]
    sd = hf_te.std()
    print(f"[mf] test set {ok.sum()} pts  HF {hf_te.min()/9.81:.1f}-{hf_te.max()/9.81:.1f} g  ({time.time()-t0:.0f}s)")

    res = {m: {n: [] for n in NS} for m in ["LF-linear", "GPR-direct", "MF-residual", "LF-feature"]}
    for rep in range(REPS):
        Xpool = lhs_d(max(NS), seed=1000 + rep)
        lf_tr_all, hf_tr_all = evals(Xpool, args.key)
        okt = np.isfinite(hf_tr_all)
        for n in NS:
            idx = np.where(okt)[0][:n]
            if len(idx) < n: continue
            Xtr, lft, hft = Xpool[idx], lf_tr_all[idx], hf_tr_all[idx]
            A = np.vstack([lft, np.ones(n)]).T
            coef, *_ = np.linalg.lstsq(A, hft, rcond=None)
            lin_te = coef[0] * lf_te + coef[1]
            res["LF-linear"][n].append(np.sqrt(np.mean((lin_te - hf_te) ** 2)) / sd)
            g = gp().fit(unit(Xtr), hft)
            res["GPR-direct"][n].append(np.sqrt(np.mean((g.predict(unit(Xte)) - hf_te) ** 2)) / sd)
            r = gp().fit(unit(Xtr), hft - (coef[0] * lft + coef[1]))
            pred = lin_te + r.predict(unit(Xte))
            res["MF-residual"][n].append(np.sqrt(np.mean((pred - hf_te) ** 2)) / sd)
            f4 = np.column_stack([unit(Xtr), (lft - lft.mean()) / (lft.std() + 1e-9)])
            t4 = np.column_stack([unit(Xte), (lf_te - lft.mean()) / (lft.std() + 1e-9)])
            g4m = gp4().fit(f4, hft)
            res["LF-feature"][n].append(np.sqrt(np.mean((g4m.predict(t4) - hf_te) ** 2)) / sd)
        if (rep + 1) % 10 == 0:
            print(f"  rep {rep+1}/{REPS}  ({time.time()-t0:.0f}s)")

    summ = {m: {n: [float(np.mean(v)), float(np.std(v))] for n, v in d.items() if v} for m, d in res.items()}
    json.dump({"metric": "NRMSE", "key": args.key, "hf": args.hf, "lf": args.lf, "n_grid": NS, "reps": REPS,
               "corr_lf_hf": float(np.corrcoef(lf_te, hf_te)[0, 1]), "summary": summ},
              open(os.path.join(args.out, "mf_results.json"), "w"), indent=2)

    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    colors = {"LF-linear": "#999", "GPR-direct": "#3A6EA5", "MF-residual": "#7F2D32", "LF-feature": "#2C7A3F"}
    for mname, d in summ.items():
        ns = sorted(d); mu = [d[n][0] for n in ns]; sg = [d[n][1] for n in ns]
        ax[0].plot(ns, mu, "-o", color=colors[mname], label=mname, lw=2, ms=4)
        ax[0].fill_between(ns, np.array(mu) - sg, np.array(mu) + sg, color=colors[mname], alpha=.12)
    ax[0].axvline(9, color="k", ls=":", lw=1); ax[0].text(9.2, ax[0].get_ylim()[1] * .95, "blueprint budget n=9", fontsize=8)
    ax[0].set_xlabel("HF training samples n"); ax[0].set_ylabel("NRMSE (test)"); ax[0].set_yscale("log")
    ax[0].legend(fontsize=9); ax[0].grid(alpha=.3, which="both")
    ax[0].set_title("Sample efficiency: physics-prior residual GP vs black-box GP")
    ax[1].scatter(lf_te, hf_te, s=12, alpha=.5, color="#3A6EA5")
    ax[1].set_xlabel("LF peak_a (analytic 1-DOF)"); ax[1].set_ylabel("HF peak_a (2-DOF nonlinear)")
    ax[1].set_title(f"LF vs HF  (corr={np.corrcoef(lf_te,hf_te)[0,1]:.3f})"); ax[1].grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "sample_efficiency.png"), dpi=115); plt.close()
    print(f"[mf] wrote mf_results.json + sample_efficiency.png → {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

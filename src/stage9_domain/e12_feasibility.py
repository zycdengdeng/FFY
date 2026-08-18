"""E12 · 可行域分类器:显式学习 P(可行 | 设计 x, 工况 c)。

动机(回应外部评审):此前 cVAE 学的是"前沿样本上的条件分布 p(x|c)",
并未显式建模可行域。本实验用已有的 95 万条**带真值标签**的池子样本,
零新增仿真地训练一个可行性分类器,使"学习可行域"成为可核验的事实而非主张。

边界声明(重要):
  · 本分类器**不进入 E5 自提升循环**——真值裁判始终是 Exudyn,不被学习模型替代;
  · 它有两个正当用途:(a) 分析仪器:把可行域及其随工况的演化定量画出来;
    (b) 部署期预筛:先生成 N 个、按 P(可行) 留前 K 个再送仿真验证,
       仿真预算与口径不变,只是把预算花在更可能成功的候选上。

数据构造:池中每个设计已知 (m, v0) 与全指标;可行性依赖 (g_cap, s_max),
故对每个设计抽 K 组约束 → 标签 = 峰值≤g_cap ∧ 行程≤s_max ∧ 指标有限(未塌陷)。
划分按**工况 cid** 而非随机,测试集用与生成实验一致的留出 cid,考的是
"没见过的工况上还准不准"。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage9_domain/e12_feasibility.py \
    --factory outputs/gen_data7/factory.jsonl --data outputs/gen_data7/gen_dataset.npz \
    --inc outputs/gen_e5/pool_increments.jsonl --out outputs/gen_e12
产出: feas_clf.pt(分类器) + e12_results.json(精度/标定/可行体积图谱数据)
成本: 零仿真;CPU 训练分钟级。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage7_generative"))
from build_dataset import GCAP_RANGE, SMAX_RANGE           # noqa: E402
from data_factory import KEYS                              # noqa: E402
from e5_loop import load_pools                             # noqa: E402
from train_cvae import norm                                # noqa: E402

iP, iS = KEYS.index("peak_a"), KEYS.index("stroke")


class FeasNet(nn.Module):
    """P(可行 | x, c):小 MLP,与 cVAE 同量级容量,避免"靠模型体量取胜"的质疑。"""

    def __init__(self, xd=7, cd=4, h=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(xd + cd, h), nn.SiLU(),
            nn.Linear(h, h), nn.SiLU(),
            nn.Linear(h, 1))

    def forward(self, x, c):
        return self.net(torch.cat([x, c], -1)).squeeze(-1)      # logit

    @torch.no_grad()
    def prob(self, X, C):
        return torch.sigmoid(self.forward(X, C)).numpy()


def build_examples(pools, lo, hi, kdraw, seed, max_ex):
    """池 → (x_norm, c_norm, label)。按 cid 记录归属,便于按工况划分。"""
    Xs, Cs, Ys, Gs = [], [], [], []
    for p in pools.values():
        rng = np.random.default_rng(seed + p.cid)
        okfin = np.isfinite(p.Y[:, iP])                 # 非塌陷(NaN 视为不可行)
        for _ in range(kdraw):
            gcap = rng.uniform(*GCAP_RANGE) * 9.81
            smax = rng.uniform(*SMAX_RANGE)
            lab = okfin & (p.Y[:, iP] <= gcap) & (p.Y[:, iS] <= smax)
            n = len(p.X)
            Xs.append(p.X)
            Cs.append(np.tile([p.m, p.v0, gcap, smax], (n, 1)))
            Ys.append(lab.astype(np.float32))
            Gs.append(np.full(n, p.cid))
    X = np.vstack(Xs); C = np.vstack(Cs)
    Y = np.concatenate(Ys); G = np.concatenate(Gs)
    if len(X) > max_ex:                                  # 均匀下采样,控内存/时长
        idx = np.random.default_rng(seed).choice(len(X), max_ex, replace=False)
        X, C, Y, G = X[idx], C[idx], Y[idx], G[idx]
    return norm(X, lo, hi).astype(np.float32), C.astype(np.float32), Y, G


def roc_auc(y, s):
    """秩法 AUC,免 sklearn 依赖。"""
    o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/gen_data7/factory.jsonl")
    ap.add_argument("--data", default="outputs/gen_data7/gen_dataset.npz")
    ap.add_argument("--inc", default="outputs/gen_e5/pool_increments.jsonl")
    ap.add_argument("--out", default="outputs/gen_e12")
    ap.add_argument("--kdraw", type=int, default=2, help="每个设计抽几组约束")
    ap.add_argument("--max-ex", type=int, default=600_000)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=4096)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=1201)
    ap.add_argument("--grid", type=int, default=12, help="能力图谱网格边长")
    ap.add_argument("--nmc", type=int, default=4000, help="可行体积蒙特卡洛样本")
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)

    meta = json.load(open(os.path.join(os.path.dirname(args.factory), "factory_meta.json")))
    ds_meta = json.load(open(args.data.replace("gen_dataset.npz", "gen_dataset_meta.json")))
    lo, hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    c_lo, c_hi = np.array(ds_meta["c_lo"]), np.array(ds_meta["c_hi"])
    test_cids = set(ds_meta["test_cids"])

    pools = load_pools(args.factory, lo, hi)
    if os.path.exists(args.inc):
        for line in open(args.inc):
            d = json.loads(line)
            Yi = np.array([[np.nan if v is None else v for v in r] for r in d["Y"]], float)
            pools[d["cid"]].absorb(np.array(d["X"], float), Yi, lo, hi)
    npool = sum(len(p.X) for p in pools.values())
    print(f"[e12] 池 {npool} 设计 × {args.kdraw} 组约束 → 样本上限 {args.max_ex}")

    t0 = time.time()
    Xn, C, Y, G = build_examples(pools, lo, hi, args.kdraw, args.seed, args.max_ex)
    Cn = norm(C, c_lo, c_hi).astype(np.float32)
    te = np.isin(G, list(test_cids))
    print(f"[e12] 样本 {len(Y)}(训练 {(~te).sum()} / 留出工况测试 {te.sum()}), "
          f"整体可行率 {Y.mean() * 100:.1f}%  ({time.time() - t0:.0f}s)")

    Xtr = torch.tensor(Xn[~te]); Ctr = torch.tensor(Cn[~te]); Ytr = torch.tensor(Y[~te])
    Xte = torch.tensor(Xn[te]); Cte = torch.tensor(Cn[te]); Yte = Y[te]

    model = FeasNet(xd=len(lo), cd=len(c_lo))
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    lossf = nn.BCEWithLogitsLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(Xtr, Ctr, Ytr),
        batch_size=args.batch, shuffle=True)
    for ep in range(args.epochs):
        tot = nb = 0
        for xb, cb, yb in dl:
            loss = lossf(model(xb, cb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item(); nb += 1
        if (ep + 1) % 5 == 0:
            print(f"  ep {ep + 1}/{args.epochs}  BCE={tot / nb:.4f}  "
                  f"({time.time() - t0:.0f}s)")
    model.eval()

    # ---------------- 留出工况上的判别力与标定 ----------------
    P = model.prob(Xte, Cte)
    auc = roc_auc(Yte, P)
    acc = float(((P > 0.5) == (Yte > 0.5)).mean())
    base = float(max(Yte.mean(), 1 - Yte.mean()))       # 多数类基线
    brier = float(np.mean((P - Yte) ** 2))
    calib = []
    for a, b in zip(np.arange(0, 1, 0.1), np.arange(0.1, 1.01, 0.1)):
        s = (P >= a) & (P < b)
        if s.sum() >= 50:
            calib.append(dict(bin=f"{a:.1f}-{b:.1f}", n=int(s.sum()),
                              pred=float(P[s].mean()), actual=float(Yte[s].mean())))
    ece = float(np.sum([c["n"] * abs(c["pred"] - c["actual"]) for c in calib])
                / max(sum(c["n"] for c in calib), 1))
    print(f"[e12] 留出工况: AUC {auc:.4f}  准确率 {acc * 100:.1f}%"
          f"(多数类基线 {base * 100:.1f}%)  Brier {brier:.4f}  ECE {ece:.3f}")

    # ---------------- 能力图谱:可行体积分数随 (m, v0) 的演化 ----------------
    rng = np.random.default_rng(args.seed + 7)
    Xmc = torch.tensor(rng.random((args.nmc, len(lo))).astype(np.float32))   # 归一化立方内均匀
    ms = np.linspace(c_lo[0], c_hi[0], args.grid)
    vs = np.linspace(c_lo[1], c_hi[1], args.grid)
    gmid = float(np.mean(GCAP_RANGE) * 9.81); smid = float(np.mean(SMAX_RANGE))
    grid = []
    for mm in ms:
        for vv in vs:
            cn = torch.tensor(np.tile(norm(np.array([mm, vv, gmid, smid]), c_lo, c_hi),
                                      (args.nmc, 1)).astype(np.float32))
            pr = model.prob(Xmc, cn)
            grid.append(dict(m=float(mm), v0=float(vv),
                             feas_vol=float(pr.mean()),          # 可行体积分数
                             p90=float(np.quantile(pr, 0.9))))
    fv = np.array([g["feas_vol"] for g in grid]).reshape(args.grid, args.grid)
    print(f"[e12] 能力图谱 {args.grid}×{args.grid}:可行体积分数 "
          f"{fv.min() * 100:.1f}%–{fv.max() * 100:.1f}%(中位 {np.median(fv) * 100:.1f}%)"
          f"@ 中位约束 g_cap={gmid / 9.81:.1f}g s_max={smid * 1e3:.0f}mm")

    torch.save(dict(state=model.state_dict(), xd=len(lo), cd=len(c_lo),
                    x_lo=lo.tolist(), x_hi=hi.tolist(),
                    c_lo=c_lo.tolist(), c_hi=c_hi.tolist()),
               os.path.join(args.out, "feas_clf.pt"))
    json.dump(dict(n_pool=npool, n_examples=int(len(Y)),
                   feas_rate=float(Y.mean()),
                   test=dict(auc=auc, acc=acc, majority_baseline=base,
                             brier=brier, ece=ece, calibration=calib,
                             n=int(te.sum())),
                   capability_grid=dict(gcap_ms2=gmid, smax_m=smid,
                                        m=ms.tolist(), v0=vs.tolist(), cells=grid)),
              open(os.path.join(args.out, "e12_results.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[e12] done → {args.out}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

"""E5 · 自提升循环:生成 → 真值验证 → 择优回灌 → 重训。

结构上是"有真值裁判的 expert iteration":
  提议者 = 当前 cVAE(按工况条件生成候选)
  裁判   = Exudyn(牛顿定律,不可欺)
  回灌   = 验证后的设计并入该工况设计池 → 前沿重算 → 数据变好 → 模型变好

三道防线(方案对应):
  防近亲繁殖 —— 每轮候选掺 ε 比例新鲜 LHS 探索样本,回灌整池而非单点;
  防考题泄露 —— 留出评测场景自始冻结,参考前沿只算一次并缓存;
  防池子发霉 —— 单位立方 3 位小数网格去重,同一设计不重复入池。

用法(A100):
  python src/stage7_generative/e5_loop.py \
    --factory outputs/gen_data7/factory.jsonl --data outputs/gen_data7/gen_dataset.npz \
    --out outputs/gen_e5 --rounds 3 --kgen 24 --workers 64
断点续跑:同命令重跑,自动从上次完成的轮次继续。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import GCAP_RANGE, SMAX_RANGE, pareto2   # noqa: E402
from data_factory import _eval_one                          # noqa: E402
from train_cvae import CVAE, fit, norm                      # noqa: E402


# ---------------------------------------------------------------- 数据结构

@dataclass
class Pool:
    """一个工况的设计池:初始 LHS + 历轮验证过的回灌设计。"""
    cid: int
    m: float
    v0: float
    X: np.ndarray                       # (n, d)
    Y: np.ndarray                       # (n, k) 全指标,NaN=不可行
    seen: set = field(default_factory=set)

    def key(self, x, lo, hi):
        return tuple(np.round((x - lo) / (hi - lo), 3))

    def absorb(self, X_new, Y_new, lo, hi):
        """去重并入;返回真正新增的条数。"""
        fresh = [i for i, x in enumerate(X_new)
                 if self.key(x, lo, hi) not in self.seen]
        for i in fresh:
            self.seen.add(self.key(X_new[i], lo, hi))
        if fresh:
            self.X = np.vstack([self.X, X_new[fresh]])
            self.Y = np.vstack([self.Y, Y_new[fresh]])
        return len(fresh)


# ---------------------------------------------------------------- 工具

def lhs(n, d, rng):
    X = np.empty((n, d))
    for j in range(d):
        e = (np.arange(n) + rng.random(n)) / n
        X[:, j] = rng.permutation(e)
    return X


def simulate(ex, X, m, v0, keys):
    """批量真值评估 → (n, k) float,不可行为 NaN。"""
    rows = list(ex.map(_eval_one, [(x, m, v0) for x in X], chunksize=2))
    return np.array([[np.nan if v is None else v for v in r] for r in rows])


def load_pools(factory_path, lo, hi):
    pools = {}
    for line in open(factory_path):
        c = json.loads(line)
        X = np.array(c["X"], float)
        Y = np.array([[np.nan if v is None else v for v in row] for row in c["Y"]], float)
        p = Pool(c["cid"], c["m"], c["v0"], X, Y)
        p.seen = {p.key(x, lo, hi) for x in X}
        pools[c["cid"]] = p
    return pools


def build_pairs(pools, test_cids, iP, iS, k_scen):
    """池 → 训练对(逻辑与 build_dataset 一致,约束场景种子按 cid 固定)。"""
    C_tr, X_tr = [], []
    for p in pools.values():
        if p.cid in test_cids:
            continue
        ok = np.isfinite(p.Y[:, iP])
        X, Y = p.X[ok], p.Y[ok]
        crng = np.random.default_rng(77_000 + p.cid)
        for _ in range(k_scen):
            gcap = crng.uniform(*GCAP_RANGE) * 9.81
            smax = crng.uniform(*SMAX_RANGE)
            feas = (Y[:, iP] <= gcap) & (Y[:, iS] <= smax)
            if feas.sum() < 3:
                continue
            front = pareto2(Y[feas][:, [iP, iS]])
            for x in X[feas][front]:
                C_tr.append([p.m, p.v0, gcap, smax]); X_tr.append(x.tolist())
    return np.array(C_tr), np.array(X_tr)


# ---------------------------------------------------------------- 评测(冻结)

def build_ref_cache(ex, C_te, lo, hi, iP, iS, nref, keys, cache_fp):
    """留出场景的参考前沿——只算一次,此后各轮共用同一把尺子。"""
    if os.path.exists(cache_fp):
        return json.load(open(cache_fp))
    cache = []
    for si, row in enumerate(C_te):
        m, v0, gcap, smax = row[:4]
        rng = np.random.default_rng(500_000 + si)
        Y = simulate(ex, lo + (hi - lo) * lhs(nref, len(lo), rng), m, v0, keys)
        okr = np.isfinite(Y[:, iP]) & (Y[:, iP] <= gcap) & (Y[:, iS] <= smax)
        if not okr.any():
            continue
        cache.append(dict(m=m, v0=v0, gcap=gcap, smax=smax,
                          ref=float(Y[okr, iP].min()),
                          span=float(np.ptp(Y[okr, iS])) if okr.sum() > 1 else 0.0))
    json.dump(cache, open(cache_fp, "w"))
    return cache


def eval_model(model, ex, refs, meta, iP, iS, ngen, keys):
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    gaps, feas_r, cov = [], [], []
    for r in refs:
        cn = torch.tensor(norm(np.array([r["m"], r["v0"], r["gcap"], r["smax"]]),
                               c_lo, c_hi), dtype=torch.float32)
        Xg = x_lo + (x_hi - x_lo) * model.sample(cn, ngen).numpy()
        Yg = simulate(ex, Xg, r["m"], r["v0"], keys)
        ok = np.isfinite(Yg[:, iP]) & (Yg[:, iP] <= r["gcap"]) & (Yg[:, iS] <= r["smax"])
        feas_r.append(float(ok.mean()))
        gaps.append((float(Yg[ok, iP].min()) - r["ref"]) / r["ref"] if ok.any() else 1.0)
        cov.append(float(np.ptp(Yg[ok, iS])) / r["span"]
                   if ok.sum() > 1 and r["span"] > 0 else np.nan)
    return dict(median_gap=float(np.median(gaps)), mean_gap=float(np.mean(gaps)),
                feas_rate=float(np.mean(feas_r)), coverage=float(np.nanmean(cov)),
                fail=int(sum(g >= 1.0 for g in gaps)), n=len(gaps))


# ---------------------------------------------------------------- 主循环

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/gen_data7/factory.jsonl")
    ap.add_argument("--data", default="outputs/gen_data7/gen_dataset.npz")
    ap.add_argument("--out", default="outputs/gen_e5")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--kgen", type=int, default=24, help="每工况每轮候选数(含探索)")
    ap.add_argument("--eps", type=float, default=0.3, help="探索比例(新鲜 LHS)")
    ap.add_argument("--kscen", type=int, default=6)
    ap.add_argument("--ngen-eval", type=int, default=40)
    ap.add_argument("--nref", type=int, default=300)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--zdim", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(os.path.join(os.path.dirname(args.factory), "factory_meta.json")))
    ds_meta = json.load(open(args.data.replace("gen_dataset.npz", "gen_dataset_meta.json")))
    keys = meta["keys"]; iP, iS = keys.index("peak_a"), keys.index("stroke")
    lo, hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    D = len(lo)
    test_cids = set(ds_meta["test_cids"])
    gmeta = dict(ds_meta, x_lo=lo.tolist(), x_hi=hi.tolist())   # 生成/评测共用

    traj_fp = os.path.join(args.out, "trajectory.json")
    traj = json.load(open(traj_fp)) if os.path.exists(traj_fp) else []
    start = len(traj)
    print(f"[e5] design dim {D}, train conds {500 - len(test_cids)}, "
          f"resume from round {start}/{args.rounds}")

    pools = load_pools(args.factory, lo, hi)
    # 追平历史轮次的回灌(断点续跑:重放已存的池增量)
    inc_fp = os.path.join(args.out, "pool_increments.jsonl")
    if os.path.exists(inc_fp):
        for line in open(inc_fp):
            inc = json.loads(line)
            pools[inc["cid"]].absorb(np.array(inc["X"]), np.array(inc["Y"]), lo, hi)

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        refs = build_ref_cache(ex, np.load(args.data)["C_te"], lo, hi, iP, iS,
                               args.nref, keys, os.path.join(args.out, "refs.json"))
        print(f"[e5] frozen eval scenarios: {len(refs)}  ({time.time()-t0:.0f}s)")

        for rd in range(start, args.rounds + 1):
            # —— 训练(第 0 轮 = 纯工厂数据基线)——
            C_tr, X_tr = build_pairs(pools, test_cids, iP, iS, args.kscen)
            c_lo, c_hi = np.array(gmeta["c_lo"]), np.array(gmeta["c_hi"])
            model, _ = fit(norm(C_tr, c_lo, c_hi), norm(X_tr, lo, hi),
                           epochs=args.epochs, zdim=args.zdim, seed=rd, verbose=False)
            score = eval_model(model, ex, refs, gmeta, iP, iS, args.ngen_eval, keys)
            npool = int(sum(len(p.X) for p in pools.values()))
            traj.append(dict(round=rd, pairs=len(C_tr), pool=npool, **score))
            json.dump(traj, open(traj_fp, "w"), indent=2)
            torch.save(dict(state=model.state_dict(), meta=gmeta, xd=D, zdim=args.zdim),
                       os.path.join(args.out, f"cvae_r{rd}.pt"))
            print(f"[e5] r{rd}: pairs={len(C_tr)} pool={npool} | "
                  f"gap median {score['median_gap']*100:.1f}% mean {score['mean_gap']*100:.1f}% "
                  f"feas {score['feas_rate']*100:.0f}% cov {score['coverage']*100:.0f}% "
                  f"({time.time()-t0:.0f}s)")
            if rd == args.rounds:
                break

            # —— 提议 + 验证 + 回灌 ——
            n_exp = int(round(args.kgen * args.eps))
            n_gen = args.kgen - n_exp
            added = 0
            with open(inc_fp, "a") as f:
                for p in pools.values():
                    if p.cid in test_cids:
                        continue
                    rng = np.random.default_rng(rd * 1_000_003 + p.cid)
                    crng = np.random.default_rng(rd * 2_000_003 + p.cid)
                    cand = [lo + (hi - lo) * lhs(n_exp, D, rng)]        # 探索
                    for _ in range(2):                                  # 两个约束场景下开采
                        c = np.array([p.m, p.v0, crng.uniform(*GCAP_RANGE) * 9.81,
                                      crng.uniform(*SMAX_RANGE)])
                        cn = torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32)
                        cand.append(lo + (hi - lo) * model.sample(cn, n_gen // 2).numpy())
                    X_new = np.clip(np.vstack(cand), lo, hi)
                    Y_new = simulate(ex, X_new, p.m, p.v0, keys)
                    added += p.absorb(X_new, Y_new, lo, hi)
                    f.write(json.dumps(dict(cid=p.cid, round=rd,
                                            X=np.round(X_new, 4).tolist(),
                                            Y=[[None if not np.isfinite(v) else float(v)
                                                for v in row] for row in Y_new])) + "\n")
            print(f"[e5] r{rd}→r{rd+1}: 回灌 {added} 新设计(ε={args.eps} 探索)"
                  f"  ({time.time()-t0:.0f}s)")

    print("\n== E5 轨迹 ==")
    for t in traj:
        print(f"  r{t['round']}: gap {t['median_gap']*100:5.1f}% | feas {t['feas_rate']*100:3.0f}%"
              f" | cov {t['coverage']*100:3.0f}% | pool {t['pool']}")
    print(f"[e5] done → {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

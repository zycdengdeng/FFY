"""E5c · 种子分布 + 合法选模:同一份数据 × N 种子,验证集挑选,测试集报告。

回答两个问题:
1) r12/r14 的低 gap 是彩票还是地板下移?——同数据 N 种子的 gap 分布一锤定音;
2) 部署哪个模型?——用全新验证集(新抽工况+新参考前沿)挑选,
   76 道冻结考题只在最后报告成绩:选择与报告分离,无考题泄露。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage7_generative/e5_seeds.py \
    --factory outputs/gen_data7/factory.jsonl --data outputs/gen_data7/gen_dataset.npz \
    --inc outputs/gen_e5/pool_increments.jsonl --refs outputs/gen_e5/refs.json \
    --out outputs/gen_e5c --nseeds 8 --workers 128
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

sys.path.insert(0, os.path.dirname(__file__))
from build_dataset import GCAP_RANGE, SMAX_RANGE            # noqa: E402
from e5_loop import (build_pairs, build_ref_cache, eval_model,  # noqa: E402
                     lhs, load_pools)
from train_cvae import fit, norm                            # noqa: E402


def fresh_validation_conditions(n, meta, seed=987_001):
    """全新验证工况:与工厂/测试集独立抽样的 (m, v0, gcap, smax)。"""
    rng = np.random.default_rng(seed)
    C = lhs(n, 2, rng)
    m = meta["c_lo"]["m"] + (meta["c_hi"]["m"] - meta["c_lo"]["m"]) * C[:, 0]
    v0 = meta["c_lo"]["v0"] + (meta["c_hi"]["v0"] - meta["c_lo"]["v0"]) * C[:, 1]
    gcap = rng.uniform(*GCAP_RANGE, n) * 9.81
    smax = rng.uniform(*SMAX_RANGE, n)
    return np.column_stack([m, v0, gcap, smax])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/gen_data7/factory.jsonl")
    ap.add_argument("--data", default="outputs/gen_data7/gen_dataset.npz")
    ap.add_argument("--inc", default="outputs/gen_e5/pool_increments.jsonl")
    ap.add_argument("--refs", default="outputs/gen_e5/refs.json",
                    help="冻结测试参考前沿(复用 E5 的缓存)")
    ap.add_argument("--out", default="outputs/gen_e5c")
    ap.add_argument("--nseeds", type=int, default=8)
    ap.add_argument("--nval", type=int, default=24)
    ap.add_argument("--nref", type=int, default=300)
    ap.add_argument("--ngen-eval", type=int, default=40)
    ap.add_argument("--kscen", type=int, default=6)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--zdim", type=int, default=3)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    meta = json.load(open(os.path.join(os.path.dirname(args.factory), "factory_meta.json")))
    ds_meta = json.load(open(args.data.replace("gen_dataset.npz", "gen_dataset_meta.json")))
    keys = meta["keys"]; iP, iS = keys.index("peak_a"), keys.index("stroke")
    lo, hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    gmeta = dict(ds_meta, x_lo=lo.tolist(), x_hi=hi.tolist())
    c_lo, c_hi = np.array(gmeta["c_lo"]), np.array(gmeta["c_hi"])

    # 数据 = 工厂 + E5 全部回灌(即"当前最好的池")
    pools = load_pools(args.factory, lo, hi)
    if os.path.exists(args.inc):
        for line in open(args.inc):
            inc = json.loads(line)
            Yi = np.array([[np.nan if v is None else v for v in row]
                           for row in inc["Y"]], float)   # null(求解失败)→ NaN,与实时路径一致
            pools[inc["cid"]].absorb(np.array(inc["X"], float), Yi, lo, hi)
    C_tr, X_tr = build_pairs(pools, set(ds_meta["test_cids"]), iP, iS, args.kscen)
    print(f"[e5c] pool={sum(len(p.X) for p in pools.values())}  pairs={len(C_tr)}")

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        test_refs = json.load(open(args.refs))
        C_val = fresh_validation_conditions(args.nval, meta)
        val_refs = build_ref_cache(ex, C_val, lo, hi, iP, iS, args.nref, keys,
                                   os.path.join(args.out, "val_refs.json"))
        print(f"[e5c] validation scenarios: {len(val_refs)}  ({time.time()-t0:.0f}s)")

        rows = []
        for sd in range(args.nseeds):
            model, _ = fit(norm(C_tr, c_lo, c_hi), norm(X_tr, lo, hi),
                           epochs=args.epochs, zdim=args.zdim, seed=sd, verbose=False)
            val = eval_model(model, ex, val_refs, gmeta, iP, iS, args.ngen_eval, keys)
            tst = eval_model(model, ex, test_refs, gmeta, iP, iS, args.ngen_eval, keys)
            rows.append(dict(seed=sd, val_gap=val["median_gap"], test_gap=tst["median_gap"],
                             test_mean=tst["mean_gap"], test_fail=tst["fail"],
                             test_cov=tst["coverage"], test_feas=tst["feas_rate"]))
            torch.save(dict(state=model.state_dict(), meta=gmeta, xd=len(lo),
                            zdim=args.zdim), os.path.join(args.out, f"cvae_s{sd}.pt"))
            print(f"  seed {sd}: val {val['median_gap']*100:5.1f}%  "
                  f"test {tst['median_gap']*100:5.1f}%  ({time.time()-t0:.0f}s)")

    tg = np.array([r["test_gap"] for r in rows])
    pick = int(np.argmin([r["val_gap"] for r in rows]))       # 只看验证集挑
    summary = dict(
        n_seeds=args.nseeds,
        test_gap_median_across_seeds=float(np.median(tg)),
        test_gap_mean=float(tg.mean()), test_gap_std=float(tg.std()),
        test_gap_min=float(tg.min()), test_gap_max=float(tg.max()),
        picked_seed=pick, picked_by="validation median_gap",
        picked_test_gap=float(rows[pick]["test_gap"]),
        picked_test_cov=float(rows[pick]["test_cov"]),
        picked_test_fail=int(rows[pick]["test_fail"]))
    json.dump(dict(summary=summary, rows=rows),
              open(os.path.join(args.out, "e5c_results.json"), "w"), indent=2)

    print("\n== E5c 种子分布(同一份数据)==")
    print(f"  test gap: 中位 {np.median(tg)*100:.1f}%  均值 {tg.mean()*100:.1f}%"
          f" ± {tg.std()*100:.1f}%  最好 {tg.min()*100:.1f}%  最差 {tg.max()*100:.1f}%")
    print(f"  验证集挑中 seed {pick} → 测试 gap {rows[pick]['test_gap']*100:.1f}%"
          f"(合法口径,无泄露)")
    print(f"[e5c] done → {args.out}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

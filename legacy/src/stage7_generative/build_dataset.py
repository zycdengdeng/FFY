"""从数据工厂产物构建条件生成训练集。

工厂只存 (m, v0) × 设计 × 全指标;本脚本把约束参数 (g_cap, stroke_max) 采样进来:
每个工况配 K 个约束场景 → 可行集 → (peak_a, stroke) 前沿段 = 该条件下的"正确答案族"。
训练对 = c=(m, v0, g_cap, stroke_max) → x=(L1, r2, r3),一对多。

划分:按工况 cid 留出 --ntest 个做评测(内插);外推评测用工况范围边角(eval 脚本现采)。

用法: python src/stage7_generative/build_dataset.py \
        --factory outputs/gen_data/factory.jsonl --out outputs/gen_data
产出: gen_dataset.npz + gen_dataset_meta.json
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

GCAP_RANGE = (4.0, 15.0)      # g
SMAX_RANGE = (0.008, 0.040)   # m


def pareto2(P):
    """双目标最小化的非支配掩码,O(N log N) 排序扫描版。
    (按目标0升序、平局按目标1升序;沿序扫描,目标1创新低者非支配。
     与 O(N²) 定义等价;完全重复点只保留一份——对训练集无害且更干净。)"""
    P = np.asarray(P, float)
    order = np.lexsort((P[:, 1], P[:, 0]))
    mask = np.zeros(len(P), bool)
    best1 = np.inf
    for i in order:
        if P[i, 1] < best1:
            mask[i] = True
            best1 = P[i, 1]
    return mask


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/gen_data/factory.jsonl")
    ap.add_argument("--out", default="outputs/gen_data")
    ap.add_argument("--k", type=int, default=6, help="每工况约束场景数")
    ap.add_argument("--ntest", type=int, default=20)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    meta_f = json.load(open(os.path.join(os.path.dirname(args.factory), "factory_meta.json")))
    KEYS = meta_f["keys"]; iP, iS = KEYS.index("peak_a"), KEYS.index("stroke")

    conds = [json.loads(l) for l in open(args.factory)]
    print(f"[ds] factory conditions: {len(conds)}")
    rng = np.random.default_rng(args.seed)
    test_cids = set(rng.choice([c["cid"] for c in conds], args.ntest, replace=False).tolist())

    C_tr, X_tr, C_te = [], [], []
    n_empty = 0
    for c in conds:
        X = np.array(c["X"], float)
        Y = np.array([[np.nan if v is None else v for v in row] for row in c["Y"]], float)
        ok = np.isfinite(Y[:, iP])
        X, Y = X[ok], Y[ok]
        crng = np.random.default_rng(77_000 + c["cid"])
        for _ in range(args.k):
            gcap = crng.uniform(*GCAP_RANGE) * 9.81
            smax = crng.uniform(*SMAX_RANGE)
            feas = (Y[:, iP] <= gcap) & (Y[:, iS] <= smax)
            if feas.sum() < 3:
                n_empty += 1; continue
            Xf, Yf = X[feas], Y[feas][:, [iP, iS]]
            front = pareto2(Yf)
            cvec = [c["m"], c["v0"], gcap, smax]
            if c["cid"] in test_cids:
                C_te.append(cvec + [c["cid"]])
            else:
                for x in Xf[front]:
                    C_tr.append(cvec); X_tr.append(x.tolist())
    C_tr, X_tr = np.array(C_tr), np.array(X_tr)
    C_te = np.array(C_te)
    print(f"[ds] train pairs: {len(C_tr)}  test scenarios: {len(C_te)}  "
          f"empty-feasible skipped: {n_empty}")

    np.savez(os.path.join(args.out, "gen_dataset.npz"),
             C_tr=C_tr, X_tr=X_tr, C_te=C_te)
    json.dump(dict(gcap_range=GCAP_RANGE, smax_range=SMAX_RANGE,
                   c_order=["m", "v0", "gcap_ms2", "smax_m"],
                   x_order=["L1", "r2", "r3"],
                   x_lo=meta_f["x_lo"], x_hi=meta_f["x_hi"],
                   c_lo=[meta_f["c_lo"]["m"], meta_f["c_lo"]["v0"], GCAP_RANGE[0] * 9.81, SMAX_RANGE[0]],
                   c_hi=[meta_f["c_hi"]["m"], meta_f["c_hi"]["v0"], GCAP_RANGE[1] * 9.81, SMAX_RANGE[1]],
                   test_cids=sorted(test_cids), k=args.k, seed=args.seed),
              open(os.path.join(args.out, "gen_dataset_meta.json"), "w"), indent=2)
    print(f"[ds] wrote gen_dataset.npz + meta → {args.out}")


if __name__ == "__main__":
    main()

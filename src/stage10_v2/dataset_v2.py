"""由 v2 工厂构建训练对,并按**束(bid)**做防泄漏切分。

两件 v1 没有的事:
 1. **按 bid 切,不按 cid 切**。一个路径束把同一批设计放进 K 个不同 cid;
    按 cid 切会让同一个设计同时出现在训练与测试侧 —— 静默泄漏,且数字虚高、
    自己很难发现。切分单元必须是束。
 2. 条件向量里 m 与 k_c 取 **log10**。两者都按对数均匀采样,且异速先验本身就是
    log-log 线性的;线性归一化会把分辨率浪费在大质量端。

条件向量 c(5 维): [log10(m), v0, log10(k_c), g_cap, s_max]
设计向量 u(7 维): 无量纲,与 m 无关(物理设计由 bioprior.expand(u, m) 还原)

用法:
  python src/stage10_v2/dataset_v2.py --factory outputs/v2_data_bio/factory.jsonl \
      --out outputs/v2_data_bio --nreq 4 --ktop 8
产出: dataset.npz(U/C 三份切分) + dataset_meta.json + paths.json(路径监督用)
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factory_v2 import KEYS_V2                    # noqa: E402

iP, iS, iL = (KEYS_V2.index("peak_a"), KEYS_V2.index("stroke"),
              KEYS_V2.index("leg_stroke"))
iSO, iMO = KEYS_V2.index("struct_over"), KEYS_V2.index("mass_over")

GCAP_RANGE = (4.0, 15.0)          # g
SMAX_RANGE = (0.008, 0.040)       # m


def pareto2(a, b):
    """二目标非支配前沿(都取小):返回下标。"""
    idx = np.argsort(a, kind="stable")
    out, best = [], np.inf
    for i in idx:
        if b[i] < best - 1e-15:
            out.append(int(i)); best = b[i]
    return out


def load_blocks(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def to_Y(blk):
    return np.array([[np.nan if v is None else v for v in y] for y in blk["Y"]], float)


def feasible_mask(Y, gcap, smax):
    """与 physics_v2.feasible_v2 同口径:s_max 作用在**腿行程**上。"""
    ok = np.isfinite(Y[:, iP])
    return (ok & (Y[:, iP] <= gcap) & (Y[:, iL] <= smax)
            & (np.nan_to_num(Y[:, iSO], nan=1.0) < 0.5)
            & (np.nan_to_num(Y[:, iMO], nan=1.0) < 0.5))


def split_by_bid(bids, rng, frac=(0.70, 0.15, 0.15)):
    """整束整块地分。返回 (train_set, val_set, test_set) 的 bid 集合。"""
    u = np.array(sorted(set(bids)))
    rng.shuffle(u)
    n1 = int(round(frac[0] * len(u))); n2 = n1 + int(round(frac[1] * len(u)))
    return set(u[:n1].tolist()), set(u[n1:n2].tolist()), set(u[n2:].tolist())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/v2_data_bio/factory.jsonl")
    ap.add_argument("--out", default="outputs/v2_data_bio")
    ap.add_argument("--nreq", type=int, default=4, help="每块抽几组设计要求")
    ap.add_argument("--ktop", type=int, default=8, help="每组要求下取前沿前几名进训练")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    blocks = load_blocks(args.factory)
    if not blocks:
        raise SystemExit(f"[dataset] 空工厂:{args.factory}")
    meta_f = json.load(open(os.path.join(os.path.dirname(args.factory),
                                         "factory_meta.json")))
    rng = np.random.default_rng(args.seed)
    tr_b, va_b, te_b = split_by_bid([b["bid"] for b in blocks], rng)
    print(f"[dataset] 块 {len(blocks)}  束 {len(tr_b)+len(va_b)+len(te_b)} "
          f"→ 训练 {len(tr_b)} / 验证 {len(va_b)} / 测试 {len(te_b)} (按 bid 切)")

    buckets = {"tr": ([], []), "va": ([], []), "te": ([], [])}
    nfeas_tot = ntot = 0
    for blk in blocks:
        Y = to_Y(blk)
        U = np.array(blk["U"], float)
        tag = "tr" if blk["bid"] in tr_b else ("va" if blk["bid"] in va_b else "te")
        crng = np.random.default_rng(90_000 + blk["cid"])
        for _ in range(args.nreq):
            gcap = float(crng.uniform(*GCAP_RANGE) * 9.81)
            smax = float(crng.uniform(*SMAX_RANGE))
            fe = feasible_mask(Y, gcap, smax)
            ntot += len(fe); nfeas_tot += int(fe.sum())
            if not fe.any():
                continue
            idx = np.where(fe)[0]
            front = pareto2(Y[idx, iP], Y[idx, iL])          # 峰值 vs 腿行程
            pick = idx[front][:args.ktop]
            c = [np.log10(blk["m"]), blk["v0"], np.log10(blk["kc"]), gcap, smax]
            for j in pick:
                buckets[tag][0].append(c); buckets[tag][1].append(U[j])

    c_lo = [np.log10(meta_f["m_range"][0]), meta_f["v0_range"][0],
            np.log10(meta_f["kc_range"][0]), GCAP_RANGE[0] * 9.81, SMAX_RANGE[0]]
    c_hi = [np.log10(meta_f["m_range"][1]), meta_f["v0_range"][1],
            np.log10(meta_f["kc_range"][1]), GCAP_RANGE[1] * 9.81, SMAX_RANGE[1]]

    arrs = {}
    for k in ("tr", "va", "te"):
        C = np.array(buckets[k][0], float).reshape(-1, 5)
        U = np.array(buckets[k][1], float).reshape(-1, 7)
        arrs[f"C_{k}"], arrs[f"U_{k}"] = C, U
        print(f"  {k}: {len(C):6d} 对")
    np.savez(os.path.join(args.out, "dataset.npz"), **arrs)

    # 路径结构单独存,供后续 path 监督实验(F/C 头)使用;此版 cVAE 暂不用
    paths = {}
    for blk in blocks:
        if blk["kind"] != "path":
            continue
        paths.setdefault(str(blk["bid"]), dict(walk=blk["walk"], steps=[]))
        paths[str(blk["bid"])]["steps"].append(
            dict(step=blk["step"], m=blk["m"], v0=blk["v0"], kc=blk["kc"],
                 cid=blk["cid"],
                 split=("tr" if blk["bid"] in tr_b else
                        "va" if blk["bid"] in va_b else "te")))
    for v in paths.values():
        v["steps"].sort(key=lambda s: s["step"])
    json.dump(paths, open(os.path.join(args.out, "paths.json"), "w"),
              indent=2, ensure_ascii=False)

    json.dump(dict(c_order=["log10_m", "v0", "log10_kc", "gcap_ms2", "smax_m"],
                   c_lo=c_lo, c_hi=c_hi, u_dim=7, arm=meta_f["arm"],
                   prior=meta_f["prior"], keys=KEYS_V2,
                   gcap_range=GCAP_RANGE, smax_range=SMAX_RANGE,
                   nreq=args.nreq, ktop=args.ktop,
                   split_by="bid", n_bundles=dict(tr=len(tr_b), va=len(va_b), te=len(te_b)),
                   bids=dict(tr=sorted(tr_b), va=sorted(va_b), te=sorted(te_b)),
                   feas_rate=float(nfeas_tot / max(ntot, 1)),
                   note="设计为无量纲 u;物理设计 = bioprior.expand(u, m)"),
              open(os.path.join(args.out, "dataset_meta.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[dataset] 平均可行率 {100*nfeas_tot/max(ntot,1):.1f}%  "
          f"路径束 {len(paths)} 条 → {args.out}")


if __name__ == "__main__":
    main()

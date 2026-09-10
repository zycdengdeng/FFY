"""由 v2 工厂构建训练对,并按**束(bid)**做防泄漏切分。

两件 v1 没有的事:
 1. **按 bid 切,不按 cid 切**。一个路径束把同一批设计放进 K 个不同 cid;
    按 cid 切会让同一个设计同时出现在训练与测试侧 —— 静默泄漏,且数字虚高、
    自己很难发现。切分单元必须是束。
 2. 条件向量里 m 与 k_c 取 **log10**。两者都按对数均匀采样,且异速先验本身就是
    log-log 线性的;线性归一化会把分辨率浪费在大质量端。

条件向量 c(5 维): [log10(m), v0, log10(k_c), g_cap, s_max]
设计向量 u(v2.0 为 7 维;v2.1 起 9 维,+触地姿态 θA/θK): 无量纲,与 m 无关(物理设计由 bioprior.expand(u, m) 还原)

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

ND_U = 7    # 运行时由 factory_meta.json 的 u_dim 覆盖(v2.1 为 9,v2.5 为 10)
FR6 = False # 运行时覆盖:v2.5 的条件向量第 6 维是 Froude 数

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from factory_v2 import KEYS_V2, KEYS_V25          # noqa: E402

KEYS_ALL = KEYS_V2 + KEYS_V25
iP, iS, iL = (KEYS_V2.index("peak_a"), KEYS_V2.index("stroke"),
              KEYS_V2.index("leg_stroke"))
iSO, iMO = KEYS_V2.index("struct_over"), KEYS_V2.index("mass_over")
iREB, iNB = KEYS_V2.index("rebound"), KEYS_V2.index("n_bounce")

# v2.5 新增列的下标。**旧的 v2.3 工厂只有前 16 列**,所以这些下标可能越界 ——
# feasible_mask 里逐个判 Y.shape[1],越界就当这条判据不存在,旧数据照样能重打分。
_i25 = {k: len(KEYS_V2) + KEYS_V25.index(k) for k in KEYS_V25}
iARES, iMUD, iMUG, iV0 = (_i25["a_res"], _i25["mu_demand"],
                          _i25["mu_ground"], _i25["v0"])
iSLIP, iLL1 = _i25["slip"], _i25["L1_mm"]

# 与 physics_v2.feasible_v2 同口径,改一处两处都要改
REB_CAP, MU_SF, G, SLIP_FRAC = 0.05, 1.0, 9.81, 0.5

# 上界 15→25 g:v1 的 [4,15] 是按 v1 那个唯一(且偏软)的等效地面标定的;
# v2 的地形跨到 1e6 N/m,硬地上生物设计盒里没有任何设计能进 15 g。
# 实测(pilot 10,200 次仿真):[4,15] 有 4% 的工况零可行,[4,25] 降到 0%。
GCAP_RANGE = (4.0, 25.0)          # g
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


def _col(Y, i):
    """取第 i 列;越界(旧工厂没这一列)返回 None。"""
    return Y[:, i] if Y.shape[1] > i else None


def feasible_mask(Y, gcap, smax, reb_cap=REB_CAP, mu_sf=MU_SF, slip_frac=SLIP_FRAC):
    """与 physics_v2.feasible_v2 同口径:s_max 作用在**腿行程**上。

    v2.5 三处改动,与 physics_v2.feasible_v2 一一对应:
      ① 过载改判合加速度 a_res(没有这一列就退回 peak_a);
      ② 回弹软闸 rebound/(v0²/2g) ≤ 5%;
      ③ 足端打滑：竖直冲击窗口内足端走开 ≤ 0.5·L1
         （不是 mu_demand ≤ mu_ground —— 滑动摩擦饱和后那个量恒等于 μ，判不出东西）。
    每一条都先看列在不在 —— **v2.3 的老工厂只有前 16 列,照样能重打分**。
    """
    ok = np.isfinite(Y[:, iP])
    a = _col(Y, iARES)
    a = Y[:, iP] if a is None else np.where(np.isfinite(a), a, Y[:, iP])
    m = (ok & (a <= gcap) & (Y[:, iL] <= smax)
         & (np.nan_to_num(Y[:, iSO], nan=1.0) < 0.5)
         & (np.nan_to_num(Y[:, iMO], nan=1.0) < 0.5))

    reb, v0 = _col(Y, iREB), _col(Y, iV0)
    if reb is not None and v0 is not None:
        h0 = np.maximum(v0 ** 2 / (2.0 * G), 1e-12)
        rr = np.nan_to_num(reb, nan=0.0) / h0
        m &= np.where(np.isfinite(v0) & (v0 > 0), rr <= reb_cap, True)

    sl, l1 = _col(Y, iSLIP), _col(Y, iLL1)
    if sl is not None and l1 is not None:
        bad = np.isfinite(sl) & np.isfinite(l1) & (l1 > 0) & (sl * 1e3 > slip_frac * l1)
        m &= ~bad
    return m


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
    global ND_U, FR6
    ND_U = int(meta_f.get("u_dim", 7))
    FR6 = bool(meta_f.get("v25")) and "Fr" in (blocks[0] if blocks else {})
    print(f"[dataset] 设计维 u_dim = {ND_U}   条件维 = {6 if FR6 else 5}"
          f"{'(含 Froude)' if FR6 else ''}")
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
            for j in pick:
                # v2.5:条件第 6 维是 Froude 数,**逐个设计不同**(工厂按设计采的 Fr),
                # 所以条件向量要进循环里拼,不能在循环外算一次。
                c = [np.log10(blk["m"]), blk["v0"], np.log10(blk["kc"]), gcap, smax]
                if FR6:
                    c.append(float(blk["Fr"][j]))
                buckets[tag][0].append(c); buckets[tag][1].append(U[j])

    c_lo = [np.log10(meta_f["m_range"][0]), meta_f["v0_range"][0],
            np.log10(meta_f["kc_range"][0]), GCAP_RANGE[0] * 9.81, SMAX_RANGE[0]]
    c_hi = [np.log10(meta_f["m_range"][1]), meta_f["v0_range"][1],
            np.log10(meta_f["kc_range"][1]), GCAP_RANGE[1] * 9.81, SMAX_RANGE[1]]
    if FR6:
        c_lo.append(0.0); c_hi.append(float(meta_f.get("fr_max", 2.0)))

    arrs = {}
    for k in ("tr", "va", "te"):
        C = np.array(buckets[k][0], float).reshape(-1, 5)
        U = np.array(buckets[k][1], float).reshape(-1, ND_U)
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
                   c_lo=c_lo, c_hi=c_hi, u_dim=ND_U, arm=meta_f["arm"],
                   v21=meta_f.get("v21", False),
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

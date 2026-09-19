# -*- coding: utf-8 -*-
"""由 v3 轮足工厂构建训练对。骨架照抄 dataset_v2（bid 防泄漏切分、log 归一化），
判据与条件向量换成 v3 口径：

条件 c(8 维): [log10 m, v0, log10 kc, gcap, smax, Fr, pitch_cap, log10 Lstop]
  - gcap  ∈ U[4,25] g               （沿用 v2.5；v3 数据 a_res P50=5.2 g，区分度够）
  - smax  ∈ U[0.02,0.15] m          （⚠ 不沿用 v2.5 的 [0.008,0.040]：那是裸足落震的
        货舱冲程口径。v3 双腿+俯仰机器的下沉 P5–P95 = 8–210 mm，拿旧尺子量会杀掉 77%
        ——口径错配而非设计差。新范围由数据分位定：下界 P5–P25 之间、上界 P75–P95 之间，
        物理解释是滑跑构型的下沉预算由起落架几何决定。实测抽签可行率 5.7%→16.3%，
        与 v2.5 al 臂的 15.2% 同量级。）
  - pitch_cap ∈ {15,25,40}°          （俯仰裕度抽签：载荷/桨叶越金贵越严）
  - Lstop ∈ logU[3,15] m             （跑道预算；条件里取 log10，与 m/kc 同理）

判据 feasible_mask_v3（与 physics_v3.feasible_v3 同口径，改一处两处都要改）：
  a_res≤gcap · leg_stroke≤smax · !struct_over · !mass_over · 回弹≤5%
  · pitch_max≤pitch_cap · stop_dist≤Lstop（None/inf 记不可行）。无 slip 闸（轮子本就滚）。

用法:
  python src/stage10_v2/dataset_v3.py --factory outputs/v3_data_bio/factory.jsonl \
      --out outputs/v3_data_bio --nreq 4 --ktop 8
产出: dataset.npz + dataset_meta.json + paths.json
"""
from __future__ import annotations

import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

# 纯后处理，只依赖 numpy —— 刻意不 import factory_v3/dataset_v2：那条链拖进 exudyn，
# 会让这个脚本在没有 exudyn 的机器（本地验收）上跑不了（agroup_io 的老教训）。
# KEYS_V3 就地复制；能 import 到真模块时自检防两边漂开。
KEYS_V3 = ["a_res", "peak_a", "stroke", "leg_stroke", "sink", "rebound",
           "pitch_max_deg", "pitch_end_deg", "roll_dist_m", "stop_dist_m",
           "stopped", "stop_extrap", "vx_end", "wheel_spin_max",
           "latch_f", "latch_r", "leg_mass_kg", "gear_mass_kg", "mass_frac",
           "struct_over", "mass_over", "D1_mm", "D2_mm", "D3_mm",
           "L1_mm", "L2_mm", "L3_mm", "v0", "v_x", "mu_ground"]
try:                                                  # 有 exudyn 的机器上做防漂自检
    from factory_v3 import KEYS_V3 as _K
    assert list(_K) == KEYS_V3, "KEYS_V3 与 factory_v3 漂开了，先同步再跑"
except ImportError:
    pass


def pareto2(a, b):
    """二目标非支配前沿（都取小），与 dataset_v2 同式。"""
    idx = np.argsort(a, kind="stable")
    out, best = [], np.inf
    for i in idx:
        if b[i] < best - 1e-15:
            out.append(int(i)); best = b[i]
    return out


def split_by_bid(bids, rng, frac=(0.70, 0.15, 0.15)):
    u = np.array(sorted(set(bids)))
    rng.shuffle(u)
    n1 = int(round(frac[0] * len(u))); n2 = n1 + int(round(frac[1] * len(u)))
    return set(u[:n1].tolist()), set(u[n1:n2].tolist()), set(u[n2:].tolist())


def load_blocks(fp):
    rows = []
    with open(fp) as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows

KI = {k: i for i, k in enumerate(KEYS_V3)}
iAR, iLS, iPM, iSD = KI["a_res"], KI["leg_stroke"], KI["pitch_max_deg"], KI["stop_dist_m"]
iSO, iMO, iREB, iV0 = KI["struct_over"], KI["mass_over"], KI["rebound"], KI["v0"]

G, REB_CAP = 9.81, 0.05
GCAP_RANGE_V3 = (4.0, 25.0)          # g
SMAX_RANGE_V3 = (0.020, 0.150)       # m（见文件头：由 v3 数据分位定，勿照抄 v2.5）
PITCH_CAPS = (15.0, 25.0, 40.0)      # °
LSTOP_RANGE = (3.0, 15.0)            # m，对数抽签

ND_U = 14


def to_Y(blk):
    return np.array([[np.nan if v is None else v for v in y] for y in blk["Y"]], float)


def feasible_mask_v3(Y, gcap_ms2, smax, pitch_cap, lstop,
                     reb_cap=REB_CAP):
    """gcap_ms2 单位 m/s²（与 dataset_v2 口径一致：条件向量里也存 m/s²）。"""
    ok = np.isfinite(Y[:, iAR])
    m = (ok & (Y[:, iAR] <= gcap_ms2) & (Y[:, iLS] <= smax)
         & (np.nan_to_num(Y[:, iSO], nan=1.0) < 0.5)
         & (np.nan_to_num(Y[:, iMO], nan=1.0) < 0.5)
         & (np.nan_to_num(Y[:, iPM], nan=1e9) <= pitch_cap)
         & (np.nan_to_num(Y[:, iSD], nan=np.inf) <= lstop))
    v0 = Y[:, iV0]
    h0 = np.maximum(np.nan_to_num(v0, nan=0.0) ** 2 / (2 * G), 1e-12)
    rr = np.nan_to_num(Y[:, iREB], nan=0.0) / h0
    m &= np.where(np.isfinite(v0) & (v0 > 0), rr <= reb_cap, True)
    return m


def draw_req(crng):
    """一组 v3 设计要求四元组。gcap 返回 m/s²。"""
    return (float(crng.uniform(*GCAP_RANGE_V3) * G),
            float(crng.uniform(*SMAX_RANGE_V3)),
            float(crng.choice(PITCH_CAPS)),
            float(10 ** crng.uniform(np.log10(LSTOP_RANGE[0]),
                                     np.log10(LSTOP_RANGE[1]))))


def cvec_v3(m, v0, kc, gcap_ms2, smax, fr, pitch_cap, lstop):
    return [np.log10(m), v0, np.log10(kc), gcap_ms2, smax,
            float(fr), pitch_cap, np.log10(lstop)]


def c_bounds(meta_f, fr_max):
    c_lo = [np.log10(meta_f["m_range"][0]), meta_f["v0_range"][0],
            np.log10(meta_f["kc_range"][0]), GCAP_RANGE_V3[0] * G, SMAX_RANGE_V3[0],
            0.0, PITCH_CAPS[0], np.log10(LSTOP_RANGE[0])]
    c_hi = [np.log10(meta_f["m_range"][1]), meta_f["v0_range"][1],
            np.log10(meta_f["kc_range"][1]), GCAP_RANGE_V3[1] * G, SMAX_RANGE_V3[1],
            max(fr_max, 1.0), PITCH_CAPS[-1], np.log10(LSTOP_RANGE[1])]
    return c_lo, c_hi


C_ORDER = ["log10_m", "v0", "log10_kc", "gcap_ms2", "smax_m",
           "Fr", "pitch_cap_deg", "log10_lstop_m"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/v3_data_bio/factory.jsonl")
    ap.add_argument("--out", default="outputs/v3_data_bio")
    ap.add_argument("--nreq", type=int, default=4)
    ap.add_argument("--ktop", type=int, default=8)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    blocks = load_blocks(args.factory)
    if not blocks:
        raise SystemExit(f"[dataset-v3] 空工厂:{args.factory}")
    meta_f = json.load(open(os.path.join(os.path.dirname(args.factory),
                                         "factory_meta.json")))
    nd_u = int(meta_f.get("u_dim", ND_U))
    fr_max = float(meta_f.get("fr_max", 5.0))
    print(f"[dataset-v3] 设计维 {nd_u}  条件维 8  Fr∈[0,{fr_max}]")
    rng = np.random.default_rng(args.seed)
    tr_b, va_b, te_b = split_by_bid([b["bid"] for b in blocks], rng)
    print(f"[dataset-v3] 块 {len(blocks)} → 束 训练{len(tr_b)}/验证{len(va_b)}/测试{len(te_b)}")

    buckets = {"tr": ([], []), "va": ([], []), "te": ([], [])}
    nfeas = ntot = 0
    kill = dict(gcap=0, smax=0, so=0, mo=0, rebound=0, tip=0, stop=0)
    for blk in blocks:
        Y = to_Y(blk); U = np.array(blk["U"], float)
        tag = "tr" if blk["bid"] in tr_b else ("va" if blk["bid"] in va_b else "te")
        crng = np.random.default_rng(90_000 + blk["cid"])
        for _ in range(args.nreq):
            gcap, smax, pc, ls = draw_req(crng)
            fe = feasible_mask_v3(Y, gcap, smax, pc, ls)
            ntot += len(fe); nfeas += int(fe.sum())
            kill["gcap"] += int((np.nan_to_num(Y[:, iAR], nan=np.inf) > gcap).sum())
            kill["smax"] += int((np.nan_to_num(Y[:, iLS], nan=np.inf) > smax).sum())
            kill["so"] += int((np.nan_to_num(Y[:, iSO], nan=1.) > .5).sum())
            kill["mo"] += int((np.nan_to_num(Y[:, iMO], nan=1.) > .5).sum())
            kill["tip"] += int((np.nan_to_num(Y[:, iPM], nan=1e9) > pc).sum())
            kill["stop"] += int((np.nan_to_num(Y[:, iSD], nan=np.inf) > ls).sum())
            v0_ = Y[:, iV0]
            h0_ = np.maximum(np.nan_to_num(v0_, nan=0.) ** 2 / (2 * G), 1e-12)
            kill["rebound"] += int((np.nan_to_num(Y[:, iREB], nan=0.) / h0_ > REB_CAP).sum())
            if not fe.any():
                continue
            idx = np.where(fe)[0]
            front = pareto2(Y[idx, iAR], Y[idx, iLS])       # a_res vs 腿行程
            for j in idx[front][:args.ktop]:
                buckets[tag][0].append(cvec_v3(blk["m"], blk["v0"], blk["kc"],
                                               gcap, smax, blk["Fr"][j], pc, ls))
                buckets[tag][1].append(U[j])

    c_lo, c_hi = c_bounds(meta_f, fr_max)
    arrs = {}
    for k in ("tr", "va", "te"):
        C = np.array(buckets[k][0], float).reshape(-1, len(c_lo))
        U = np.array(buckets[k][1], float).reshape(-1, nd_u)
        arrs[f"C_{k}"], arrs[f"U_{k}"] = C, U
        print(f"  {k}: {len(C):6d} 对")
    print("[dataset-v3] 可行 %d / %d (%.1f%%)  逐闸(可叠加): "
          "gcap %.0f%% smax %.0f%% 细长比 %.0f%% 质量 %.0f%% 回弹 %.0f%% 翻倒 %.0f%% 停车 %.0f%%"
          % (nfeas, ntot, 100 * nfeas / max(ntot, 1),
             *[100 * kill[k] / max(ntot, 1)
               for k in ("gcap", "smax", "so", "mo", "rebound", "tip", "stop")]))
    if nfeas < 0.02 * ntot:
        print("[dataset-v3] ⚠ 可行率低于 2%——先看哪条闸在杀人，别急着训练。")
    np.savez(os.path.join(args.out, "dataset.npz"), **arrs)

    paths = {}
    for blk in blocks:
        if blk["kind"] != "path":
            continue
        paths.setdefault(str(blk["bid"]), dict(walk=blk["walk"], steps=[]))
        paths[str(blk["bid"])]["steps"].append(
            dict(step=blk["step"], m=blk["m"], v0=blk["v0"], kc=blk["kc"], cid=blk["cid"],
                 split=("tr" if blk["bid"] in tr_b else
                        "va" if blk["bid"] in va_b else "te")))
    for v in paths.values():
        v["steps"].sort(key=lambda s: s["step"])
    json.dump(paths, open(os.path.join(args.out, "paths.json"), "w"),
              indent=2, ensure_ascii=False)

    json.dump(dict(c_order=C_ORDER, c_lo=c_lo, c_hi=c_hi, u_dim=nd_u,
                   arm=meta_f["arm"], v3=True, prior=meta_f["prior"], keys=KEYS_V3,
                   gcap_range=GCAP_RANGE_V3, smax_range=SMAX_RANGE_V3,
                   pitch_caps=list(PITCH_CAPS), lstop_range=list(LSTOP_RANGE),
                   nreq=args.nreq, ktop=args.ktop, split_by="bid",
                   n_bundles=dict(tr=len(tr_b), va=len(va_b), te=len(te_b)),
                   bids=dict(tr=sorted(tr_b), va=sorted(va_b), te=sorted(te_b)),
                   feas_rate=float(nfeas / max(ntot, 1)),
                   note="v3 轮足；设计 u14 = 10 维腿 + 4 维轮；物理设计 = WheelPrior.expand(u,m)"),
              open(os.path.join(args.out, "dataset_meta.json"), "w"),
              indent=2, ensure_ascii=False)
    print(f"[dataset-v3] 可行率 {100*nfeas/max(ntot,1):.1f}%  路径束 {len(paths)} → {args.out}")


if __name__ == "__main__":
    main()

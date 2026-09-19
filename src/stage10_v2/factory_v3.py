# -*- coding: utf-8 -*-
"""v3 数据工厂：轮足（双镜像腿 + B+C 轮）。骨架照抄 factory_v2，四处不同：

 1. 先验 = WheelPrior（14 维：10 维腿 BioPrior + 4 维轮工程盒，盒界见 physics_v3，
    由 v3_grid 正确符号相图定案）；
 2. 评价 = physics_v3.eval_v3（双镜像腿、俯仰自由、滚动阻力、停车距离；npass=1，
    v3 口径——腿按 m/2 定尺不做质量回代，mass_frac 落盘供事后复判）；
 3. Fr ∈ [0, 5]（v2.5 主工厂是 [0,2]），精确取零比例降为 20%（v3 的主场就是滑跑）；
 4. 落盘 KEYS_V3（含 pitch/stop/latch 等轮足量）。gcap/smax/pitch_cap/Lstop 仍为
    事后抽签的设计要求，不进仿真（省算力设计沿用 v1 §1）。

预算参考：eval_v3 单次 ~10–17 s（T=2 s 停车窗口）。默认 375 块 × 96 设计 = 36k 次，
128 workers 约 1.5 h。

用法（A100，单行）：
  OMP_NUM_THREADS=1 python src/stage10_v2/factory_v3.py --workers 128 --out outputs/v3_data_bio
产出：factory.jsonl（逐块追加，可断点续跑）+ factory_meta.json
"""
from __future__ import annotations

import argparse, json, os, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import froude as FR                                    # noqa: E402
from physics_v3 import (WheelPrior, eval_v3, PITCH_CAPS, LSTOP_RANGE,   # noqa: E402
                        WB_RANGE, TAU_RANGE, RW_RANGE, OMTH_RANGE)
from factory_v2 import (M_RANGE, V0_RANGE, KC_RANGE, v0_cap, draw_v0,   # noqa: E402
                        zeta_of_kc, lhs, loguni)

FR_MAX_V3 = 5.0            # v3_grid：wb≥0.8 全 Fr 存活，包线实测到 5
FR_ZERO_FRAC_V3 = 0.20     # 垂直着陆保基线；主场是滑跑，比 v2.5 的 40% 低

# 落盘指标（顺序固定；None = 该次评价 fail 或量非有限）
KEYS_V3 = ["a_res", "peak_a", "stroke", "leg_stroke", "sink", "rebound",
           "pitch_max_deg", "pitch_end_deg", "roll_dist_m", "stop_dist_m",
           "stopped", "stop_extrap", "vx_end", "wheel_spin_max",
           "latch_f", "latch_r", "leg_mass_kg", "gear_mass_kg", "mass_frac",
           "struct_over", "mass_over", "D1_mm", "D2_mm", "D3_mm",
           "L1_mm", "L2_mm", "L3_mm", "v0", "v_x", "mu_ground"]


def _eval_one(a):
    x14, m, v0, kc, zc, v_x = a
    try:
        r = eval_v3(tuple(x14), m, v0, kc, zeta_c=zc, v_x=v_x)
        if r.get("fail") == "solver":                  # 一次半步长重试
            r = eval_v3(tuple(x14), m, v0, kc, zeta_c=zc, v_x=v_x, h=1e-4)
    except Exception as e:                             # 无人值守，子进程异常不打断整批
        return [None] * len(KEYS_V3) + [type(e).__name__]
    if r is None or r.get("fail"):
        return [None] * len(KEYS_V3) + [(r or {}).get("fail", "none")]
    out = []
    for k in KEYS_V3:
        v = r.get(k, np.nan)
        v = float(v) if not isinstance(v, bool) else float(bool(v))
        out.append(v if np.isfinite(v) else None)
    return out + ["ok"]


def make_global_blocks(n, nd, prior, rng):
    C = lhs(n, 3, rng)
    out = []
    for i in range(n):
        m = float(loguni(C[i, 0], M_RANGE))
        kc = float(loguni(C[i, 2], KC_RANGE))
        v0 = draw_v0(C[i, 1], kc)
        U = lhs(nd, prior.ndim, np.random.default_rng(30_000 + i))
        out.append(dict(kind="global", walk=None, step=0, m=m, v0=v0, kc=kc, U=U))
    return out


WALK_MIX_V3 = [("v0", 8), ("m_allo", 6), ("m_iso", 4), ("kc", 4), ("fr", 3)]
# 新增 fr 走法：同一批设计沿 Fr 走 —— v3 的主效应路径（v2.5 里 Fr 是块内抽签，
# 没有束级路径；轮足的边界穿越首先发生在 Fr 轴上，值得单列）。


def make_path_bundles(npath, nd, K, prior, rng, mix=WALK_MIX_V3):
    plan = []
    tot = sum(w for _, w in mix)
    for name, w in mix:
        plan += [name] * max(1, round(npath * w / tot))
    plan = plan[:npath] if len(plan) >= npath else plan + [mix[0][0]] * (npath - len(plan))
    out = []
    for bi, walk in enumerate(plan):
        arng = np.random.default_rng(60_000 + bi)
        U0 = lhs(nd, prior.ndim, arng)
        m0 = float(loguni(arng.random(), M_RANGE))
        kc0 = float(loguni(arng.random(), KC_RANGE))
        v00 = draw_v0(arng.random(), kc0)
        fr0 = float(arng.uniform(0, FR_MAX_V3))
        for t in range(K):
            f = t / (K - 1)
            m, v0, kc, U, fr = m0, v00, kc0, U0, fr0
            if walk == "v0":
                v0 = draw_v0(f, kc)
            elif walk == "kc":
                kc = float(loguni(f, KC_RANGE)); v0 = min(v0, v0_cap(kc))
            elif walk in ("m_allo", "m_iso"):
                m = float(loguni(f, M_RANGE))
            elif walk == "fr":
                fr = f * FR_MAX_V3
            out.append(dict(kind="path", walk=walk, step=t, bundle=bi,
                            m=m, v0=v0, kc=kc, U=U, m_anchor=m0, fr_fixed=fr))
    return out


def block_designs(blk, prior):
    if blk.get("walk") == "m_iso":
        X = prior.expand(blk["U"], blk["m_anchor"])
        U = prior.contract(X, blk["m"])
    else:
        X = prior.expand(blk["U"], blk["m"])
        U = np.asarray(blk["U"], float)
    return np.atleast_2d(U), np.atleast_2d(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bio")
    ap.add_argument("--nglobal", type=int, default=250)
    ap.add_argument("--npath", type=int, default=25)
    ap.add_argument("--K", type=int, default=5)
    ap.add_argument("--nd", type=int, default=96)
    ap.add_argument("--workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--fr-max", type=float, default=FR_MAX_V3)
    ap.add_argument("--out", default="outputs/v3_data_bio")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "factory.jsonl")

    prior = WheelPrior(args.arm)
    print(f"[factory-v3] 设计维 {prior.ndim}  质量 {M_RANGE}  Fr∈[0,{args.fr_max}] "
          f"(精确取零 {100*FR_ZERO_FRAC_V3:.0f}%)")
    rng = np.random.default_rng(args.seed)
    blocks = make_global_blocks(args.nglobal, args.nd, prior, rng)
    blocks += make_path_bundles(args.npath, args.nd, args.K, prior, rng)
    for i, b in enumerate(blocks):
        b["bid"] = b.get("bundle", -1) if b["kind"] == "path" else 10_000 + i
        b["cid"] = i
    nsim = len(blocks) * args.nd
    print(f"[factory-v3] {args.nglobal} 独立块 + {args.npath}束×{args.K}步 = "
          f"{len(blocks)} 块 × {args.nd} 设计 = {nsim} 次仿真（~{nsim*14/args.workers/3600:.1f} h）")

    done = set()
    if os.path.exists(fp):
        with open(fp) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cid"])
                except Exception:
                    pass
        print(f"[factory-v3] 续跑：已完成 {len(done)} 块")

    json.dump(dict(arm=args.arm, prior=prior.describe(), keys=KEYS_V3,
                   u_dim=prior.ndim, v3=True, mat="al7075",
                   c_phys_order=["m", "v0", "kc", "Fr"],
                   req_order=["gcap", "smax", "pitch_cap", "lstop"],
                   pitch_caps=list(PITCH_CAPS), lstop_range=list(LSTOP_RANGE),
                   wb_range=list(WB_RANGE), tau_range=list(TAU_RANGE),
                   rw_range=list(RW_RANGE), omth_range=list(OMTH_RANGE),
                   m_range=list(M_RANGE), v0_range=list(V0_RANGE),
                   kc_range=list(KC_RANGE), fr_max=args.fr_max,
                   fr_zero_frac=FR_ZERO_FRAC_V3,
                   nglobal=args.nglobal, npath=args.npath, K=args.K, nd=args.nd,
                   seed=args.seed, walk_mix=WALK_MIX_V3, npass=1,
                   note="v3 轮足工厂；bid 为切分单元；fr 走法束的 Fr 由 fr_fixed 给出"),
              open(os.path.join(args.out, "factory_meta.json"), "w"),
              indent=2, ensure_ascii=False)

    t0, ndone = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(fp, "a") as f:
        for blk in blocks:
            if blk["cid"] in done:
                continue
            U, X = block_designs(blk, prior)
            zc = zeta_of_kc(blk["kc"])
            if blk.get("walk") == "fr":                  # fr 束：整块统一 Fr
                frs = np.full(len(X), blk["fr_fixed"])
            else:
                frs = FR.sample_fr(len(X), np.random.default_rng(8_000_000 + blk["cid"]),
                                   fr_max=args.fr_max, zero_frac=FR_ZERO_FRAC_V3)
            vxs = FR.fr_to_vx(frs, blk["m"])
            Y = list(ex.map(_eval_one,
                            [(x, blk["m"], blk["v0"], blk["kc"], zc, float(vx))
                             for x, vx in zip(X, vxs)], chunksize=2))
            fails = [y[-1] for y in Y]
            f.write(json.dumps(dict(
                cid=blk["cid"], bid=blk["bid"], kind=blk["kind"], walk=blk["walk"],
                step=blk["step"], m=blk["m"], v0=blk["v0"], kc=blk["kc"], zeta_c=zc,
                Fr=np.round(frs, 5).tolist(), v_x=np.round(np.asarray(vxs), 5).tolist(),
                U=np.round(U, 5).tolist(), X=np.round(X, 4).tolist(),
                Y=[y[:-1] for y in Y], fail=fails)) + "\n")
            f.flush()
            ndone += 1
            if ndone % 5 == 0 or blk is blocks[-1]:
                el = time.time() - t0
                nbad = sum(1 for v in fails if v != "ok")
                left = el / max(ndone, 1) * (len(blocks) - len(done) - ndone) / 60
                print(f"[factory-v3] {ndone}/{len(blocks)-len(done)}  本块失败 "
                      f"{nbad}/{args.nd}  ({el:.0f}s, ~{el/max(ndone,1):.0f}s/块, "
                      f"预计剩 {left:.0f} 分钟)", flush=True)
    print(f"[factory-v3] done → {fp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

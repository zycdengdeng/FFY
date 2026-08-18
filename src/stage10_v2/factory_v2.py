"""v2 数据工厂:质量条件先验 + condition-path 预埋。

与 v1 工厂的四点不同:
 1. 设计在**无量纲 u 空间**采样,再由 `bioprior.expand(u, m)` 展开成物理设计
    —— 于是训练集里第一次真的存在 p(x|m);
 2. 物理工况扩为 (m, v0, k_c),k_c 是绝对地面刚度(ζ_c 由 k_c 单调导出,见 zeta_of_kc);
    g_cap / s_max 仍是**事后抽签的设计要求**,不进仿真(沿用 v1 §1 的省算力设计);
 3. 一部分块改成**路径束**:同一批锚点设计沿一条工况线走 K 步,
    使模型能观察 ∂y/∂c 与可行边界的穿越(《方案》§七);
 4. 每块带 `bid`(束编号)。**训练/测试必须按 bid 切,不能按 cid 切**——
    一个束把同一批设计放进 K 个不同 cid,按 cid 切会静默泄漏。

路径类型:
  v0       固定 u 与 m,扫着水速度            —— 动力学效应最强,边界穿越最清楚
  m_allo   固定 u(含 u_L)扫质量               —— 保持异速生长的工况路径:
                                               追踪"同一形态型"随体重的演化
  m_iso    固定**物理设计**扫质量              —— 反事实对照:纯物理的质量效应
  kc       固定 u 与 m,扫地面刚度              —— 环境路径
  m_allo 与 m_iso 成对存在,直接给出分解:
      dy/dm|_allo = ∂y/∂m|_x + (∂y/∂x)·(dx/dm)|_bio
      (总效应)      (m_iso)     (差值 = 生物先验的贡献)

用法(A100):
  OMP_NUM_THREADS=1 python src/stage10_v2/factory_v2.py \
      --arm bio --nglobal 375 --npath 25 --nd 120 --workers 128 \
      --out outputs/v2_data_bio
产出: factory.jsonl(逐块追加,可断点续跑) + factory_meta.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import physics_v2 as P                       # noqa: E402
from bioprior import BioPrior, ARMS          # noqa: E402

# 工况范围
M_RANGE = (1.0, 12.0)          # kg,对数均匀(尺度研究应按对数均匀铺)
V0_RANGE = (0.5, 2.0)          # m/s,均匀
KC_RANGE = P.KC_RANGE          # N/m,对数均匀,[5e4, 2e6] 内模型全程有效

# 存进 Y 的指标(顺序固定,后续一切下游都按此索引)
KEYS_V2 = ["peak_a", "stroke", "leg_stroke", "sink", "eta", "cfe", "peak_jerk",
           "E_abs", "F_peak", "rebound", "n_bounce", "t_settle",
           "leg_mass_kg", "mass_frac", "struct_over", "mass_over"]

# 路径束的构成(《方案》§七):共 25 束时的配比
WALK_MIX = [("v0", 10), ("m_allo", 6), ("m_iso", 4), ("kc", 5)]


def zeta_of_kc(kc):
    """接触阻尼比由地面刚度单调导出:越软越耗散。

    由 TERRAIN 的四个锚点(2e6→0.05, 8e5→0.08, 2e5→0.15, 5e4→0.30)对数拟合,
    ζ = 59.0·k_c^(-0.4875),再夹到 [0.05, 0.35]。把 ζ 和 k_c 绑定是为了不让
    工况空间里出现"混凝土却像泥一样耗散"这种物理上不存在的组合。
    """
    return float(np.clip(10 ** (1.771 - 0.4875 * np.log10(kc)), 0.05, 0.35))


def lhs(n, d, rng):
    X = np.empty((n, d))
    for j in range(d):
        e = (np.arange(n) + rng.random(n)) / n
        X[:, j] = rng.permutation(e)
    return X


def loguni(u, rg):
    return 10 ** (np.log10(rg[0]) + (np.log10(rg[1]) - np.log10(rg[0])) * u)


def _eval_one(a):
    x7, m, v0, kc, zc, npass = a
    r = P.eval_v2(tuple(x7), m, v0, kc=kc, zeta_c=zc, npass=npass)
    if r is None or r.get("fail"):
        return [None] * len(KEYS_V2) + [(r or {}).get("fail", "none")]
    out = []
    for k in KEYS_V2:
        v = r.get(k, np.nan)
        v = float(v) if not isinstance(v, bool) else float(bool(v))
        out.append(v if np.isfinite(v) else None)
    return out + ["ok"]


# ------------------------------------------------------------------ 块的生成
def make_global_blocks(n, nd, prior, rng):
    """独立块:工况 LHS × 设计 LHS(u 空间)。仍是覆盖全空间的主力。"""
    C = lhs(n, 3, rng)
    out = []
    for i in range(n):
        m = float(loguni(C[i, 0], M_RANGE))
        v0 = float(V0_RANGE[0] + (V0_RANGE[1] - V0_RANGE[0]) * C[i, 1])
        kc = float(loguni(C[i, 2], KC_RANGE))
        U = lhs(nd, 7, np.random.default_rng(20_000 + i))
        out.append(dict(kind="global", walk=None, step=0,
                        m=m, v0=v0, kc=kc, U=U))
    return out


def make_path_bundles(npath, nd, K, prior, rng, mix=WALK_MIX):
    """路径束:同一批锚点设计沿一条工况线走 K 步。"""
    plan = []
    tot = sum(w for _, w in mix)
    for name, w in mix:
        plan += [name] * max(1, round(npath * w / tot))
    plan = plan[:npath] if len(plan) >= npath else plan + [mix[0][0]] * (npath - len(plan))

    out = []
    for bi, walk in enumerate(plan):
        arng = np.random.default_rng(50_000 + bi)
        U0 = lhs(nd, 7, arng)                       # 锚点设计(u 空间),整束共用
        m0 = float(loguni(arng.random(), M_RANGE))
        v00 = float(V0_RANGE[0] + (V0_RANGE[1] - V0_RANGE[0]) * arng.random())
        kc0 = float(loguni(arng.random(), KC_RANGE))
        for t in range(K):
            f = t / (K - 1)                          # 0 → 1 沿路径的位置
            m, v0, kc, U = m0, v00, kc0, U0
            if walk == "v0":
                v0 = float(V0_RANGE[0] + (V0_RANGE[1] - V0_RANGE[0]) * f)
            elif walk == "kc":
                kc = float(loguni(f, KC_RANGE))
            elif walk in ("m_allo", "m_iso"):
                m = float(loguni(f, M_RANGE))
            out.append(dict(kind="path", walk=walk, step=t, bundle=bi,
                            m=m, v0=v0, kc=kc, U=U, m_anchor=m0))
    return out


def block_designs(blk, prior):
    """把块里的 u 展开成物理设计。m_iso 走法要反过来:物理设计固定,u 随 m 变。"""
    if blk.get("walk") == "m_iso":
        X = prior.expand(blk["U"], blk["m_anchor"])       # 物理设计钉在锚点质量上
        U = prior.contract(X, blk["m"])                   # 同一物理设计在本步 m 下的 u
    else:
        X = prior.expand(blk["U"], blk["m"])
        U = np.asarray(blk["U"], float)
    return np.atleast_2d(U), np.atleast_2d(X)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", default="bio", choices=list(ARMS))
    ap.add_argument("--nglobal", type=int, default=375, help="独立块数")
    ap.add_argument("--npath", type=int, default=25, help="路径束数")
    ap.add_argument("--K", type=int, default=5, help="每束的步数")
    ap.add_argument("--nd", type=int, default=120, help="每块设计数")
    ap.add_argument("--npass", type=int, default=2)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="outputs/v2_data_bio")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "factory.jsonl")

    prior = BioPrior(args.arm)
    rng = np.random.default_rng(args.seed)
    blocks = make_global_blocks(args.nglobal, args.nd, prior, rng)
    blocks += make_path_bundles(args.npath, args.nd, args.K, prior, rng)
    for i, b in enumerate(blocks):
        b["bid"] = b.get("bundle", -1) if b["kind"] == "path" else 10_000 + i
        b["cid"] = i

    nsim = len(blocks) * args.nd * args.npass
    print(f"[factory-v2] 臂={args.arm}  独立块 {args.nglobal} + 路径 {args.npath}束×{args.K}步 "
          f"= {len(blocks)} 块 × {args.nd} 设计 × {args.npass} 遍 = {nsim} 次仿真")
    print(f"[factory-v2] 路径占比 {100*args.npath*args.K/len(blocks):.0f}%  "
          f"先验: {json.dumps(prior.describe(), ensure_ascii=False)}")

    done = set()
    if os.path.exists(fp):
        with open(fp) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cid"])
                except Exception:
                    pass
        print(f"[factory-v2] 续跑:已完成 {len(done)} 块")

    json.dump(dict(arm=args.arm, prior=prior.describe(), keys=KEYS_V2,
                   c_phys_order=["m", "v0", "kc"],
                   m_range=M_RANGE, v0_range=V0_RANGE, kc_range=list(KC_RANGE),
                   nglobal=args.nglobal, npath=args.npath, K=args.K, nd=args.nd,
                   npass=args.npass, seed=args.seed, walk_mix=WALK_MIX,
                   note="设计在 u 空间采样;bid 为切分单元(路径束整体进训练或测试)"),
              open(os.path.join(args.out, "factory_meta.json"), "w"),
              indent=2, ensure_ascii=False)

    t0, ndone = time.time(), 0
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(fp, "a") as f:
        for blk in blocks:
            if blk["cid"] in done:
                continue
            U, X = block_designs(blk, prior)
            zc = zeta_of_kc(blk["kc"])
            Y = list(ex.map(_eval_one,
                            [(x, blk["m"], blk["v0"], blk["kc"], zc, args.npass)
                             for x in X], chunksize=2))
            fails = [y[-1] for y in Y]
            f.write(json.dumps(dict(
                cid=blk["cid"], bid=blk["bid"], kind=blk["kind"], walk=blk["walk"],
                step=blk["step"], m=blk["m"], v0=blk["v0"], kc=blk["kc"], zeta_c=zc,
                U=np.round(U, 5).tolist(), X=np.round(X, 4).tolist(),
                Y=[y[:-1] for y in Y], fail=fails)) + "\n")
            f.flush()
            ndone += 1
            if ndone % 10 == 0 or blk is blocks[-1]:
                el = time.time() - t0
                nbad = sum(1 for v in fails if v != "ok")
                print(f"[factory-v2] {ndone}/{len(blocks) - len(done)}  "
                      f"本块失败 {nbad}/{args.nd}  ({el:.0f}s, "
                      f"~{el/max(ndone,1):.1f}s/块, 预计剩 "
                      f"{el/max(ndone,1)*(len(blocks)-len(done)-ndone)/60:.0f} 分钟)")
    print(f"[factory-v2] done → {fp}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()

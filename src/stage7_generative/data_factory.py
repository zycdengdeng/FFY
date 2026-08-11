"""生成阶段 · 数据工厂:批量仿真,为条件生成模型自产训练数据。

设计要点(省算力的关键):
- 仿真只依赖 (m, v0) 和设计 x —— g_cap/stroke_max 是"事后筛选参数",
  不进仿真。所以工厂只扫 (m, v0) × 设计,全指标存盘;
  之后任意 (g_cap, stroke_max) 组合的训练对都能从库里免费构建,不用重仿真。
- 断点续跑:逐工况追加写 jsonl,重启自动跳过已完成工况。

用法(A100,conda env ffy;先 pip install exudyn):
  python src/stage7_generative/data_factory.py --nc 500 --nd 80 --workers 32 \
         --out outputs/gen_data
产出:outputs/gen_data/factory.jsonl(每行一个工况:c + 80 设计全指标)
     outputs/gen_data/factory_meta.json
预计:500×80=4 万次仿真,32 核 ≈ 30–60 分钟。
"""
from __future__ import annotations
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "stage6_surrogate"))
import models as M                      # noqa: E402
from hf_exudyn import exu_eval, SCEN_BIRD_X, NAN_METRICS  # noqa: E402

KEYS = list(NAN_METRICS.keys())
# 工况范围(v1 方案 §1)
C_LO = dict(m=1.0, v0=0.5)
C_HI = dict(m=12.0, v0=2.0)
KAPPA = 4.0


def lhs(n, d, rng):
    X = np.empty((n, d))
    for j in range(d):
        e = (np.arange(n) + rng.random(n)) / n
        X[:, j] = rng.permutation(e)
    return X


def _eval_one(args):
    x, m, v0 = args
    sc = M.bird_size_x({**SCEN_BIRD_X, "m": m, "v0": v0, "kappa": KAPPA}, x)
    r = exu_eval(tuple(x[:3]), sc)
    return [float(r[k]) if np.isfinite(r[k]) else None for k in KEYS]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nc", type=int, default=500, help="工况数")
    ap.add_argument("--nd", type=int, default=80, help="每工况设计数")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--dim", type=int, default=3, choices=[3, 7],
                    help="3=几何(v1);7=几何+刚度/阻尼解耦(E4)")
    ap.add_argument("--out", default="outputs/gen_data")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    fp = os.path.join(args.out, "factory.jsonl")

    # 断点续跑:已完成的工况 id
    done = set()
    if os.path.exists(fp):
        with open(fp) as f:
            for line in f:
                try:
                    done.add(json.loads(line)["cid"])
                except Exception:
                    pass
        print(f"[factory] resume: {len(done)} conditions already done")

    rng = np.random.default_rng(args.seed)
    C = lhs(args.nc, 2, rng)
    C[:, 0] = C_LO["m"] + (C_HI["m"] - C_LO["m"]) * C[:, 0]
    C[:, 1] = C_LO["v0"] + (C_HI["v0"] - C_LO["v0"]) * C[:, 1]

    if args.dim == 3:
        lo, hi = np.array(M.LO_BIRD), np.array(M.HI_BIRD)
        note = "3维:几何实测边界 v2;弹簧按 bird_size 规则配"
    else:
        lo, hi = np.array(M.LO_BIRD7), np.array(M.HI_BIRD7)
        note = "7维 E4:几何实测边界 + κ踝/κ膝/κ髋/ζ 解耦(bird_size_x)"
    meta = dict(nc=args.nc, nd=args.nd, seed=args.seed, kappa=KAPPA, dim=args.dim,
                keys=KEYS, c_lo=C_LO, c_hi=C_HI,
                x_lo=lo.tolist(), x_hi=hi.tolist(), note=note)
    json.dump(meta, open(os.path.join(args.out, "factory_meta.json"), "w"),
              indent=2, ensure_ascii=False)

    from concurrent.futures import ProcessPoolExecutor
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex, open(fp, "a") as f:
        for cid in range(args.nc):
            if cid in done:
                continue
            m, v0 = float(C[cid, 0]), float(C[cid, 1])
            drng = np.random.default_rng(10_000 + cid)   # 每工况独立可复现
            X = lo + (hi - lo) * lhs(args.nd, len(lo), drng)
            Y = list(ex.map(_eval_one, [(x, m, v0) for x in X], chunksize=2))
            nfail = sum(1 for y in Y if y[0] is None)
            f.write(json.dumps(dict(cid=cid, m=m, v0=v0,
                                    X=np.round(X, 4).tolist(), Y=Y)) + "\n")
            f.flush()
            if (cid + 1) % 10 == 0 or cid == args.nc - 1:
                el = time.time() - t0
                done_n = cid + 1 - len([d for d in done if d <= cid])
                print(f"[factory] {cid+1}/{args.nc}  fail={nfail}/{args.nd}  "
                      f"({el:.0f}s, ~{el/max(done_n,1):.1f}s/cond)")
    print(f"[factory] done → {fp}  ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

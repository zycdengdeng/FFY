"""E5 重测不重跑:用存档的回灌增量重建各轮池子,每轮多种子重测,产论文级轨迹带。

进化不重来(池子资产不动),只把"单种子仪表盘"换成"多种子分布带"重读历史:
  每轮 r 的教材 = 工厂 + 第 0..r-1 轮的增量(pool_increments.jsonl 里都存着)
  → 训 k 个种子 → 76 道冻结考题打分 → 均值 ± 极差带。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage7_generative/e5_remeasure.py \
    --factory outputs/gen_data7/factory.jsonl --data outputs/gen_data7/gen_dataset.npz \
    --inc outputs/gen_e5/pool_increments.jsonl --refs outputs/gen_e5/refs.json \
    --traj outputs/gen_e5/trajectory.json --out outputs/gen_e5_remeasure \
    --seeds 3 --workers 128
断点续跑:同命令,已测轮次自动跳过。约 2-3.5 小时(训练为主,CPU)。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from e5_loop import build_pairs, eval_model, load_pools     # noqa: E402
from train_cvae import fit, norm                            # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default="outputs/gen_data7/factory.jsonl")
    ap.add_argument("--data", default="outputs/gen_data7/gen_dataset.npz")
    ap.add_argument("--inc", default="outputs/gen_e5/pool_increments.jsonl")
    ap.add_argument("--refs", default="outputs/gen_e5/refs.json")
    ap.add_argument("--traj", default="outputs/gen_e5/trajectory.json",
                    help="原单种子轨迹(叠图对比用,可缺省)")
    ap.add_argument("--out", default="outputs/gen_e5_remeasure")
    ap.add_argument("--seeds", type=int, default=3, help="每轮重测种子数")
    ap.add_argument("--rounds", default="", help="逗号分隔轮次列表,空=全部")
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
    test_cids = set(ds_meta["test_cids"])
    refs = json.load(open(args.refs))

    # 增量按轮分桶
    incs = {}
    for line in open(args.inc):
        d = json.loads(line)
        incs.setdefault(d["round"], []).append(d)
    max_round = max(incs) + 1 if incs else 0
    rounds = ([int(x) for x in args.rounds.split(",") if x != ""]
              if args.rounds else list(range(max_round + 1)))
    print(f"[rm] rounds to measure: {rounds}  seeds/round: {args.seeds}")

    fp = os.path.join(args.out, "remeasure.json")
    done = json.load(open(fp)) if os.path.exists(fp) else {}

    pools = load_pools(args.factory, lo, hi)
    applied = -1                                     # 已并入增量的最大轮号
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rd in sorted(rounds):
            for k in range(applied + 1, rd):         # 增量推进到 rd 轮的教材状态
                for d in incs.get(k, []):
                    Yi = np.array([[np.nan if v is None else v for v in row]
                                   for row in d["Y"]], float)   # null → NaN,与实时路径一致
                    pools[d["cid"]].absorb(np.array(d["X"], float), Yi, lo, hi)
                applied = k
            if str(rd) in done:
                continue
            C_tr, X_tr = build_pairs(pools, test_cids, iP, iS, args.kscen)
            scores = []
            for s in range(args.seeds):
                model, _ = fit(norm(C_tr, c_lo, c_hi), norm(X_tr, lo, hi),
                               epochs=args.epochs, zdim=args.zdim,
                               seed=rd * 100 + s, verbose=False)
                sc = eval_model(model, ex, refs, gmeta, iP, iS, args.ngen_eval, keys)
                scores.append(sc)
            g = [s["median_gap"] for s in scores]
            done[str(rd)] = dict(round=rd, pairs=len(C_tr),
                                 pool=int(sum(len(p.X) for p in pools.values())),
                                 gaps=g, cov=[s["coverage"] for s in scores],
                                 fail=[s["fail"] for s in scores])
            json.dump(done, open(fp, "w"), indent=2)
            print(f"[rm] r{rd}: gaps {[f'{x*100:.0f}%' for x in g]}  "
                  f"mean {np.mean(g)*100:.1f}%  ({time.time()-t0:.0f}s)")

    # ---- 论文级图 ----
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    from matplotlib import font_manager as fm
    for p in ["/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc"]:
        if os.path.exists(p): fm.fontManager.addfont(p)
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    rds = sorted(int(k) for k in done)
    G = [np.array(done[str(r)]["gaps"]) * 100 for r in rds]
    mean = np.array([g.mean() for g in G])
    gmin = np.array([g.min() for g in G]); gmax = np.array([g.max() for g in G])

    fig, ax = plt.subplots(figsize=(8.2, 5.0), dpi=140)
    ax.fill_between(rds, gmin, gmax, color="#7F2D32", alpha=.18,
                    label=f"{args.seeds} 种子极差带")
    ax.plot(rds, mean, "-o", color="#7F2D32", lw=2.2, ms=5, label="多种子均值")
    if os.path.exists(args.traj):
        tj = json.load(open(args.traj))
        ax.plot([t["round"] for t in tj], [t["median_gap"] * 100 for t in tj],
                "--", color="#999", lw=1.3, label="原单种子读数(对比)")
    ax.set_xlabel("自提升轮次"); ax.set_ylabel("最优差距 gap 中位数 (%)")
    ax.set_title("E5 自提升循环:多种子重测轨迹(冻结考题,进化未重跑)")
    ax.legend(fontsize=9); ax.grid(alpha=.3); ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "e5_trajectory_band.png"), bbox_inches="tight")
    print(f"[rm] wrote remeasure.json + e5_trajectory_band.png → {args.out}  "
          f"({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()

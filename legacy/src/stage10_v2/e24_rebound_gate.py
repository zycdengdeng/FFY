# -*- coding: utf-8 -*-
"""E24 · 回弹软闸补挂（A 组，零新增仿真）。

发现：`physics_v2.feasible_v2` 只有四条判据 —— gcap / smax / slenderness / massbudget，
**没有回弹**。而 `hf_exudyn._metrics` 从头到尾都在算 `rebound` 和 `n_bounce`，
《回弹判据调研》v3 也早把判据定死了：

    回能比 R = rebound / (v0² / 2g)  ≤  5%      （软闸，超了判不可行）
    足端离地 n_bounce                 只记录，不判死

本脚本用工厂里已存的 `rebound` / `n_bounce` 对全部样本重打分，回答一个问题：
**现有的可行率是不是虚高？** 如果重打分打掉 >20%，训练目标本身就是错的，
必须在 v2.5 重训前把这条闸接进 `feasible_v2`。

用法：
    python src/stage10_v2/e24_rebound_gate.py
    python src/stage10_v2/e24_rebound_gate.py --factory outputs/v23_data_bio/factory.jsonl --reb-cap 0.05
"""
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from agroup_io import (load_blocks, to_Y, feasible_mask, pareto2, selfcheck,   # noqa: E402
                        iter_requirements, iP, iL, iREB, iNB, G,
                        default_factory, setup_font, CRIM, STEEL)


def reb_ratio(Y, v0):
    """回能比 = 回弹高度 / (v0²/2g)。分母是"如果完全弹回"的等效高度。"""
    h0 = v0 * v0 / (2.0 * G)
    return np.nan_to_num(Y[:, iREB], nan=np.inf) / max(h0, 1e-12)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default=default_factory(ROOT))
    ap.add_argument("--out", default=None, help="默认写到 factory 同目录")
    ap.add_argument("--reb-cap", type=float, default=0.05, help="回能比上限，默认 5%")
    ap.add_argument("--nreq", type=int, default=4, help="每块抽几组设计要求（与 dataset_v2 同口径）")
    ap.add_argument("--ktop", type=int, default=8)
    ap.add_argument("--fig", action="store_true", help="额外出一张图")
    args = ap.parse_args()
    out_dir = args.out or os.path.dirname(args.factory)
    os.makedirs(out_dir, exist_ok=True)

    selfcheck()
    blocks = load_blocks(args.factory)
    if not blocks:
        raise SystemExit("[e24] 空工厂：%s" % args.factory)
    print("[e24] 块 %d，每块 %d 个设计" % (len(blocks), len(blocks[0]["Y"])))

    n_all = n_feas = n_kill = 0          # 全样本层面
    n_tr = n_tr_kill = 0                 # 进训练集的前沿点层面（这才是真正影响模型的）
    R_feas, NB_feas = [], []
    per_block = []

    for blk in blocks:
        Y = to_Y(blk)
        R = reb_ratio(Y, blk["v0"])
        bk_all = bk_feas = bk_kill = 0
        for gcap, smax in iter_requirements(blk, args.nreq):   # 与 dataset_v2 同种子
            fe = feasible_mask(Y, gcap, smax)
            bk_all += len(fe); bk_feas += int(fe.sum())
            if not fe.any():
                continue
            idx = np.where(fe)[0]
            bk_kill += int((R[idx] > args.reb_cap).sum())
            R_feas.append(R[idx]); NB_feas.append(Y[idx, iNB])
            # 前沿层面
            pick = idx[pareto2(Y[idx, iP], Y[idx, iL])][:args.ktop]
            n_tr += len(pick)
            n_tr_kill += int((R[pick] > args.reb_cap).sum())
        n_all += bk_all; n_feas += bk_feas; n_kill += bk_kill
        per_block.append(dict(cid=blk["cid"], m=blk["m"], v0=blk["v0"], kc=blk["kc"],
                              feas=bk_feas, kill=bk_kill))

    R_feas = np.concatenate(R_feas) if R_feas else np.zeros(0)
    NB_feas = np.concatenate(NB_feas) if NB_feas else np.zeros(0)
    kill_pct = 100.0 * n_kill / max(n_feas, 1)
    tr_pct = 100.0 * n_tr_kill / max(n_tr, 1)

    print("-" * 68)
    print("样本总数           %8d" % n_all)
    print("现判据下可行       %8d  (%.1f%%)" % (n_feas, 100.0 * n_feas / max(n_all, 1)))
    print("其中回能比 > %.0f%%   %8d  (占可行的 %.1f%%)  <<< 关键数字"
          % (100 * args.reb_cap, n_kill, kill_pct))
    print("进训练集的前沿点   %8d，其中被打掉 %d (%.1f%%)" % (n_tr, n_tr_kill, tr_pct))
    print("-" * 68)
    if len(R_feas):
        qs = [50, 75, 90, 95, 99]
        print("可行样本的回能比分位： " + "  ".join(
            "P%d=%.3f" % (q, np.percentile(R_feas, q)) for q in qs))
        print("足端离地(n_bounce≥1) 占可行样本 %.1f%%（只记录，不判死）"
              % (100.0 * (NB_feas >= 1).mean()))
    print("-" * 68)
    if kill_pct > 20:
        print("结论：打掉 %.1f%% > 20%% —— **现有可行率虚高，训练目标本身是错的**。" % kill_pct)
        print("      必须在 v2.5 重训前把软闸接进 feasible_v2，否则模型学的是会弹的设计。")
    elif kill_pct > 5:
        print("结论：打掉 %.1f%%，影响可观但不致命。建议照样接上（零成本），并在报告里列出这个数。" % kill_pct)
    else:
        print("结论：打掉 %.1f%% —— 回弹本来就不是活跃约束。接上软闸是补形式上的完整，" % kill_pct)
        print("      不会改变已有结论；这本身也是一条可以写进论文的结果。")

    res = dict(reb_cap=args.reb_cap, n_all=n_all, n_feas=n_feas, n_kill=n_kill,
               kill_pct=kill_pct, n_train_front=n_tr, n_train_kill=n_tr_kill,
               train_kill_pct=tr_pct,
               reb_pct={str(q): float(np.percentile(R_feas, q)) for q in (50, 75, 90, 95, 99)}
               if len(R_feas) else {},
               liftoff_pct=float(100.0 * (NB_feas >= 1).mean()) if len(NB_feas) else 0.0,
               per_block=per_block)
    fp = os.path.join(out_dir, "e24_rebound_gate.json")
    json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % fp)

    if args.fig and len(R_feas):
        _fig(R_feas, per_block, args, out_dir)


def _fig(R_feas, per_block, args, out_dir):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    fig, ax = plt.subplots(1, 2, figsize=(11.2, 4.2))
    a = ax[0]
    a.hist(np.clip(R_feas, 0, 0.5), bins=80, color=STEEL, alpha=.85)
    a.axvline(args.reb_cap, color=CRIM, lw=2, ls="--")
    a.text(args.reb_cap * 1.15, a.get_ylim()[1] * .88,
           "软闸 %.0f%%\n右侧被打掉 %.1f%%" % (100 * args.reb_cap,
                                        100.0 * (R_feas > args.reb_cap).mean()),
           color=CRIM, fontsize=10, fontweight="bold")
    a.set_xlabel("回能比  rebound / (v0^2 / 2g)"); a.set_ylabel("可行样本数")
    a.set_title("(a) 可行样本的回弹分布", fontsize=11)

    b = ax[1]
    m = np.array([p["m"] for p in per_block]); f = np.array([p["feas"] for p in per_block])
    k = np.array([p["kill"] for p in per_block]); v0 = np.array([p["v0"] for p in per_block])
    ok = f > 0
    m, k, f, v0 = m[ok], k[ok], f[ok], v0[ok]
    pct = 100.0 * k / f
    b.scatter(m, pct, s=13, c=v0, cmap="viridis", alpha=.55, lw=0)
    ed = np.logspace(np.log10(m.min()), np.log10(m.max()) + 1e-9, 11)
    bi = np.clip(np.digitize(m, ed) - 1, 0, len(ed) - 2)
    xs, med, q1, q3 = [], [], [], []
    for j in range(len(ed) - 1):
        sel = bi == j
        if sel.sum() < 3:
            continue
        xs.append(np.sqrt(ed[j] * ed[j + 1]))
        med.append(np.median(pct[sel])); q1.append(np.percentile(pct[sel], 25))
        q3.append(np.percentile(pct[sel], 75))
    b.fill_between(xs, q1, q3, color=CRIM, alpha=.18)
    b.plot(xs, med, "-o", color=CRIM, lw=2.2, ms=5, label="中位（阴影 = 四分位区间）")
    b.set_xscale("log"); b.set_xlabel("机体质量 m / kg")
    b.set_ylabel("该工况下被回弹闸打掉的比例 / %")
    b.set_title("(b) 打掉比例随质量的变化（颜色 = 下沉速度 v0）", fontsize=11)
    b.legend(fontsize=8.5, frameon=False, loc="upper left"); b.grid(alpha=.25)
    fig.tight_layout()
    fp = os.path.join(out_dir, "e24_rebound_gate.png")
    fig.savefig(fp, dpi=170); print("→ %s" % fp)


if __name__ == "__main__":
    main()

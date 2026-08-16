"""E7 · 生成 + 打磨:同预算(9 次仿真)对决 BO-9。

问题:部署时若允许少量验证仿真,cVAE 热启动 + 局部精修能到什么水平?
协议(每道冻结考题,总预算 9 次真值仿真,与 BO-9 完全同价):
  臂 A gen4+polish5:cVAE 出 40 候选(0 仿真)→ 最远点采样挑 4 个实摔
                    → 以最优者为中心做 5 步收缩高斯精修(σ 逐步 ×0.65);
  臂 B gen9:同 40 候选挑 9 个直接实摔,不精修(消融:精修值不值)。
已知参照(同一冻结考卷):纯 cVAE(40 候选全验证口径)5.9%;BO-9 36.5%。
预算阶梯故事:0 仿真 5.9%(信模型)| 9 仿真 本实验 | BO-9 36.5%(不信模型)。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage7_generative/e7_polish.py \
    --model outputs/gen_e5c_r85/cvae_s1.pt --refs outputs/gen_e5/refs.json \
    --out outputs/gen_e7 --workers 128
产出:e7_results.json;总仿真 76 题 × 18 次 ≈ 1,400 次,分钟级。
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
from data_factory import _eval_one, KEYS                    # noqa: E402
from train_cvae import CVAE, norm                           # noqa: E402

iP, iS = KEYS.index("peak_a"), KEYS.index("stroke")


def score(y, gcap, smax):
    """比较键:(级别, 数值)。0=可行按峰值,1=违约按违约量,2=仿真失败。"""
    p, s = y[iP], y[iS]
    if not np.isfinite(p):
        return (2, np.inf)
    viol = max(0.0, p - gcap) / gcap + max(0.0, s - smax) / smax
    return (0, p) if viol == 0 else (1, viol)


def fps(Xn, k):
    """最远点采样:从质心最近点起,贪心最大化最小距离 → k 个多样候选。"""
    d0 = np.linalg.norm(Xn - Xn.mean(0), axis=1)
    idx = [int(np.argmin(d0))]
    d = np.linalg.norm(Xn - Xn[idx[0]], axis=1)
    while len(idx) < k:
        i = int(np.argmax(d))
        idx.append(i)
        d = np.minimum(d, np.linalg.norm(Xn - Xn[i], axis=1))
    return idx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/gen_e5c_r85/cvae_s1.pt")
    ap.add_argument("--refs", default="outputs/gen_e5/refs.json")
    ap.add_argument("--out", default="outputs/gen_e7")
    ap.add_argument("--ngen", type=int, default=40)
    ap.add_argument("--kinit", type=int, default=4, help="臂A初摔数")
    ap.add_argument("--kpolish", type=int, default=5, help="臂A精修步数")
    ap.add_argument("--sigma0", type=float, default=0.08, help="精修初始步长(归一化)")
    ap.add_argument("--decay", type=float, default=0.65)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=606_001)
    args = ap.parse_args(); os.makedirs(args.out, exist_ok=True)

    ck = torch.load(args.model, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    D = ck["xd"]
    model = CVAE(xd=D, cd=len(c_lo), z=ck["zdim"])
    model.load_state_dict(ck["state"]); model.eval()

    refs = json.load(open(args.refs))
    print(f"[e7] scenarios {len(refs)}  design dim {D}  "
          f"budget {args.kinit}+{args.kpolish}={args.kinit+args.kpolish}")

    # —— 每题生成 40 候选,准备两臂初始批 ——
    S = []
    for si, r in enumerate(refs):
        rng = np.random.default_rng(args.seed + si)
        torch.manual_seed(args.seed + si)
        cn = torch.tensor(norm(np.array([r["m"], r["v0"], r["gcap"], r["smax"]]),
                               c_lo, c_hi), dtype=torch.float32)
        Xn = model.sample(cn, args.ngen).numpy()             # 归一化候选
        ia = fps(Xn, args.kinit)                             # 臂A:4 个多样点
        ib = fps(Xn, args.kinit + args.kpolish)              # 臂B:9 个多样点
        S.append(dict(r=r, rng=rng, Xn=Xn, ia=ia, ib=ib, peakA=np.nan,
                      bestA=None, keyA=(3, np.inf), bestB=None, keyB=(3, np.inf)))

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        # —— 初始批:臂A 4×N + 臂B 9×N 一次打平 ——
        jobs, tags = [], []
        for si, s in enumerate(S):
            for j in s["ia"]:
                jobs.append((x_lo + (x_hi - x_lo) * s["Xn"][j], s["r"]["m"], s["r"]["v0"]))
                tags.append(("A", si, j))
            for j in s["ib"]:
                jobs.append((x_lo + (x_hi - x_lo) * s["Xn"][j], s["r"]["m"], s["r"]["v0"]))
                tags.append(("B", si, j))
        Y = list(ex.map(_eval_one, jobs, chunksize=4))
        for (arm, si, j), y in zip(tags, Y):
            s = S[si]; y = [np.nan if v is None else v for v in y]
            k = score(y, s["r"]["gcap"], s["r"]["smax"])
            if arm == "A" and k < s["keyA"]:
                s["keyA"], s["bestA"], s["peakA"] = k, s["Xn"][j].copy(), y[iP]
            if arm == "B" and k < s["keyB"]:
                s["keyB"], s["bestB"] = k, y[iP]             # B 臂只需峰值
        print(f"[e7] init batch done ({time.time()-t0:.0f}s)")

        # —— 臂A:5 步收缩精修(每步每题 1 摔,批间同步) ——
        sig = args.sigma0
        for step in range(args.kpolish):
            jobs, tags = [], []
            for si, s in enumerate(S):
                xc = s["bestA"] if s["bestA"] is not None else s["Xn"][s["ia"][0]]
                xn = np.clip(xc + s["rng"].normal(0, sig, D), 0, 1)
                jobs.append((x_lo + (x_hi - x_lo) * xn, s["r"]["m"], s["r"]["v0"]))
                tags.append((si, xn))
            Y = list(ex.map(_eval_one, jobs, chunksize=4))
            for (si, xn), y in zip(tags, Y):
                s = S[si]; y = [np.nan if v is None else v for v in y]
                k = score(y, s["r"]["gcap"], s["r"]["smax"])
                if k < s["keyA"]:
                    s["keyA"], s["bestA"], s["peakA"] = k, xn, y[iP]
            sig *= args.decay
            print(f"[e7] polish {step+1}/{args.kpolish} σ={sig:.3f} ({time.time()-t0:.0f}s)")

    rows = []
    for si, s in enumerate(S):
        ref = s["r"]["ref"]
        gA = (s["peakA"] - ref) / ref if s["keyA"][0] == 0 else 1.0
        gB = (s["bestB"] - ref) / ref if s["keyB"][0] == 0 else 1.0
        rows.append(dict(sc=si, gap_polish=float(gA), gap_gen9=float(gB),
                         feasA=s["keyA"][0] == 0, feasB=s["keyB"][0] == 0))
    gA = np.array([r["gap_polish"] for r in rows])
    gB = np.array([r["gap_gen9"] for r in rows])
    summary = dict(
        n=len(rows), budget=args.kinit + args.kpolish,
        polish_median=float(np.median(gA)), polish_mean=float(gA.mean()),
        polish_fail=int((gA >= 1.0).sum()),
        gen9_median=float(np.median(gB)), gen9_mean=float(gB.mean()),
        gen9_fail=int((gB >= 1.0).sum()),
        context=dict(pure_cvae_40="5.9%(E5c-r85 官方)", bo9="36.5%(E4)"))
    json.dump(dict(summary=summary, rows=rows),
              open(os.path.join(args.out, "e7_results.json"), "w"), indent=2)
    print("\n== E7 同预算(9 仿真)对决 ==")
    print(f"  gen4+polish5 : 中位 {np.median(gA)*100:5.1f}%  均值 {gA.mean()*100:5.1f}%  "
          f"崩盘 {summary['polish_fail']}")
    print(f"  gen9(不精修): 中位 {np.median(gB)*100:5.1f}%  均值 {gB.mean()*100:5.1f}%  "
          f"崩盘 {summary['gen9_fail']}")
    print(f"  参照:纯 cVAE 5.9% | BO-9 36.5%")
    print(f"[e7] done → {args.out}")


if __name__ == "__main__":
    main()

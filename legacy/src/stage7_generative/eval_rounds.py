"""在指定考卷上重考已存档的逐轮模型(不重训、不重跑循环)。

用途:E11 消融的 bio 臂直接复用 gen_e5 的 cvae_r0..rN,在**并集标尺**上重考,
与 wide/shift 臂同尺可比;也可用于任何"换考卷重判"的场景。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage7_generative/eval_rounds.py \
    --model-dir outputs/gen_e5 --refs outputs/gen_abl/refs_union.json \
    --rounds 0-20 --out outputs/gen_abl/bio_trajectory.json --workers 128
仿真量: (轮数+1) × 题数 × ngen
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
from data_factory import KEYS                    # noqa: E402
from e5_loop import eval_model                   # noqa: E402
from train_cvae import CVAE                      # noqa: E402

iP, iS = KEYS.index("peak_a"), KEYS.index("stroke")


def parse_rounds(spec, model_dir):
    if "-" in spec:
        a, b = spec.split("-")
        rs = list(range(int(a), int(b) + 1))
    else:
        rs = [int(v) for v in spec.split(",")]
    return [r for r in rs
            if os.path.exists(os.path.join(model_dir, f"cvae_r{r}.pt"))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", default="outputs/gen_e5")
    ap.add_argument("--refs", default="outputs/gen_abl/refs_union.json")
    ap.add_argument("--rounds", default="0-20", help="如 0-20 或 0,5,10")
    ap.add_argument("--out", default="outputs/gen_abl/bio_trajectory.json")
    ap.add_argument("--ngen-eval", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    refs = json.load(open(args.refs))
    rounds = parse_rounds(args.rounds, args.model_dir)
    if not rounds:                      # 静默空转会让上游脚本误以为成功,必须显式失败
        raise SystemExit(f"[eval] 错误:{args.model_dir} 下没有匹配 --rounds "
                         f"'{args.rounds}' 的 cvae_r*.pt 存档,无事可做")
    if not refs:
        raise SystemExit(f"[eval] 错误:考卷为空:{args.refs}")
    print(f"[eval] 存档 {len(rounds)} 轮 × 考题 {len(refs)} × {args.ngen_eval} 设计 "
          f"= {len(rounds) * len(refs) * args.ngen_eval} 次仿真")

    traj, t0 = [], time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for rd in rounds:
            ck = torch.load(os.path.join(args.model_dir, f"cvae_r{rd}.pt"),
                            map_location="cpu", weights_only=False)
            meta = ck["meta"]
            model = CVAE(xd=ck["xd"], cd=len(meta["c_lo"]), z=ck["zdim"])
            model.load_state_dict(ck["state"]); model.eval()
            sc = eval_model(model, ex, refs, meta, iP, iS, args.ngen_eval, KEYS)
            traj.append(dict(round=rd, **sc))
            json.dump(traj, open(args.out, "w"), indent=2)
            print(f"  r{rd}: gap 中位 {sc['median_gap'] * 100:5.1f}%  "
                  f"可行率 {sc['feas_rate'] * 100:3.0f}%  覆盖 {sc['coverage'] * 100:3.0f}%  "
                  f"崩盘 {sc['fail']}  ({time.time() - t0:.0f}s)")
    print(f"[eval] done → {args.out}")


if __name__ == "__main__":
    main()

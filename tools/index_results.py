"""结果索引:自动发现每个输出目录里的摘要文件,抽成一份 HEADLINE.json。

为什么要自动发现:实验目录名与摘要文件名在几个月里改过多次(gen_e5c → gen_e5c_r85,
e6_summary → ...),任何硬编码清单都会在冻结时静默漏掉一半结果。这里改成:
扫目录 → 按结构判断 → 分类摘要,漏不掉,也不需要我记住命名。

用法:
  python tools/index_results.py --out archive/v1.0-mass-invariant/HEADLINE.json
  python tools/index_results.py --roots outputs --max-mb 4 --verbose
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

SCALAR = (int, float, bool, str)
MAX_KEYS = 40          # 每个文件最多记多少个标量,防止把原始数据抄进索引
MAX_STR = 120


def digest_traj(t):
    """轮次轨迹(e5_loop/eval_rounds 产物):记首轮、最好、末轮、末5轮中位。"""
    g = [x["median_gap"] * 100 for x in t if isinstance(x, dict) and "median_gap" in x]
    if not g:
        return None
    d = dict(kind="trajectory", rounds=len(t), r0=g[0], best=min(g), last=g[-1],
             last5_median=float(np.median(g[-5:])))
    for k in ("fail", "feas_rate", "coverage", "round"):
        if isinstance(t[-1], dict) and k in t[-1]:
            d[f"{k}_last"] = t[-1][k]
    return d


def flatten(o, prefix="", out=None, depth=0):
    """把嵌套 dict 摊平成标量条目,只下探两层,并限量。"""
    out = {} if out is None else out
    if len(out) >= MAX_KEYS:
        return out
    if isinstance(o, dict):
        for k, v in o.items():
            key = f"{prefix}{k}"
            if isinstance(v, SCALAR):
                out[key] = v[:MAX_STR] if isinstance(v, str) else v
            elif isinstance(v, list) and v and all(isinstance(e, SCALAR) for e in v):
                out[key] = v[:8] if len(v) <= 8 else dict(n=len(v), head=v[:5])
            elif isinstance(v, dict) and depth < 2:
                flatten(v, key + ".", out, depth + 1)
            elif isinstance(v, list):
                out[key] = f"list[{len(v)}]"
            if len(out) >= MAX_KEYS:
                out["…"] = "truncated"
                break
    return out


def digest_json(fp):
    try:
        o = json.load(open(fp))
    except Exception as e:
        return dict(kind="unreadable", error=str(e)[:MAX_STR])
    if isinstance(o, list):
        if o and isinstance(o[0], dict) and "median_gap" in o[0]:
            return digest_traj(o)
        if o and isinstance(o[0], dict):
            return dict(kind="records", n=len(o), fields=list(o[0])[:16],
                        first=flatten(o[0]))
        return dict(kind="list", n=len(o))
    if isinstance(o, dict):
        return dict(kind="summary", **flatten(o))
    return dict(kind="scalar", value=str(o)[:MAX_STR])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--roots", nargs="*", default=["outputs"])
    ap.add_argument("--out", default="archive/v1.0-mass-invariant/HEADLINE.json")
    ap.add_argument("--max-mb", type=float, default=4.0,
                    help="超过此大小的 json 视为原始数据,不进索引")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    H, skipped, nfile = {}, [], 0
    for root in args.roots:
        if not os.path.isdir(root):
            continue
        for d in sorted(os.listdir(root)):
            dp = os.path.join(root, d)
            if not os.path.isdir(dp):
                continue
            entry, arts = {}, []
            for cur, _, files in os.walk(dp):
                for fn in sorted(files):
                    fp = os.path.join(cur, fn)
                    rel = os.path.relpath(fp, dp)
                    ext = os.path.splitext(fn)[1].lower()
                    if ext in (".png", ".pdf", ".svg", ".md", ".csv", ".tex"):
                        arts.append(rel); continue
                    if ext != ".json":
                        continue
                    sz = os.path.getsize(fp)
                    if sz > args.max_mb * 1e6:
                        skipped.append((fp, sz)); continue
                    entry[rel] = digest_json(fp); nfile += 1
            if entry or arts:
                H[dp] = dict(files=entry, artifacts=arts[:30],
                             n_ckpt=len([f for _, _, fs in os.walk(dp)
                                         for f in fs if f.endswith(".pt")]))

    json.dump(H, open(args.out, "w"), indent=2, ensure_ascii=False, default=float)
    print(f"[index] {len(H)} 个目录 / {nfile} 个 json → {args.out}")
    for dp, e in H.items():
        tr = [(k, v) for k, v in e["files"].items()
              if isinstance(v, dict) and v.get("kind") == "trajectory"]
        sm = [k for k, v in e["files"].items()
              if isinstance(v, dict) and v.get("kind") == "summary"]
        line = f"  {dp:<28} json={len(e['files']):<3} 存档={e['n_ckpt']:<3}"
        if tr:
            k, v = tr[0]
            line += f"  {k}: r0 {v['r0']:.1f}% → 末5轮 {v['last5_median']:.1f}% ({v['rounds']}轮)"
        elif sm:
            line += f"  摘要: {', '.join(sm[:3])}"
        print(line)
    if skipped:
        print(f"[index] 跳过 {len(skipped)} 个大文件(>{args.max_mb}MB,视为原始数据):")
        for fp, sz in skipped[:8]:
            print(f"    {fp}  {sz/1e6:.0f}MB")


if __name__ == "__main__":
    main()

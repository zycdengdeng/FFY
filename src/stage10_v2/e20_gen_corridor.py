# -*- coding: utf-8 -*-
"""E20 · 生成侧的体重走廊:cVAE 出的设计,可行率与力学指标随体重怎么走。

与 E18b 严格配对(同工况、同体重梯子、同判官、同样本数),唯一差别是设计从哪来:
    E18b  在设计盒里**闭眼随机抓**   → 量的是"盒子(生物先验)放对位置没有"
    E20   由 **cVAE 条件生成**       → 量的是"模型在盒子里找得准不准"
两条曲线的差 = 模型本身的贡献,与先验的贡献分开记账。

体重梯子超出 12 kg 的部分是**双重外推**,图上必须分开标:
  · 物理外推:接触模型只在 m∈[1,12] 标定过(见 E18b 的 deep_sink 自检)
  · 网络外推:训练时 log10(m) 被归一化到 [0,1](即 1–12 kg),
             95 kg 归一化后是 1.83 —— 网络输入本身就在训练范围外。
其余四个条件维(v0/kc/gcap/smax)全部落在训练范围内,所以**只有质量轴在外推**。

真实机型锚点(给听众体量感,不代表我们能给它做起落架 —— 多旋翼是垂直可控降落):
    DJI FlyCart 30:空机含双电池 65 kg · 最大起飞 95 kg · 最大载重 40 kg

用法(A100,单行):
  OMP_NUM_THREADS=1 python src/stage10_v2/e20_gen_corridor.py --workers 128 --out outputs/v2_e20

只跑一个工况看看(约 2 分钟):
  OMP_NUM_THREADS=1 python src/stage10_v2/e20_gen_corridor.py --workers 128 --nz 96 --conds turf1.2 --out outputs/v2_e20_quick
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P                              # noqa: E402
from bioprior import BioPrior                       # noqa: E402
from factory_v2 import zeta_of_kc                   # noqa: E402
from e17_emergent_b import load as load_cvae        # noqa: E402
from train_cvae import norm                         # noqa: E402
from e18b_corridor_multi import CONDS, GCAP_G, SMAX  # noqa: E402  同一套工况与要求

# 与 E18b 完全相同的体重梯子(0.5→120 kg, 16 级对数),保证两条曲线逐点可比
M_GRID = 10 ** np.linspace(np.log10(0.5), np.log10(120.0), 16)   # 可由 --mgrid 覆盖
# 额外的真实机型锚点,单独评价,不进梯子(否则两边网格不一致)
ANCHORS = {30.0: "货运无人机量级", 65.0: "FlyCart 30 空机(含双电池)",
           85.0: "FlyCart 30 半载", 95.0: "FlyCart 30 最大起飞重量"}
# 存下来的力学指标(顺序固定,下游按此索引)
MET = ["peak_g", "eta", "cfe", "peak_jerk", "leg_stroke_mm", "sink_mm",
       "leg_mass_g", "mass_frac", "F_peak"]


BASE_V21 = None   # 由 --v21 设定;None 时完全走老路径,既有结果可复现


def _base():
    if BASE_V21 is None:
        return None
    return {**P.SCEN_BIRD_X, "hip_damp_unified": True}

def _probe(a):
    """跑一个生成设计,回 (是否可行, 指标数组)。不可行也回指标,B 图要看分布。"""
    x7, m, v0, kc = a
    r = P.eval_v2(tuple(x7), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=_base())
    ok, _ = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    if r is None or r.get("fail"):
        return False, [np.nan] * len(MET)
    return bool(ok), [
        r["peak_a"] / 9.81, r.get("eta", np.nan), r.get("cfe", np.nan),
        r.get("peak_jerk", np.nan), r["leg_stroke_mm"], r["sink_mm"],
        r["leg_mass_kg"] * 1e3, r["mass_frac"], r.get("F_peak", np.nan)]


def latest_ckpt(root, arm):
    fs = glob.glob(os.path.join(root, f"v2_e5_{arm}", "cvae_r*.pt"))
    assert fs, f"找不到 {root}/v2_e5_{arm}/cvae_r*.pt"
    return max(fs, key=lambda p: int(re.search(r"cvae_r(\d+)\.pt", p).group(1)))


def gen_designs(model, meta, prior, m, v0, kc, nz, seed):
    """给定条件采 nz 个隐变量 → 解码 → 展开成物理设计 x7。不裁剪条件,外推照跑。"""
    c = np.array([np.log10(m), v0, np.log10(kc), GCAP_G * 9.81, SMAX])
    cn = norm(c, np.array(meta["c_lo"]), np.array(meta["c_hi"]))
    torch.manual_seed(seed)
    with torch.no_grad():
        u = model.sample(torch.tensor(cn, dtype=torch.float32), nz).numpy()
    return prior.expand(np.clip(u, 0.0, 1.0), float(m)), float(cn[0])


def run_arm(arm, ckpt, conds, ms, nz, workers, outdir):
    model, meta = load_cvae(ckpt)
    pr = meta["prior"]
    prior = BioPrior(arm, sigma=pr["sigma"], u_max=pr["u_max"], v21=(BASE_V21 is not None))
    print(f"[{arm}] {os.path.basename(ckpt)}  zdim={model.zdim}  "
          f"训练质量区间 10^{meta['c_lo'][0]:.3f}–10^{meta['c_hi'][0]:.3f} kg", flush=True)

    blob = dict(arm=arm, ckpt=os.path.basename(ckpt), nz=nz,
                gcap_g=GCAP_G, smax=SMAX, met=MET,
                m_grid=[round(float(v), 3) for v in ms],
                anchors={str(k): v for k, v in ANCHORS.items()},
                c_lo=meta["c_lo"], c_hi=meta["c_hi"], conds={})
    raw = {}
    for cname in conds:
        cd = CONDS[cname]
        jobs, X7, cn0 = [], [], []
        allm = list(ms) + sorted(ANCHORS)
        for mi, m in enumerate(allm):
            x, c0 = gen_designs(model, meta, prior, m, cd["v0"], cd["kc"],
                                nz, seed=1000 + mi)
            X7.append(x); cn0.append(c0)
            jobs += [(tuple(v), float(m), cd["v0"], cd["kc"]) for v in x]
        t0 = time.time()
        with ProcessPoolExecutor(max_workers=workers) as ex:
            out = list(ex.map(_probe, jobs, chunksize=4))
        okv = np.array([o[0] for o in out]).reshape(len(allm), nz)
        mv = np.array([o[1] for o in out], float).reshape(len(allm), nz, len(MET))
        print(f"[{arm}/{cname}] {len(jobs)} 次 ({time.time()-t0:.0f}s)  "
              f"可行率 " + " ".join(f"{v*100:.0f}" for v in okv.mean(1)), flush=True)

        ng = len(ms)
        blob["conds"][cname] = dict(
            label=cd["label"], kc=cd["kc"], v0=cd["v0"],
            cn_m=[round(v, 4) for v in cn0[:ng]],
            feas=[round(float(v), 4) for v in okv[:ng].mean(1)],
            anchor_feas={str(m): round(float(okv[ng + i].mean()), 4)
                         for i, m in enumerate(sorted(ANCHORS))})
        raw[f"{cname}__ok"] = okv
        raw[f"{cname}__met"] = mv
        raw[f"{cname}__x7"] = np.stack(X7, 0)

    os.makedirs(outdir, exist_ok=True)
    json.dump(blob, open(os.path.join(outdir, f"e20_{arm}.json"), "w"),
              indent=1, ensure_ascii=False)
    np.savez_compressed(os.path.join(outdir, f"e20_{arm}_raw.npz"),
                        m_all=np.array(list(ms) + sorted(ANCHORS)), **raw)
    print(f"[{arm}] → {outdir}/e20_{arm}.json (+_raw.npz)\n", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="bio")
    ap.add_argument("--outroot", default="outputs", help="v2_e5_<arm> 的父目录")
    ap.add_argument("--conds", default=",".join(CONDS))
    ap.add_argument("--nz", type=int, default=216,
                    help="每格采多少隐变量。216 约 50 分钟(128 核);"
                         "432 与 E18b 每格样本数完全对齐,但要约 1.6 小时")
    ap.add_argument("--v21", action="store_true",
                    help="用 v2.1/v2.2 物理:9 维设计(含姿态)+ 髋阻尼统一式")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--mgrid", default=None, help='"lo,hi,n" 覆盖质量网格,如 "2,40,16"')
    ap.add_argument("--anchors", default=None, help='"30:标签,20:标签" 覆盖锚点')
    ap.add_argument("--out", default="outputs/v2_e20")
    a = ap.parse_args()
    global BASE_V21
    if a.v21: BASE_V21 = True
    global M_GRID, ANCHORS
    if a.mgrid:
        lo,hi,n = a.mgrid.split(','); M_GRID = 10 ** np.linspace(np.log10(float(lo)), np.log10(float(hi)), int(n))
    if a.anchors:
        ANCHORS = {float(k): v for k, v in (t.split(':') for t in a.anchors.split(','))}
    print(f"[{__file__.split('/')[-1]}] v21 物理 = {BASE_V21 is not None}", flush=True)
    conds = [c for c in a.conds.split(",") if c in CONDS]
    assert conds, f"未知工况;可选 {list(CONDS)}"
    n = len(conds) * (len(M_GRID) + len(ANCHORS)) * a.nz
    print(f"[e20] {len(a.arms.split(','))} 臂 × {len(conds)} 工况 × "
          f"{len(M_GRID)}+{len(ANCHORS)} 级 × {a.nz} = {n} 次评价/臂\n")
    for arm in a.arms.split(","):
        run_arm(arm, latest_ckpt(a.outroot, arm), conds, M_GRID,
                a.nz, a.workers, a.out)
    print(f"[e20] 完成 → {a.out}")


if __name__ == "__main__":
    main()

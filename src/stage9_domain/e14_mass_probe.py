"""E14 · 质量无关性验尸(v1 版本冻结前的最后一份体检报告)。

背景:v1 的尺寸化规则把**每一个力**都写成了 m 的正比量——
    关节簧 k = κ·m·g·L,接触簧 kc = 4000·m·g,杆件质量 = 5%·m,重力 = m·g。
牛顿第二定律 a = F/m 中 m 上下约掉,于是"归一化响应"(peak_a[g]、stroke、η)
在 m∈[1,12] kg 上**逐位相同**,只有绝对力 F_peak = m·a 随 m 变。
本脚本把这件事从"我算过"变成"有证据、可复现、可写进论文附录"。

三项体检(总仿真量 < 700 次,单机几分钟):
  A 解码器灵敏度(0 仿真):固定 z,单独扫 c 的每一维,量解码设计移动了多少。
    预期:S(m) ≪ S(v0)——模型压根没在用 m。
  B 跨质量移植:在 m_i 下生成并选出最优设计 x*(m_i),拿去 m_j 下实摔。
    预期:交叉表每一列都相同——"为 1kg 设计的腿"在 12kg 上一样好。
  C 不变性验尸:固定设计扫 m,报 peak_a/stroke/η/F_peak 的相对离散度。
    预期:前三者 ~1e-12(数值噪声),F_peak 严格正比 m。

用法(A100):
  OMP_NUM_THREADS=1 python src/stage9_domain/e14_mass_probe.py \
    --model outputs/gen_e5c/cvae_r85.pt --out outputs/gen_e14 --workers 64
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "stage6_surrogate"))
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import models as M                                   # noqa: E402
from hf_exudyn import exu_eval, SCEN_BIRD_X, NAN_METRICS   # noqa: E402
from train_cvae import CVAE                          # noqa: E402

KEYS = list(NAN_METRICS.keys())
iP, iS, iE = KEYS.index("peak_a"), KEYS.index("stroke"), KEYS.index("eta")
C_NAMES = ["m(kg)", "v0(m/s)", "gcap(m/s2)", "smax(m)"]


def _eval_one(a):
    x, m, v0 = a
    sc = M.bird_size_x({**SCEN_BIRD_X, "m": m, "v0": v0, "kappa": 4.0}, x)
    r = exu_eval(tuple(x[:3]), sc)
    return [float(r[k]) if np.isfinite(r[k]) else None for k in KEYS]


def to_arr(rows):
    return np.array([[np.nan if v is None else v for v in r] for r in rows], float)


def load_model(fp):
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    mdl = CVAE(xd=ck["xd"], cd=len(meta["c_lo"]), z=ck["zdim"])
    mdl.load_state_dict(ck["state"]); mdl.eval()
    return mdl, meta


# --------------------------------------------------------------- A 解码器灵敏度
def probe_decoder(mdl, meta, nz=512, seed=0):
    """固定同一批 z,单独把 c 的第 j 维从 lo 推到 hi,量归一化设计 x 的位移。

    这是**零仿真**证据:如果模型学到了"重的鸟要换腿",S(m) 应与 S(v0) 同量级。
    """
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    torch.manual_seed(seed)
    z = torch.randn(nz, mdl.zdim)

    def dec(cn):
        cc = torch.tensor(cn, dtype=torch.float32).expand(nz, -1)
        with torch.no_grad():
            return mdl.dec(torch.cat([z, cc], -1)).numpy()

    mid = np.full(len(c_lo), 0.5)
    out = {}
    for j, nm in enumerate(C_NAMES[:len(c_lo)]):
        a, b = mid.copy(), mid.copy()
        a[j], b[j] = 0.0, 1.0
        Xa, Xb = dec(a), dec(b)                       # 归一化空间 [0,1]^7
        d = Xb - Xa
        out[nm] = dict(
            move_L2=float(np.linalg.norm(d, axis=1).mean()),      # 平均位移(归一化)
            move_per_dim=[float(v) for v in np.abs(d).mean(0)],
            # 物理量纲下,几何第一维 L1 的变化(mm),便于直观理解
            dL1_mm=float(((Xb[:, 0] - Xa[:, 0]) * (x_hi[0] - x_lo[0])).mean()))
    ref = max(v["move_L2"] for v in out.values()) or 1.0
    for v in out.values():
        v["relative"] = v["move_L2"] / ref
    return out


# --------------------------------------------------------------- B 跨质量移植
def probe_transplant(mdl, meta, ex, masses, v0=1.2, gcap=10 * 9.81, smax=0.024,
                     ngen=24, seed=1):
    """在 m_i 下生成→选优→拿到 m_j 下实摔,得到 |m|×|m| 交叉表。

    读法:若"对角线最优"(为自己质量设计的最好),说明条件生成有效;
          若"每一列常数"(谁设计的都一样),说明 m 这一维是死的。
    """
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    x_lo, x_hi = np.array(meta["x_lo"]), np.array(meta["x_hi"])
    torch.manual_seed(seed)

    # 1) 每个 m 生成 ngen 个设计,在自己质量下实摔,取最优可行者
    best = {}
    for m in masses:
        cn = torch.tensor((np.array([m, v0, gcap, smax]) - c_lo) / (c_hi - c_lo),
                          dtype=torch.float32)
        Xg = x_lo + (x_hi - x_lo) * mdl.sample(cn, ngen).numpy()
        Y = to_arr(list(ex.map(_eval_one, [(x, m, v0) for x in Xg], chunksize=2)))
        ok = np.isfinite(Y[:, iP]) & (Y[:, iP] <= gcap) & (Y[:, iS] <= smax)
        idx = int(np.argmin(np.where(ok, Y[:, iP], np.inf))) if ok.any() \
            else int(np.nanargmin(Y[:, iP]))
        best[m] = dict(x=Xg[idx].tolist(), feasible=bool(ok.any()),
                       own_peak_g=float(Y[idx, iP] / 9.81))

    # 2) 交叉实摔
    jobs, tag = [], []
    for md in masses:                       # 设计所针对的质量
        for mt in masses:                   # 实际测试的质量
            jobs.append((np.array(best[md]["x"]), mt, v0)); tag.append((md, mt))
    Yc = to_arr(list(ex.map(_eval_one, jobs, chunksize=2)))
    table = {}
    for (md, mt), y in zip(tag, Yc):
        table[f"{md:g}->{mt:g}"] = dict(peak_g=float(y[iP] / 9.81),
                                        stroke_mm=float(y[iS] * 1000),
                                        Fpeak_N=float(y[iP] * mt))
    # 3) 摘要:每个测试质量下,"专为它设计"相对"最好那个设计"的优势
    adv = []
    for mt in masses:
        col = np.array([table[f"{md:g}->{mt:g}"]["peak_g"] for md in masses])
        own = table[f"{mt:g}->{mt:g}"]["peak_g"]
        adv.append(dict(m_test=mt, own_peak_g=own, col_best=float(np.nanmin(col)),
                        col_spread_pct=float(100 * (np.nanmax(col) - np.nanmin(col))
                                             / max(np.nanmin(col), 1e-9)),
                        own_advantage_pct=float(100 * (np.nanmin(col) - own)
                                                / max(np.nanmin(col), 1e-9))))
    return dict(masses=list(masses), best=best, table=table, summary=adv)


# --------------------------------------------------------------- C 不变性验尸
def probe_invariance(ex, masses, ndes=24, v0=1.2, seed=2):
    """固定设计扫 m:归一化响应应逐位相同,绝对力应严格正比 m。"""
    lo, hi = np.array(M.LO_BIRD7), np.array(M.HI_BIRD7)
    rng = np.random.default_rng(seed)
    X = lo + (hi - lo) * rng.random((ndes, len(lo)))
    jobs = [(x, m, v0) for x in X for m in masses]
    Y = to_arr(list(ex.map(_eval_one, jobs, chunksize=2))).reshape(ndes, len(masses), -1)

    def spread(a):                    # 每个设计沿 m 的相对极差,取中位数
        r = (np.nanmax(a, 1) - np.nanmin(a, 1)) / np.maximum(np.abs(np.nanmedian(a, 1)), 1e-12)
        return float(np.nanmedian(r))

    F = Y[:, :, iP] * np.array(masses)[None, :]
    Fn = F / np.maximum(F[:, :1], 1e-12)                      # 相对 m=masses[0]
    return dict(masses=list(masses), ndes=ndes, v0=v0,
                rel_spread=dict(peak_a=spread(Y[:, :, iP]),
                                stroke=spread(Y[:, :, iS]),
                                eta=spread(Y[:, :, iE])),
                Fpeak_ratio_mean=[float(v) for v in np.nanmean(Fn, 0)],
                Fpeak_ratio_expected=[float(m / masses[0]) for m in masses],
                example_peak_g=[float(v) for v in Y[0, :, iP] / 9.81])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="outputs/gen_e5c/cvae_r85.pt")
    ap.add_argument("--out", default="outputs/gen_e14")
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--masses", default="1,2,4,8,12")
    ap.add_argument("--ngen", type=int, default=24)
    ap.add_argument("--ndes", type=int, default=24)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    masses = [float(v) for v in args.masses.split(",")]

    mdl, meta = load_model(args.model)
    print(f"[e14] 模型 {args.model}  xd={mdl.dec[-2].out_features} zdim={mdl.zdim}")

    print("\n=== A 解码器灵敏度(0 仿真:固定 z,单独扫每一维工况)===")
    A = probe_decoder(mdl, meta)
    print(f"{'工况维':<12}{'设计位移(归一化L2)':>20}{'相对最大':>10}{'ΔL1(mm)':>10}")
    for k, v in A.items():
        print(f"{k:<12}{v['move_L2']:>20.4f}{v['relative']:>10.3f}{v['dL1_mm']:>10.1f}")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        print("\n=== B 跨质量移植(为 m_i 设计的腿,拿到 m_j 上摔)===")
        B = probe_transplant(mdl, meta, ex, masses, ngen=args.ngen)
        hdr = "".join(f"{m:>9g}" for m in masses)
        lbl = "设计\\测试"
        print(f"{lbl:<12}{hdr}   (峰值 g)")
        for md in masses:
            row = "".join(f"{B['table'][f'{md:g}->{mt:g}']['peak_g']:>9.3f}" for mt in masses)
            print(f"{md:<12g}{row}")
        print("每一列的离散度 / 专属设计的优势:")
        for s in B["summary"]:
            print(f"  m={s['m_test']:>4g}kg  列内极差 {s['col_spread_pct']:6.2f}%   "
                  f"专属设计优势 {s['own_advantage_pct']:+6.2f}%")

        print("\n=== C 不变性验尸(固定设计扫 m)===")
        C = probe_invariance(ex, masses, ndes=args.ndes)
        for k, v in C["rel_spread"].items():
            print(f"  {k:<8} 沿 m 的相对极差(中位) = {v:.3e}")
        print(f"  F_peak/F_peak(m={masses[0]:g}) 实测 "
              f"{[round(v, 4) for v in C['Fpeak_ratio_mean']]}")
        print(f"  {'':>34}理论 {[round(v, 4) for v in C['Fpeak_ratio_expected']]}")

    res = dict(model=args.model, decoder_sensitivity=A, transplant=B, invariance=C)
    fp = os.path.join(args.out, "e14_mass_probe.json")
    json.dump(res, open(fp, "w"), indent=2, ensure_ascii=False)
    print(f"\n[e14] → {fp}")

    sm = max(v["relative"] for k, v in A.items() if k.startswith("m("))
    inv = C["rel_spread"]["peak_a"]
    print("\n[结论] " + ("质量维确认为死维:" if (sm < 0.35 and inv < 1e-6) else "存在质量依赖:")
          + f" 解码器 m 相对灵敏度 {sm:.3f}, 响应沿 m 相对极差 {inv:.1e}")


if __name__ == "__main__":
    main()

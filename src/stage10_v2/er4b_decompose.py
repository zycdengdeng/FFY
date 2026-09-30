# -*- coding: utf-8 -*-
"""E-R4b · 先验分解(锚点 × 指数 × 宽度)——E18c 绝对腿长扫描的后处理

任何先验盒:  log10 L_c(m) = log10 L0 + k·σ + b·log10(m/m0),  半宽 = w·u_max·σ
  b : 缩放指数(bio 0.391 / bio407 0.407 / geo 1/3 / elastic 1/4 / none 0)
  k : 锚点平移(σ 单位;0 = 参考类群在 m0 的典型长度)
  w : 宽度倍率(0.5 / 1 / 2)
盒内可行率 = f(m, L) 在盒内对 u 均匀平均(生成器条件盒内均匀采样,与 e18b 的 pooled 同口径)。
盒子超出扫描范围 [L_lo, L_hi] 的部分不计入并报告覆盖率;覆盖率 < 0.9 的格子标 †。

输出:markdown 表(按轻 <12 / 重 >12 / 全 4–36 kg 三段)、json、可选图。
自举:对每个 (m, L) 格子的探针重采样 B 次,配对得到差值的 95% 区间。

用法:
  python src/stage10_v2/er4b_decompose.py --scan outputs/e18c_scan --out outputs/er4b
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bioprior as BP                       # noqa: E402  (只用先验参数,不碰物理)

EXPS = {"bio": 0.39112926377807683, "bio407": 0.4067488503, "geo": 1 / 3, "elastic": 0.25, "none": 0.0}


def load_scan(d):
    files = sorted(glob.glob(os.path.join(d, "e18c_*.json")))
    assert files, f"{d} 下没有 e18c_*.json"
    conds, stamp0 = {}, None
    for fp in files:
        j = json.load(open(fp))
        if stamp0 is None:
            stamp0 = j["stamp"]
        elif any(j["stamp"].get(k) != stamp0.get(k) for k in stamp0):
            raise SystemExit(f"[er4b] stamp 不一致: {fp}")
        npz = os.path.join(d, f"e18c_{j['cond']}_probes.npz")
        code = np.load(npz)["code"] if os.path.exists(npz) else None
        conds[j["cond"]] = dict(f=np.array(j["f"]), code=code, a_med=np.array(j["a_res_med_feas"]),
                                a_min=np.array(j["a_res_min_feas"]), r_invalid=np.array(j["r_invalid"]))
    m = np.array(j["m_grid"]); L = np.array(j["L_grid"])
    return conds, m, L, stamp0


def box_mean(f_row, logL, lc_log, hw, nu=41):
    """单个质量档:f_row 在 log10 L 网格上的值;盒中心 lc_log,半宽 hw(log10)。"""
    u = np.linspace(lc_log - hw, lc_log + hw, nu)
    inside = (u >= logL[0]) & (u <= logL[-1])
    if inside.sum() == 0:
        return np.nan, 0.0
    return float(np.interp(u[inside], logL, f_row).mean()), float(inside.mean())


def eval_box(F, m, L, b, k, w, pr):
    """F: (nm, nl) 可行率(已按工况平均或单工况)。

    返回 (盒内条件可行率 p̃, 覆盖率 γ, 下界, 上界),逐质量档。
    p̃ 只是"已扫描部分中的可行率";完整先验可行率 p_F 满足
        γ·p̃ ≤ p_F ≤ γ·p̃ + (1−γ)        (S9,2026-09-30 按外部审计加入)
    γ<1 时结论只许用上下界区间,不许用 p̃ 冒充完整值。"""
    logL = np.log10(L)
    l0 = np.log10(pr.l1_center(BP.M_REF_KG))
    out, cov = np.zeros(len(m)), np.zeros(len(m))
    for i, mi in enumerate(m):
        lc = l0 + k * pr.sigma + b * np.log10(mi / BP.M_REF_KG)
        out[i], cov[i] = box_mean(F[i], logL, lc, w * pr.u_max * pr.sigma)
    lo = cov * out
    hi = cov * out + (1.0 - cov)               # 零覆盖 ⇒ [0,1] 无信息界,质量档不会被 nanmean 剔除
    return out, cov, lo, hi


def bands(m):
    return {"轻<12": m < 12 - 1e-9, "重>12": m > 12 + 1e-9, "全4–36": np.ones(len(m), bool)}


def summarize(F, m, L, pr, configs):
    rows = {}
    for name, (b, k, w) in configs.items():
        v, cov, lo, hi = eval_box(F, m, L, b, k, w, pr)
        rows[name] = {bn: (float(np.nanmean(v[msk])), float(cov[msk].min()),
                           float(np.nanmean(lo[msk])), float(np.nanmean(hi[msk])))
                      for bn, msk in bands(m).items()}
    return rows


def md_table(title, rows, keys):
    s = [f"\n### {title}", "| 配置 | 轻 <12 kg | 重 >12 kg | 全 4–36 kg |", "|---|---|---|---|"]
    for k in keys:
        r = rows[k]
        def cell(bn):
            v, cv, lo, hi = r[bn]
            if cv >= 0.999:
                return f"{v:.3f}"
            return f"[{lo:.3f},{hi:.3f}]†"      # 覆盖不足:只报 S9 上下界
        s.append(f"| {k} | {cell('轻<12')} | {cell('重>12')} | {cell('全4–36')} |")
    return "\n".join(s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scan", default="outputs/e18c_scan")
    ap.add_argument("--out", default="outputs/er4b")
    ap.add_argument("--anchors", default="-1,0,1,2")
    ap.add_argument("--widths", default="0.5,1,2")
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--min-feas", type=int, default=5, help="峰值口径:该 (m,L) 至少多少个可行探针才参与")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    conds, m, L, stamp = load_scan(a.scan)
    pr = BP.BioPrior("bio", v21=True, v25=True)
    if abs(stamp.get("m_ref_kg", BP.M_REF_KG) - BP.M_REF_KG) > 1e-6 or abs(stamp.get("u_max", pr.u_max) - pr.u_max) > 1e-9:
        raise SystemExit(f"[er4b] 扫描 stamp 的 m_ref/u_max 与当前 bioprior 不一致: {stamp}")
    print(f"[er4b] 工况 {list(conds)}  m {m.round(2).tolist()}  L {L.round(0).tolist()}")
    print(f"[er4b] stamp: {json.dumps(stamp, ensure_ascii=False)}")
    Fmean = np.mean([c["f"] for c in conds.values()], 0)

    ks = [float(x) for x in a.anchors.split(",")]
    ws = [float(x) for x in a.widths.split(",")]
    cfg_exp = {f"b={n}({b:.3f}) k=0 w=1": (b, 0.0, 1.0) for n, b in EXPS.items()}
    cfg_anc = {f"b=bio407 k={k:+.1f}σ w=1": (EXPS["bio407"], k, 1.0) for k in ks}
    cfg_wid = {f"b=bio407 k=0 w={w:g}": (EXPS["bio407"], 0.0, w) for w in ws}
    cfg_none_anc = {f"b=none k={k:+.1f}σ w=1": (0.0, k, 1.0) for k in ks}
    allcfg = {**cfg_exp, **cfg_anc, **cfg_wid, **cfg_none_anc}

    md = [f"# E-R4b 先验分解(数据 {a.scan};六工况平均;† = 盒子超出扫描范围,该格只报 S9 上下界 [γp̃, γp̃+1−γ])",
          f"stamp: `{json.dumps(stamp, ensure_ascii=False)}`",
          f"m0 = {BP.M_REF_KG:.1f} kg,L0 = {pr.l1_center(BP.M_REF_KG):.1f} mm,σ = {pr.sigma:.4f} log10,u_max = {pr.u_max}"]
    rows = summarize(Fmean, m, L, pr, allcfg)
    md.append(md_table("① 同锚点同宽度,改指数", rows, list(cfg_exp)))
    md.append(md_table("② 同指数(0.407)同宽度,改锚点", rows, list(cfg_anc)))
    md.append(md_table("③ 同指数同锚点,改宽度", rows, list(cfg_wid)))
    md.append(md_table("④ 不缩放(b=0)下改锚点(工程'一根长度到底'能否靠选对长度补救)", rows, list(cfg_none_anc)))

    # 分工况表(只给指数臂与锚点臂的全段值;覆盖<1 的格子按 S9 界报)
    md.append("\n### ⑤ 分工况(全 4–36 kg;γ<1 的格子报 [S9 下界,上界]†)")
    md.append("| 工况 | " + " | ".join(list(cfg_exp) + list(cfg_anc)) + " |")
    md.append("|---|" + "---|" * (len(cfg_exp) + len(cfg_anc)))
    per_cond = {}
    for cn, c in conds.items():
        r = summarize(c["f"], m, L, pr, {**cfg_exp, **cfg_anc})
        per_cond[cn] = {k: (v["全4–36"][0], v["全4–36"][1], v["全4–36"][2], v["全4–36"][3]) for k, v in r.items()}
        def _pc(k):
            v, cv, lo_, hi_ = per_cond[cn][k]
            return f"{v:.3f}" if cv >= 0.999 else f"[{lo_:.3f},{hi_:.3f}]†"
        md.append(f"| {cn} | " + " | ".join(_pc(k) for k in list(cfg_exp) + list(cfg_anc)) + " |")

    # 配对自举:主对比的差值区间
    boot = {}
    pairs = [("bio407 − geo", cfg_exp[f"b=bio407({EXPS['bio407']:.3f}) k=0 w=1"], cfg_exp[f"b=geo({EXPS['geo']:.3f}) k=0 w=1"]),
             ("bio407 − none", cfg_exp[f"b=bio407({EXPS['bio407']:.3f}) k=0 w=1"], cfg_exp[f"b=none({0.0:.3f}) k=0 w=1"]),
             ("k=+1σ − k=0", cfg_anc["b=bio407 k=+1.0σ w=1"], cfg_anc["b=bio407 k=+0.0σ w=1"]),
             ("k=−1σ − k=0", cfg_anc["b=bio407 k=-1.0σ w=1"], cfg_anc["b=bio407 k=+0.0σ w=1"]),
             ("w=0.5 − w=1", cfg_wid["b=bio407 k=0 w=0.5"], cfg_wid["b=bio407 k=0 w=1"]),
             ("w=2 − w=1", cfg_wid["b=bio407 k=0 w=2"], cfg_wid["b=bio407 k=0 w=1"])]
    codes = [c["code"] for c in conds.values()]
    if all(c is not None for c in codes) and a.boot > 0:
        rng = np.random.default_rng(0)
        nprobe = codes[0].shape[2]
        md.append(f"\n### ⑥ 配对自举 95% 区间(B={a.boot},探针重采样;仅对两臂覆盖率均=1 的段有效,覆盖不足段以 S9 区间为准)")
        md.append("| 对比 | 轻 <12 | 重 >12 | 全 4–36 |")
        md.append("|---|---|---|---|")
        for name, c1, c2 in pairs:
            d = {bn: [] for bn in bands(m)}
            for _ in range(a.boot):
                idx = rng.integers(0, nprobe, nprobe)
                Fb = np.mean([(c[:, :, idx] == 0).mean(2) for c in codes], 0)
                v1 = eval_box(Fb, m, L, *c1, pr)[0]; v2 = eval_box(Fb, m, L, *c2, pr)[0]
                for bn, msk in bands(m).items():
                    d[bn].append(np.nanmean(v1[msk]) - np.nanmean(v2[msk]))
            boot[name] = {bn: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for bn, v in d.items()}
            cell = lambda bn: f"[{boot[name][bn][0]:+.4f}, {boot[name][bn][1]:+.4f}]"
            md.append(f"| {name} | {cell('轻<12')} | {cell('重>12')} | {cell('全4–36')} |")

    # 两种口径的"最优腿长"曲线(生成器无关)
    md.append("\n### ⑦ 各质量档最优腿长(六工况平均;峰值口径只计可行探针≥min-feas 的格子)")
    md.append("| m (kg) | L*_feas (mm) | f_best | L*_peak-med (mm) | L*_peak-min (mm) | bio407 中心 | none 中心 |")
    md.append("|---|---|---|---|---|---|---|")
    Lfe, Lpm, Lpn = [], [], []
    nfeas = np.sum([np.sum(c["code"] == 0, 2) if c["code"] is not None else np.zeros_like(c["f"]) for c in conds.values()], 0)
    Amed = np.nanmean([np.where(c["a_med"] > 0, c["a_med"], np.nan) for c in conds.values()], 0)
    Amin = np.nanmean([np.where(c["a_min"] > 0, c["a_min"], np.nan) for c in conds.values()], 0)
    for i, mi in enumerate(m):
        j = int(np.argmax(Fmean[i])); Lfe.append(L[j])
        ok = nfeas[i] >= a.min_feas * len(conds)
        jm = int(np.nanargmin(np.where(ok, Amed[i], np.nan))) if ok.any() and np.isfinite(np.where(ok, Amed[i], np.nan)).any() else -1
        jn = int(np.nanargmin(np.where(ok, Amin[i], np.nan))) if ok.any() and np.isfinite(np.where(ok, Amin[i], np.nan)).any() else -1
        Lpm.append(L[jm] if jm >= 0 else np.nan); Lpn.append(L[jn] if jn >= 0 else np.nan)
        cb = pr.l1_center(BP.M_REF_KG) * (mi / BP.M_REF_KG) ** EXPS["bio407"]
        md.append(f"| {mi:.1f} | {L[j]:.0f} | {Fmean[i, j]:.3f} | {Lpm[-1]:.0f} | {Lpn[-1]:.0f} | {cb:.0f} | {pr.l1_center(BP.M_REF_KG):.0f} |")
    def slope(y):
        y = np.array(y); ok = np.isfinite(y)
        return float(np.polyfit(np.log(m[ok]), np.log(y[ok]), 1)[0]) if ok.sum() >= 3 else np.nan
    md.append(f"\n对数斜率(描述性):b_feas = {slope(Lfe):.3f};b_peak-med = {slope(Lpm):.3f};b_peak-min = {slope(Lpn):.3f}")

    open(os.path.join(a.out, "er4b_tables.md"), "w", encoding="utf-8").write("\n".join(md) + "\n")
    json.dump(dict(stamp=stamp, m_grid=m.tolist(), L_grid=L.tolist(), rows=rows, per_cond=per_cond, boot=boot,
                   L_feas=[float(v) for v in Lfe], L_peak_med=[float(v) for v in Lpm], L_peak_min=[float(v) for v in Lpn],
                   F_mean=Fmean.round(4).tolist()),
              open(os.path.join(a.out, "er4b_results.json"), "w"), indent=1, ensure_ascii=False)
    print("\n".join(md))
    print(f"\n[er4b] → {a.out}/er4b_tables.md, er4b_results.json")


if __name__ == "__main__":
    main()

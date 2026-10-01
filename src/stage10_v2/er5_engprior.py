# -*- coding: utf-8 -*-
"""E-R5 · ENG-prior 工程长度先验对照——"鸟类先验值多少次仿真"

协议(预注册 2026-10-01,stilt/2_证据/工程侧_ENG-prior_协议_v1.md;规则 R-H):
  开发集 = 独立的 E18c 扫描(outputs/e18c_dev:质量 4.5–30 kg 5 档、工况 soil/loam × 1.3/1.7 m/s、探针种子 1),
  测试集 = 正式 E18c 扫描(outputs/e18c_scan_fix)。两者 stamp 必须一致。
  工程先验与鸟先验同形式、同宽度:log10 L_c(m) = log10 α + b·log10(m/m0),半宽 u_max·σ(σ、u_max、m0 取自 bioprior)。
  只开发 (α, b):在开发集上最大化"盒内可行率 S9 下界 γ·p̃"的质量×工况均值;并列取 b 小者。
  开发预算分级:从开发池(工况×质量×腿长×探针,共 18,240 次评价)均匀无放回抽 N 次(默认 100…全部),每级 R 次重抽;
  抽到的格子按可行比例估计 f,没抽到的格子沿腿长轴线性插值,整行无数据的质量档不计入目标。
  主指标 B* = 工程先验在测试集上的 p_F(重抽均值,S9 下界)≥ bio407 的 p_F − 0.5 pp 的最小预算级。
  次要:宽度也开发(w∈{0.5,1,2}),单列;"泄漏上界" = 直接在测试集上开发(只作参考线,不入判定)。

用法:
  python src/stage10_v2/er5_engprior.py --dev outputs/e18c_dev --test outputs/e18c_scan_fix --out outputs/er5_engprior
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import bioprior as BP                                   # noqa: E402
from er4b_decompose import EXPS, load_scan              # noqa: E402
try:                                                    # 分析脚本不需要物理引擎;无 exudyn 的机器也能跑
    from evalcfg import same_stamp
except ImportError:                                     # 与 evalcfg.same_stamp 的键表逐字相同
    def same_stamp(s1, s2, keys=("physics", "v25", "foot", "planar", "mu_from_ground",
                                 "hip_damp_unified", "npass", "material", "gcap_g", "smax",
                                 "reb_cap", "slip_frac", "m_ref_kg", "u_max")):
        bad = [k for k in keys if s1.get(k) != s2.get(k)]
        return (not bad), bad

MARGIN = 0.005          # 盈亏平衡容差 0.5 pp(R-H 冻结)


# ---------------------------------------------------------------- 盒内可行率(与 er4b.box_mean 同口径,向量化)
def box_grid(F, m, L, log_alpha, bs, ws, pr, nu=41):
    """F:(nm,nl) 可行率;返回 lower[na,nb,nw,nm], ptilde, cov。盒中心 log10α + b·log10(m/m0),半宽 w·u_max·σ。"""
    logL = np.log10(L)
    lm = np.log10(np.asarray(m) / BP.M_REF_KG)                              # (nm,)
    lc = log_alpha[:, None, None] + bs[None, :, None] * lm[None, None, :]   # (na,nb,nm)
    hw = (ws * pr.u_max * pr.sigma)                                         # (nw,)
    t = np.linspace(-1, 1, nu)
    u = lc[:, :, None, :, None] + hw[None, None, :, None, None] * t          # (na,nb,nw,nm,nu)
    inside = (u >= logL[0]) & (u <= logL[-1])
    vals = np.empty_like(u)
    for i in range(len(m)):
        vals[..., i, :] = np.interp(u[..., i, :], logL, F[i])
    vals = np.where(inside, vals, 0.0)
    n_in = inside.sum(-1)
    cov = n_in / nu
    with np.errstate(invalid="ignore", divide="ignore"):
        pt = np.where(n_in > 0, vals.sum(-1) / np.maximum(n_in, 1), 0.0)
    return cov * pt, pt, cov


def develop(F_dev, m_dev, L, grid, pr):
    """在开发集上选 (α,b[,w]):最大化 S9 下界的质量均值;并列 → b 小 → w 大(更宽=更保守)→ α 小。"""
    la, bs, ws = grid
    lower, _, _ = box_grid(F_dev, m_dev, L, la, bs, ws, pr)
    obj = lower.mean(-1)                                                    # (na,nb,nw)
    best = obj.max()
    cand = np.argwhere(obj >= best - 1e-12)
    # 排序键:b 小、w 大、α 小
    cand = sorted(cand.tolist(), key=lambda ijk: (bs[ijk[1]], -ws[ijk[2]], la[ijk[0]]))
    i, j, k = cand[0]
    return dict(alpha_mm=float(10 ** la[i]), b=float(bs[j]), w=float(ws[k]), obj_dev=float(best))


def test_pf(F_test, m_test, L, alpha_mm, b, w, pr):
    lower, pt, cov = box_grid(F_test, m_test, L, np.array([np.log10(alpha_mm)]), np.array([b]), np.array([w]), pr)
    lo = lower[0, 0, 0]; up = lo + (1 - cov[0, 0, 0])
    return float(lo.mean()), float(up.mean()), float(pt[0, 0, 0].mean()), float(cov[0, 0, 0].min())


def fmean_from_codes(codes_list):
    return np.mean([(c == 0).mean(2) for c in codes_list], 0)


def subsample_dev(codes_list, N, rng, logL):
    """从开发池均匀无放回抽 N 次评价,返回 (F_dev(nm,nl), 有效质量档掩码)。"""
    nc = len(codes_list); nm, nl, npb = codes_list[0].shape
    flat = rng.choice(nc * nm * nl * npb, N, replace=False)
    ci, mi, li, pi = np.unravel_index(flat, (nc, nm, nl, npb))
    ok = np.zeros((nc, nm, nl)); cnt = np.zeros((nc, nm, nl))
    feas = np.array([codes_list[c][m, l, p] == 0 for c, m, l, p in zip(ci, mi, li, pi)], float)
    np.add.at(ok, (ci, mi, li), feas); np.add.at(cnt, (ci, mi, li), 1.0)
    with np.errstate(invalid="ignore"):
        f = np.where(cnt > 0, ok / np.maximum(cnt, 1), np.nan)
    for c in range(nc):                                     # 没抽到的格子沿腿长轴插值
        for m in range(nm):
            row = f[c, m]; have = np.isfinite(row)
            if have.any() and not have.all():
                f[c, m] = np.interp(logL, logL[have], row[have])
    F = np.nanmean(f, 0)                                    # 工况平均(有数据的工况)
    valid = np.isfinite(F).all(1)
    return F, valid


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="outputs/e18c_dev")
    ap.add_argument("--test", default="outputs/e18c_scan_fix")
    ap.add_argument("--out", default="outputs/er5_engprior")
    ap.add_argument("--levels", default="100,200,400,800,1600,3200,6400,12800,all")
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--boot", type=int, default=200)
    ap.add_argument("--alpha-grid", default="50,250,41")
    ap.add_argument("--b-grid", default="0,0.6,25")
    ap.add_argument("--allow-leak", action="store_true", help="允许 dev==test(仅烟测;结果标 LEAK)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    dev, m_dev, L_dev, st_dev = load_scan(a.dev)
    tst, m_tst, L_tst, st_tst = load_scan(a.test)
    ok, bad = same_stamp(st_dev, st_tst)
    if not ok:
        raise SystemExit(f"[er5] 开发集与测试集 stamp 不一致: {bad}")
    assert np.allclose(L_dev, L_tst), "开发/测试腿长网格必须相同"
    leak = os.path.abspath(a.dev) == os.path.abspath(a.test) or set(dev) & set(tst)
    if leak and not a.allow_leak:
        raise SystemExit("[er5] 开发集与测试集共享工况/路径——违反 R-H 分离铁律")
    if set(np.round(m_dev, 3)) & set(np.round(m_tst, 3)) and not a.allow_leak:
        raise SystemExit("[er5] 开发集与测试集质量点重合——违反 R-H 分离铁律")
    codes_dev = [c["code"] for c in dev.values()]
    codes_tst = [c["code"] for c in tst.values()]
    assert all(c is not None for c in codes_dev + codes_tst), "需要 *_probes.npz"
    nprobe = codes_dev[0].shape[2]
    pr = BP.BioPrior("bio", v21=True, v25=True)
    L = L_tst

    a0, a1, na = a.alpha_grid.split(","); b0, b1, nb = a.b_grid.split(",")
    la = np.linspace(np.log10(float(a0)), np.log10(float(a1)), int(na))
    bs = np.linspace(float(b0), float(b1), int(nb))
    grid_fixw = (la, bs, np.array([1.0]))
    grid_freew = (la, bs, np.array([0.5, 1.0, 2.0]))
    pool = len(dev) * len(m_dev) * len(L) * nprobe
    levels = [pool if s == "all" else int(s) for s in a.levels.split(",")]
    levels = sorted(set(min(n, pool) for n in levels))
    logL = np.log10(L)

    F_tst = fmean_from_codes(codes_tst)
    # 参考臂(零开发预算):bio407 / bio / geo / none,k=0,w=1
    ref = {}
    for name, b in EXPS.items():
        ref[name] = test_pf(F_tst, m_tst, L, pr.l1_center(BP.M_REF_KG), b, 1.0, pr)
    bio_lo, bio_hi, bio_pt, bio_cov = ref["bio407"]
    # 泄漏上界(直接在测试集上开发;仅参考)
    leak_fix = develop(F_tst, m_tst, L, grid_fixw, pr)
    leak_fix["test"] = test_pf(F_tst, m_tst, L, leak_fix["alpha_mm"], leak_fix["b"], leak_fix["w"], pr)
    leak_free = develop(F_tst, m_tst, L, grid_freew, pr)
    leak_free["test"] = test_pf(F_tst, m_tst, L, leak_free["alpha_mm"], leak_free["b"], leak_free["w"], pr)

    rng = np.random.default_rng(20261001)
    results = {"fixw": {}, "freew": {}}
    for n in levels:
        reps_fix, reps_free = [], []
        for r in range(a.reps if n < pool else 1):
            F_dev, valid = subsample_dev(codes_dev, n, rng, logL)
            for grid, store in ((grid_fixw, reps_fix), (grid_freew, reps_free)):
                d = develop(F_dev[valid], np.asarray(m_dev)[valid], L, grid, pr)
                d["n_mass_used"] = int(valid.sum())
                d["test"] = test_pf(F_tst, m_tst, L, d["alpha_mm"], d["b"], d["w"], pr)
                store.append(d)
        budget = n
        for key, store in (("fixw", reps_fix), ("freew", reps_free)):
            lo = np.array([d["test"][0] for d in store]); hi = np.array([d["test"][1] for d in store])
            results[key][n] = dict(
                budget=budget, reps=len(store),
                pf_lo_mean=float(lo.mean()), pf_lo_sd=float(lo.std(ddof=1)) if len(lo) > 1 else 0.0,
                pf_lo_min=float(lo.min()), pf_hi_mean=float(hi.mean()),
                frac_reps_reach=float(np.mean(lo >= bio_pt - MARGIN)),
                alpha_mean=float(np.mean([d["alpha_mm"] for d in store])),
                alpha_sd=float(np.std([d["alpha_mm"] for d in store], ddof=1)) if len(store) > 1 else 0.0,
                b_mean=float(np.mean([d["b"] for d in store])),
                b_sd=float(np.std([d["b"] for d in store], ddof=1)) if len(store) > 1 else 0.0,
                w_mean=float(np.mean([d["w"] for d in store])),
                d_alpha_sigma=float(np.mean([(np.log10(d["alpha_mm"]) - np.log10(pr.l1_center(BP.M_REF_KG))) / pr.sigma for d in store])),
                reps_detail=store)
        print(f"[er5] N={budget:>6}  fixw p_F={results['fixw'][n]['pf_lo_mean']:.4f}±{results['fixw'][n]['pf_lo_sd']:.4f} "
              f"(α={results['fixw'][n]['alpha_mean']:.0f} mm, b={results['fixw'][n]['b_mean']:.3f})   "
              f"freew p_F={results['freew'][n]['pf_lo_mean']:.4f} (w={results['freew'][n]['w_mean']:.2f})", flush=True)

    # 配对自举(测试探针重采样):各级"重抽均值 − bio407"的 95% 区间
    boot = {}
    if a.boot > 0:
        rb = np.random.default_rng(7)
        np_t = codes_tst[0].shape[2]
        diffs = {n: [] for n in levels}
        for _ in range(a.boot):
            idx = rb.integers(0, np_t, np_t)
            Fb = np.mean([(c[:, :, idx] == 0).mean(2) for c in codes_tst], 0)
            vb = test_pf(Fb, m_tst, L, pr.l1_center(BP.M_REF_KG), EXPS["bio407"], 1.0, pr)[2]
            for n in levels:
                ve = np.mean([test_pf(Fb, m_tst, L, d["alpha_mm"], d["b"], d["w"], pr)[0] for d in results["fixw"][n]["reps_detail"]])
                diffs[n].append(ve - vb)
        boot = {n: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))] for n, v in diffs.items()}

    # 判定
    reach = [n for n in levels if results["fixw"][n]["pf_lo_mean"] >= bio_pt - MARGIN]
    B_star = results["fixw"][reach[0]]["budget"] if reach else None
    if B_star is None:
        verdict = "出口③:全预算内工程先验未达 bio407 − 0.5 pp——该开发预算未能恢复鸟类先验所含信息"
    elif B_star <= 3200:
        verdict = f"出口①:B* = {B_star} ≤ 3,200——鸟类长度信息可由极小仿真预算替代"
    else:
        verdict = f"出口②:B* = {B_star}——鸟类先验相当于 {B_star} 次仿真的开发"

    tag = " **[LEAK 烟测,无效]**" if leak else ""
    md = [f"# E-R5 ENG-prior 工程先验对照{tag}",
          f"开发集 `{a.dev}`(工况 {list(dev)},m {np.round(m_dev,2).tolist()},探针 {nprobe});测试集 `{a.test}`(工况 {list(tst)},m {np.round(m_tst,2).tolist()})",
          f"stamp: `{json.dumps(st_tst, ensure_ascii=False)}`",
          f"先验形式 log10 L_c = log10 α + b·log10(m/{BP.M_REF_KG:.0f} kg),半宽 w·{pr.u_max}·{pr.sigma:.4f};开发目标 = S9 下界 γ·p̃ 的质量×工况均值;α 网格 {a0}–{a1} mm {na} 点,b 网格 {b0}–{b1} {nb} 点",
          "", "### ⓪ 零预算参考臂(测试集 p_F,全 4–36 kg;γ<1 报 [下界,上界])",
          "| 臂 | p_F |", "|---|---|"]
    for name, (lo_, hi_, pt_, cv_) in ref.items():
        md.append(f"| {name} (b={EXPS[name]:.3f}, α=118.7 mm, w=1) | " + (f"{pt_:.4f}" if cv_ >= 0.999 else f"[{lo_:.4f},{hi_:.4f}]†") + " |")
    md += ["", f"### ① 主结果:同宽度(w=1),开发 (α,b)——测试集 p_F(S9 下界;{a.reps} 次重抽均值 ± SD;最差一次)",
           "| 开发预算 N(次评价) | p_F 均值 ± SD | 最差 | 上界均值 | 达标重抽比例 | α (mm) | b | Δα (σ) | 自举 95%(均值−bio407) |",
           "|---|---|---|---|---|---|---|---|---|"]
    for n in levels:
        r = results["fixw"][n]
        ci = f"[{boot[n][0]:+.4f},{boot[n][1]:+.4f}]" if boot else "—"
        md.append(f"| {r['budget']} | {r['pf_lo_mean']:.4f} ± {r['pf_lo_sd']:.4f} | {r['pf_lo_min']:.4f} | {r['pf_hi_mean']:.4f} | {r['frac_reps_reach']:.2f} | {r['alpha_mean']:.0f} ± {r['alpha_sd']:.0f} | {r['b_mean']:.3f} ± {r['b_sd']:.3f} | {r['d_alpha_sigma']:+.2f} | {ci} |")
    md += [f"| bio407(零预算,N=0) | {bio_pt:.4f} | — | {bio_hi:.4f} | — | 119 | 0.407 | 0 | — |",
           f"| 泄漏上界(测试集上开发,仅参考) | {leak_fix['test'][0]:.4f} | — | {leak_fix['test'][1]:.4f} | — | {leak_fix['alpha_mm']:.0f} | {leak_fix['b']:.3f} | {(np.log10(leak_fix['alpha_mm'])-np.log10(pr.l1_center(BP.M_REF_KG)))/pr.sigma:+.2f} | — |",
           "", f"**判定(R-H,容差 {MARGIN*100:.1f} pp):{verdict}**",
           "", "### ② 次要:宽度也开发(w∈{0.5,1,2})",
           "| 开发预算 N | p_F 均值 ± SD | 最差 | α (mm) | b | w 均值 |", "|---|---|---|---|---|---|"]
    for n in levels:
        r = results["freew"][n]
        md.append(f"| {r['budget']} | {r['pf_lo_mean']:.4f} ± {r['pf_lo_sd']:.4f} | {r['pf_lo_min']:.4f} | {r['alpha_mean']:.0f} ± {r['alpha_sd']:.0f} | {r['b_mean']:.3f} ± {r['b_sd']:.3f} | {r['w_mean']:.2f} |")
    md.append(f"| 泄漏上界(仅参考) | {leak_free['test'][0]:.4f} | — | {leak_free['alpha_mm']:.0f} | {leak_free['b']:.3f} | {leak_free['w']:.2f} |")
    md.append("\n注:p_F 为盒内可行率(对数均匀盒内平均,九质量档等权);'± SD'是 20 次重抽(开发噪声)的离散,自举区间只覆盖测试探针噪声,两者不可互相代替;'达标' = 该次重抽的工程先验 ≥ bio407 − 0.5 pp;Δα 为工程锚点相对鸟锚点 118.7 mm 的偏移(σ=0.0784 log10 单位)。")
    txt = "\n".join(md) + "\n"
    open(os.path.join(a.out, "er5_tables.md"), "w", encoding="utf-8").write(txt)
    json.dump(dict(stamp=st_tst, dev=a.dev, test=a.test, leak=bool(leak), levels=levels, margin=MARGIN,
                   ref=ref, bio407=dict(lo=bio_lo, hi=bio_hi, pt=bio_pt, cov=bio_cov),
                   leak_fixw=leak_fix, leak_freew=leak_free, boot={str(k): v for k, v in boot.items()},
                   B_star=B_star, verdict=verdict,
                   results={k: {str(n): v for n, v in d.items()} for k, d in results.items()}),
              open(os.path.join(a.out, "er5_results.json"), "w"), indent=1, ensure_ascii=False, default=float)
    print(txt)
    print(f"[er5] → {a.out}/er5_tables.md, er5_results.json")


if __name__ == "__main__":
    main()

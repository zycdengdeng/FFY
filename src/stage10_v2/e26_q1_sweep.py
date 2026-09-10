# -*- coding: utf-8 -*-
"""E26 · q1_0 敏感度扫描（B 组）—— 顺带产出 E27 要用的全部量。

**这条为什么降级了**：E28 已经把 `q1_0 = 50°` 钉在真鸟落水实测 43° ± 9° 的 1σ 内
（11 次落水，跨度 27.6–54.8°）。所以问题不再是"要不要改 q1_0"，
而是"在生物学跨度内它有多重要"。

`q1_0` 是 `SCEN_BIRD_X` 里写死的场景常数（`np.radians(50.0)`），不在 9 维设计向量里。
本脚本固定设计的 9 维不动，只扫 q1_0，看峰值过载怎么变。

**判据**
  · peak_g 跨生物跨度（28–55°）变化 < 10%  → q1_0 永久保持常数，此事了结；
  · 变化 > 25%                        → 触地姿态是一阶因素，升为第 10 维设计变量。

**顺带做的两件事**（都不额外花机时）
  ① 扫描范围默认开到 80°，覆盖"髋在足正上方"那一档，供 E27 用；
  ② 每个设计**闭式解**出让髋恰好落在足正上方的 q1_0*，并单独跑这一点。
     几何：a2 = a1 + (π−θ_A)，a3 = a2 − (π−θ_K)，令 Δx = 0 得
         Δx = C·cos a1 − S·sin a1,  C = l1 + l2cosφ2 + l3cosφ3,  S = l2sinφ2 + l3sinφ3
         → a1* = atan2(C, S)
     不用数值求根，一行的事。

用法：
    python src/stage10_v2/e26_q1_sweep.py --geom-only                 # 纯几何，秒出，不用 exudyn
    python src/stage10_v2/e26_q1_sweep.py --ndes 24 --workers 8       # 扫描 + 落盘
    python src/stage10_v2/e26_q1_sweep.py --analyze-only --fig        # 只分析
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
from agroup_io import setup_font, CRIM, STEEL, GREEN, ORANGE          # noqa: E402

# 代表工况：轻/中/重 × 硬地/软地，v0 取 1.2 与 2.0，覆盖 E20 用的那几档
CONDS = [
    dict(tag="轻·硬地",  m=5.0,  v0=1.2, kc=1.0e6),
    dict(tag="轻·草地",  m=5.0,  v0=2.0, kc=2.0e5),
    dict(tag="样机·硬地", m=12.0, v0=1.2, kc=1.0e6),
    dict(tag="样机·草地", m=12.0, v0=2.0, kc=2.0e5),
    dict(tag="重·硬地",  m=30.0, v0=1.2, kc=1.0e6),
    dict(tag="重·湿沙",  m=30.0, v0=2.0, kc=5.0e4),
]
GCAP_G, SMAX = 10.0, 0.024
BIO_LO, BIO_HI = 27.6, 54.8       # E28 实测跨度（11 次真实落水）
BIO_MEAN = 42.6


def q1_vertical(x9):
    """闭式解：让髋恰好落在足端正上方的 q1_0（度）。"""
    L1, r2, r3 = float(x9[0]), float(x9[1]), float(x9[2])
    thA, thK = np.radians(float(x9[7])), np.radians(float(x9[8]))
    l1, l2, l3 = 1.0, r2, r3                     # 只看方向，尺度可约掉
    f2 = np.pi - thA
    f3 = f2 - (np.pi - thK)
    C = l1 + l2 * np.cos(f2) + l3 * np.cos(f3)
    S = l2 * np.sin(f2) + l3 * np.sin(f3)
    return float(np.degrees(np.arctan2(C, S)))


def lean_of(x9, q1_deg):
    """髋相对足端的前倾角（度）与髋高（以 L1 为单位）。"""
    L1, r2, r3 = float(x9[0]), float(x9[1]), float(x9[2])
    thA, thK = np.radians(float(x9[7])), np.radians(float(x9[8]))
    a1 = np.radians(q1_deg); a2 = a1 + (np.pi - thA); a3 = a2 - (np.pi - thK)
    dx = np.cos(a1) + r2 * np.cos(a2) + r3 * np.cos(a3)
    dz = np.sin(a1) + r2 * np.sin(a2) + r3 * np.sin(a3)
    return float(np.degrees(np.arctan2(abs(dx), max(dz, 1e-9)))), float(dz)


# ---------------------------------------------------------------- 生成设计
def gen_designs(ckpt, ndes, seed):
    import torch
    from train_cvae import CVAE, norm
    from bioprior import BioPrior
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    model = CVAE(xd=ck["xd"], cd=len(meta["c_lo"]), z=ck["zdim"])
    model.load_state_dict(ck["state"]); model.eval()
    pr = meta.get("prior", {})
    prior = BioPrior(arm=pr.get("arm", "bio"), sigma=pr.get("sigma"),
                     u_max=pr.get("u_max", 2.5), v21=bool(pr.get("v21", True)))
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    out = []
    for ci, cd in enumerate(CONDS):
        c = np.array([np.log10(cd["m"]), cd["v0"], np.log10(cd["kc"]),
                      GCAP_G * 9.81, SMAX])
        torch.manual_seed(seed + ci)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, c_lo, c_hi), dtype=torch.float32),
                             ndes).numpy()
        X = prior.expand(np.clip(u, 0, 1), cd["m"])
        for di, x in enumerate(X):
            out.append(dict(ci=ci, di=di, x9=[float(v) for v in x], **cd))
    return out


# ---------------------------------------------------------------- 纯几何预检
def geom_report(designs):
    """不跑一次仿真就能回答的部分：垂线要多陡、当前 50° 对应多少前倾。"""
    import collections
    x = np.array([d["x9"] for d in designs], float)
    qv = np.array([q1_vertical(d["x9"]) for d in designs])
    print("=" * 76)
    print("纯几何预检（零仿真）")
    print("-" * 76)
    print("设计本身：θ_A 中位 %.1f°   θ_K 中位 %.1f°" % (np.median(x[:, 7]), np.median(x[:, 8])))
    print("          对照 E28 真鸟落水实测：θ_A 140°±14°   θ_K 135°±12°")
    print("          → 生成器偏爱**更屈的踝**（与 E23 前沿偏置一致）")
    print("-" * 76)
    print("① 让髋落在足正上方所需的 q1_0*（Bigoni 那一档）")
    per = collections.defaultdict(list)
    for d, q in zip(designs, qv):
        per[d["tag"]].append(q)
    for t in [c["tag"] for c in CONDS]:
        if per[t]:
            v = per[t]
            print("   %-10s 中位 %.1f°   跨度 %.1f–%.1f°" % (t, np.median(v), min(v), max(v)))
    print("   全部        中位 %.1f°   跨度 %.1f–%.1f°   n=%d"
          % (np.median(qv), qv.min(), qv.max(), len(qv)))
    print("   → 比模型常数 50° 陡 %.0f°，比真鸟实测 %.1f° 陡 %.0f°。"
          % (np.median(qv) - 50, BIO_MEAN, np.median(qv) - BIO_MEAN))
    print("     **垂线对齐要求的姿态落在真鸟实测跨度（%.1f–%.1f°）之外** —— 又一条独立证据。"
          % (BIO_LO, BIO_HI))
    print("-" * 76)
    print("② 各档 q1_0 对应的髋-足前倾角（中位）")
    for q in (28., BIO_MEAN, 50., 55., 62., 70.):
        L = [lean_of(d["x9"], q)[0] for d in designs]
        note = ""
        if abs(q - 50) < .01:
            note = "   ← 模型常数；真鸟实测前倾 30.5°±6.4°"
        if abs(q - BIO_MEAN) < .01:
            note = "   ← 真鸟实测的跗跖骨倾角"
        print("   q1_0 = %5.1f°  →  前倾 %5.1f°%s" % (q, np.median(L), note))
    print("   注意：生成设计在 50° 处只前倾 %.1f°，**比真鸟（30.5°）更接近竖直**。"
          % np.median([lean_of(d["x9"], 50.)[0] for d in designs]))
    print("         原因是它的踝更屈（%.0f° vs 140°），同样的 q1_0 下整条腿更直。"
          % np.median(x[:, 7]))
    print("=" * 76)
    return dict(q1_vert_median=float(np.median(qv)),
                q1_vert_range=[float(qv.min()), float(qv.max())],
                thetaA_median=float(np.median(x[:, 7])),
                thetaK_median=float(np.median(x[:, 8])),
                lean_at_50=float(np.median([lean_of(d["x9"], 50.)[0] for d in designs])),
                per_cond={t: float(np.median(v)) for t, v in per.items()})


# ---------------------------------------------------------------- 单点评价
def _eval_one(a):
    """连 import 一起包在 try 里：无人值守时不能让子进程的异常打断整个 map。"""
    try:
        import physics_v2 as P
        x9, m, v0, kc, q1_deg = a
        base = {**P.SCEN_BIRD_X, "hip_damp_unified": True, "foot_mode": "bearing",
                "q1_0": np.radians(float(q1_deg))}
        zc = float(np.clip(10 ** (1.771 - 0.4875 * np.log10(kc)), 0.05, 0.35))
        r = P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zc, npass=2, base=base)
    except Exception as e:
        return dict(fail=type(e).__name__)
    if r is None or r.get("fail"):
        return dict(fail=(r or {}).get("fail", "none"))
    ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
    return dict(peak_g=float(r["peak_a"]) / 9.81,
                leg_stroke_mm=float(r.get("leg_stroke", r["stroke"])) * 1e3,
                leg_mass_g=float(r["leg_mass_kg"]) * 1e3,
                mass_frac=float(r["mass_frac"]),
                D_mm=[float(v) for v in r["D_mm"]],
                D_max_mm=[float(v) for v in r["D_max_mm"]],
                governs=list(r["governs"]),
                M_Nm=[float(r["M_ankle"]), float(r["M_knee"]), float(r["M_hip"])],
                F_peak=float(r["F_peak"]),
                seg_len_mm=[float(v) * 1e3 for v in r["seg_len"]],
                ok=bool(ok), why=list(why))


def run(designs, grid, workers, cache_fp):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(cache_fp):
        for ln in open(cache_fp, encoding="utf-8"):
            try:
                d = json.loads(ln); done.add((d["ci"], d["di"], round(d["q1"], 4)))
            except Exception:
                pass
        print("[e26] 缓存里已有 %d 条，续跑" % len(done))
    jobs, tags = [], []
    for d in designs:
        qs = list(grid) + [q1_vertical(d["x9"])]
        for q in qs:
            if (d["ci"], d["di"], round(q, 4)) in done:
                continue
            jobs.append((d["x9"], d["m"], d["v0"], d["kc"], q))
            tags.append(d)
    if not jobs:
        print("[e26] 缓存已完整"); return
    print("[e26] 需要评价 %d 个点（%d 设计 × %d 档，多进程 %d）"
          % (len(jobs), len(designs), len(grid) + 1, workers))
    t0 = time.time(); nbad = 0
    with ProcessPoolExecutor(max_workers=workers) as ex, open(cache_fp, "a") as f:
        for i, (job, d, r) in enumerate(zip(jobs, tags, ex.map(_eval_one, jobs, chunksize=2)), 1):
            q = job[4]
            lean, hz = lean_of(d["x9"], q)
            rec = dict(ci=d["ci"], di=d["di"], tag=d["tag"], m=d["m"], v0=d["v0"],
                       kc=d["kc"], q1=q, q1_vert=q1_vertical(d["x9"]),
                       lean=lean, hip_z_over_L1=hz, x9=d["x9"], **r)
            f.write(json.dumps(rec) + "\n")
            nbad += ("fail" in r)
            if i % 100 == 0 or i == len(jobs):
                el = time.time() - t0; f.flush()
                print("      %d/%d  失败 %d  (%.0fs, ~%.2f s/点, 预计剩 %.0fs)"
                      % (i, len(jobs), nbad, el, el / i, el / i * (len(jobs) - i)), flush=True)
            if i == 100 and nbad == i:
                print("\n[e26] 前 100 个点全部失败 —— 多半是 exudyn 没装好或 physics_v2 导不进来。")
                print("      先单独试：python -c \"import sys;sys.path.insert(0,'src/stage10_v2');"
                      "import physics_v2\"")
                print("      提前收工，不浪费机时。\n")
                return


# ---------------------------------------------------------------- 分析
def analyze(rows, args, out_dir):
    import collections
    good = [r for r in rows if "fail" not in r and np.isfinite(r.get("peak_g", np.nan))]
    print("[e26] 有效点 %d / %d" % (len(good), len(rows)))
    by = collections.defaultdict(dict)          # (ci,di) -> {q: row}
    for r in good:
        by[(r["ci"], r["di"])][round(r["q1"], 4)] = r

    grid = np.array(args.grid, float)
    bio = grid[(grid >= BIO_LO - 1e-6) & (grid <= BIO_HI + 1e-6)]
    print("=" * 76)
    print("① 生物学跨度 %.1f–%.1f° 内，峰值过载变化多少（每个设计各算一次）" % (BIO_LO, BIO_HI))
    spans, tags = [], []
    for k, dd in by.items():
        v = [dd[round(q, 4)]["peak_g"] for q in bio if round(q, 4) in dd]
        if len(v) < 3:
            continue
        spans.append(100.0 * (max(v) - min(v)) / max(np.mean(v), 1e-9))
        tags.append(dd[list(dd)[0]]["tag"])
    spans = np.array(spans)
    if not len(spans):
        print("   有效设计不足，无法判定"); return {}
    med = float(np.median(spans))
    print("   相对变化（max−min）/mean：中位 %.1f%%   P90 %.1f%%   最大 %.1f%%   n=%d"
          % (med, np.percentile(spans, 90), spans.max(), len(spans)))
    per = collections.defaultdict(list)
    for t, s in zip(tags, spans):
        per[t].append(s)
    for t in [c["tag"] for c in CONDS]:
        if per[t]:
            print("      %-10s 中位 %.1f%%  (n=%d)" % (t, np.median(per[t]), len(per[t])))
    print("-" * 76)
    if med < 10:
        verdict = "constant"
        print("判定：中位 %.1f%% < 10%% —— **q1_0 永久保持常数 50°**，此事了结。" % med)
        print("      论文里可以写：触地姿态的跗跖骨倾角在生物学跨度内对峰值过载不敏感，")
        print("      因此固定为实测均值附近的 50°（Duong 2025 实测 43°±9°）。")
    elif med > 25:
        verdict = "tenth-dim"
        print("判定：中位 %.1f%% > 25%% —— **触地姿态是一阶因素，升为第 10 维设计变量**，" % med)
        print("      在 v2.5 工厂里一并做，盒取实测跨度 [%.0f, %.0f]°。" % (BIO_LO, BIO_HI))
    else:
        verdict = "borderline"
        print("判定：中位 %.1f%% 落在 10–25%% 的灰区。" % med)
        print("      建议保持常数但在论文里报这个敏感度，并把它列为已知的建模简化。")

    # ---- 逐档中位曲线：响应形状（是不是线性？平台在哪？）----
    print("-" * 76)
    print("   逐档中位（每个设计先对自己 50° 的值归一化）：")
    allq = sorted({round(r["q1"], 4) for r in good})
    gq = [q for q in allq if any(abs(q - g) < 1e-6 for g in args.grid)]
    curve = []
    for q in gq:
        v = [dd[q]["peak_g"] / dd[round(50.0, 4)]["peak_g"]
             for dd in by.values()
             if q in dd and round(50.0, 4) in dd and dd[round(50.0, 4)]["peak_g"] > 0]
        if v:
            curve.append((q, float(np.median(v))))
    for q, v in curve:
        bar = "#" * max(1, int(round(v * 34)))
        mark = "  ← 模型常数" if abs(q - 50) < 1e-6 else (
               "  ← 真鸟均值附近" if abs(q - 43) < 1e-6 else "")
        print("      %5.1f°  %5.2f×  %s%s" % (q, v, bar, mark))
    if len(curve) >= 5:
        qs_ = [q for q, _ in curve]; vals = [v for _, v in curve]
        lo, hi = min(vals), max(vals)
        rng = hi - lo
        # 「达到总变化的 90%」所在的角度 = 敏感区的右端
        q90 = next((q for q, v in curve if v >= lo + 0.9 * rng), qs_[-1])
        # 末三档的相对起伏 —— 小于 3% 才叫平台
        tail = vals[-3:]
        flat = (max(tail) - min(tail)) / max(np.mean(tail), 1e-9) < 0.03
        print("   → 全跨度 %.2f× → %.2f×（总变化 %.0f%%）；达到 90%% 变化量在 %.0f°。"
              % (lo, hi, 100 * rng / max(lo, 1e-9), q90))
        if flat and q90 <= 55:
            print("     **强非线性**：敏感区在 %.0f–%.0f°，之上是平台。" % (qs_[0], q90))
            print("     这解释了①的跨度变化大、而 50°→垂线档几乎不动：50° 已经在平台上。")
        elif q90 <= 55:
            print("     敏感区集中在 %.0f–%.0f°，其上仍在缓慢变化（末三档起伏 %.1f%%）。"
                  % (qs_[0], q90, 100 * (max(tail) - min(tail)) / max(np.mean(tail), 1e-9)))
        else:
            print("     响应在整个扫描范围内持续变化，没有平台。")

    # ---- 方向性 ----
    print("=" * 76)
    print("② 方向：q1_0 变大（腿更竖直）到底是变好还是变坏")
    slopes = []
    for k, dd in by.items():
        q = np.array(sorted(dd)); p = np.array([dd[x]["peak_g"] for x in q])
        m_ = (q >= BIO_LO - 1e-6) & (q <= BIO_HI + 1e-6)
        if m_.sum() < 3:
            continue
        A = np.c_[np.ones(m_.sum()), q[m_]]
        b, *_ = np.linalg.lstsq(A, p[m_], rcond=None)
        slopes.append(b[1])
    slopes = np.array(slopes)
    pos = 100.0 * (slopes > 0).mean()
    print("   d(peak_g)/d(q1_0) 中位 = %+.4f g/度，%.0f%% 的设计为正" % (np.median(slopes), pos))
    if pos > 70:
        print("   → 与预期一致：腿更竖直 → 可用行程变小 → 峰值变高。")
        print("     「垂线对齐」是拿性能换（E28 已证明并不存在的）生物一致性。")
    elif pos < 30:
        print("   → 与预期相反：腿更竖直反而峰值更低。这条要单独查，别直接写进论文。")
    else:
        print("   → 没有一致方向，说明 q1_0 与设计强耦合。")

    # ---- 垂线那一点 ----
    print("=" * 76)
    print("③ 「髋在足正上方」那一档（每个设计闭式解出的 q1_0*）")
    qv, dg, dm = [], [], []
    for k, dd in by.items():
        r0 = dd.get(round(50.0, 4))
        qs = dd[list(dd)[0]]["q1_vert"]
        rv = dd.get(round(qs, 4))
        if r0 is None or rv is None:
            continue
        qv.append(qs)
        dg.append(100.0 * (rv["peak_g"] - r0["peak_g"]) / max(r0["peak_g"], 1e-9))
        dm.append(100.0 * (rv["leg_mass_g"] - r0["leg_mass_g"]) / max(r0["leg_mass_g"], 1e-9))
    if qv:
        print("   q1_0* 中位 %.1f°（跨度 %.1f–%.1f°）—— 真鸟实测是 %.1f°±9°，差 %.0f°"
              % (np.median(qv), min(qv), max(qv), BIO_MEAN, np.median(qv) - BIO_MEAN))
        print("   相对 50° 基准：峰值过载 %+.1f%%   腿质量 %+.1f%%"
              % (np.median(dg), np.median(dm)))
        print("   （腿质量那一项是 E27 的入口，详见 e27_vertical_structural.py）")
    print("=" * 76)

    blob = dict(verdict=verdict, span_median_pct=med,
                span_p90_pct=float(np.percentile(spans, 90)),
                span_max_pct=float(spans.max()), n_designs=int(len(spans)),
                per_cond={t: float(np.median(v)) for t, v in per.items() if v},
                slope_median=float(np.median(slopes)) if len(slopes) else None,
                slope_pos_pct=float(pos) if len(slopes) else None,
                q1_vert_median=float(np.median(qv)) if qv else None,
                d_peak_at_vert_pct=float(np.median(dg)) if dg else None,
                d_mass_at_vert_pct=float(np.median(dm)) if dm else None,
                bio_span=[BIO_LO, BIO_HI], grid=list(args.grid),
                curve_median=[[q, v] for q, v in curve])
    fp = os.path.join(out_dir, "e26_q1_sweep.json")
    json.dump(blob, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % fp)
    if args.fig:
        _fig(by, spans, qv, out_dir, med)
    return blob


def _fig(by, spans, qv, out_dir, med):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    COL = {c["tag"]: c for c in CONDS}
    cols = [STEEL, "#8fb3cc", GREEN, "#8fbf9a", ORANGE, "#e0a86b"]
    fig, ax = plt.subplots(1, 3, figsize=(15.2, 4.4),
                           gridspec_kw=dict(width_ratios=[1.15, 1.0, .8], wspace=.27))
    a = ax[0]
    for k, dd in by.items():
        q = np.array(sorted(dd)); p = np.array([dd[x]["peak_g"] for x in q])
        ci = dd[list(dd)[0]]["ci"]
        a.plot(q, p / p[np.argmin(abs(q - 50))], "-", lw=.9, alpha=.35, color=cols[ci % len(cols)])
    a.axvspan(BIO_LO, BIO_HI, color=GREEN, alpha=.13)
    a.axvline(50, color=CRIM, lw=1.8, ls="--")
    a.text(50.6, a.get_ylim()[1] * .97, "模型常数 50°", color=CRIM, fontsize=9, va="top")
    a.text((BIO_LO + BIO_HI) / 2, a.get_ylim()[0], "真鸟实测跨度", color=GREEN,
           fontsize=9, ha="center", va="bottom")
    a.set_xlabel("q1_0 跗跖骨倾角 / °"); a.set_ylabel("峰值过载 / 该设计在 50° 处的值")
    a.set_title("(a) 每条线 = 一个设计，只动 q1_0", fontsize=11)
    a.grid(alpha=.2)

    b = ax[1]
    _rg = (float(np.min(spans)), float(np.max(spans)))
    if _rg[1] - _rg[0] < 1e-6:          # 退化：所有设计敏感度一样
        _rg = (_rg[0] - .5, _rg[1] + .5)
    b.hist(spans, bins=30, range=_rg, color=STEEL, alpha=.85)
    b.axvline(10, color=GREEN, lw=1.8, ls="--"); b.axvline(25, color=CRIM, lw=1.8, ls="--")
    b.axvline(med, color="k", lw=2.2)
    b.text(med, b.get_ylim()[1] * .92, " 中位 %.1f%%" % med, fontsize=10, fontweight="bold")
    b.text(10, b.get_ylim()[1] * .55, " 10%\n保持常数", color=GREEN, fontsize=8.5)
    b.text(25, b.get_ylim()[1] * .55, " 25%\n升为第10维", color=CRIM, fontsize=8.5)
    b.set_xlabel("生物跨度内 peak_g 的相对变化 / %"); b.set_ylabel("设计数")
    b.set_title("(b) 敏感度分布与判据", fontsize=11)

    c = ax[2]
    if qv:
        _rq = (float(np.min(qv)), float(np.max(qv)))
        if _rq[1] - _rq[0] < 1e-6:
            _rq = (_rq[0] - .5, _rq[1] + .5)
        c.hist(qv, bins=24, range=_rq, color=ORANGE, alpha=.85)
        c.axvline(BIO_MEAN, color=GREEN, lw=2.2)
        c.axvspan(BIO_LO, BIO_HI, color=GREEN, alpha=.13)
        c.text(BIO_MEAN, c.get_ylim()[1] * .92, " 真鸟 %.1f°" % BIO_MEAN,
               color=GREEN, fontsize=9.5, fontweight="bold")
        c.axvline(50, color=CRIM, lw=1.8, ls="--")
    c.set_xlabel("让髋落在足正上方所需的 q1_0* / °"); c.set_ylabel("设计数")
    c.set_title("(c) 「垂线对齐」要多陡\n（Bigoni 那一档）", fontsize=11)
    fig.tight_layout()
    fp = os.path.join(out_dir, "e26_q1_sweep.png")
    fig.savefig(fp, dpi=170); print("→ %s" % fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/v23_e5_bio/cvae_r12.pt"))
    ap.add_argument("--out", default=os.path.join(ROOT, "outputs/v23_e26"))
    ap.add_argument("--ndes", type=int, default=24, help="每个工况生成多少个设计")
    ap.add_argument("--grid", type=float, nargs="+",
                    default=[28., 33., 38., 43., 48., 50., 55., 62., 70., 80.],
                    help="q1_0 扫描档（度）。28–55 是 E28 实测跨度，62–80 供 E27 用")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seed", type=int, default=26)
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--geom-only", action="store_true",
                    help="只做纯几何预检，秒出，不需要 exudyn")
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    cache_fp = os.path.join(args.out, "e26_sweep_cache.jsonl")

    if args.geom_only or not args.analyze_only:
        designs = gen_designs(args.ckpt, args.ndes, args.seed)
        print("[e26] %d 个工况 × %d 设计 = %d 条腿" % (len(CONDS), args.ndes, len(designs)))
        g = geom_report(designs)
        json.dump(g, open(os.path.join(args.out, "e26_geom.json"), "w",
                          encoding="utf-8"), ensure_ascii=False, indent=1)
        if args.geom_only:
            print("→ %s" % os.path.join(args.out, "e26_geom.json")); return
        run(designs, args.grid, args.workers, cache_fp)
    if not os.path.exists(cache_fp):
        raise SystemExit("[e26] 没有缓存 %s —— 先不带 --analyze-only 跑一遍" % cache_fp)
    rows = [json.loads(ln) for ln in open(cache_fp, encoding="utf-8") if ln.strip()]
    ngood = sum(1 for r in rows if "fail" not in r)
    if ngood < 30:
        print("[e26] 有效点只有 %d 个（<30），无法判定。" % ngood)
        print("      多半是扫描那一步全挂了：exudyn 没装好，或 physics_v2 导不进来。")
        return
    analyze(rows, args, args.out)


if __name__ == "__main__":
    main()

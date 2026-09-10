# -*- coding: utf-8 -*-
"""E22 · 管径-段长标度律 vs 生物实测（A 组，本地重评，不占 A100）。

**为什么这条可能出论文**：`physics_v2.size_segment` 纯粹由弯矩、轴力、屈曲和几何上限
反推外径 D，**全程没有用过任何生物的 D 信息**。所以把它长出来的
    log10 D = α · log10 L + c
和鸟骨的实测指数比，是**真正非循环**的一次比较 —— 与 L1 那条幂律不同，
后者是硬编码在 `bioprior.expand` 里的。

**为什么不是零成本**：工厂只存了 KEYS_V2 这 16 个量，**D 不在里面**
（`eval_v2` 其实返回了 `D_mm`，只是 factory 没落盘）。所以要对抽样的设计重跑一遍
`eval_v2`。好消息是本地多进程就能跑完，不用上 A100；坏消息是它不是"零机时"。
→ 顺手把 `D_mm` 加进 v2.5 的 KEYS_V2，下次这条就真免费了。

**必须同时报告的风险量**：D 被 `D_MIN = 4 mm` 和 `D_MAX_RATIO = 0.25` 上下夹逼。
被夹住的样本里 D 不再由力学决定，标度自然是假的。**夹住比例 > 20% 这条结论就不能要。**

生物侧的指数**不写死在代码里** —— 用 --bio 传，或写在 e22_bio_ref.json 里。
（截至 2026-09-10，会议里那组 −0.38 / −0.57 / −0.12 / +0.02 尚未核实，不予硬编码。）

用法：
    # 第一步：抽样重评，落盘缓存（可中断续跑）
    python src/stage10_v2/e22_dl_scaling.py --n 3000 --workers 8
    # 第二步：只做分析（读缓存，秒出）
    python src/stage10_v2/e22_dl_scaling.py --analyze-only --fig
    # 带上生物侧参照
    python src/stage10_v2/e22_dl_scaling.py --analyze-only --bio e22_bio_ref.json --fig
"""
import argparse, json, os, sys, time
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from agroup_io import (load_blocks, to_Y, feasible_mask, selfcheck,      # noqa: E402
                       iter_requirements, default_factory, setup_font,
                       CRIM, STEEL, GREEN, ORANGE)

SEGS = ["tarso (L1)", "tibio (L2)", "femur (L3)"]

# 理论锚点：细长比口径 d/L ∝ L^α 下，几种定尺准则各自给出的指数。
# 会上 Hsia 在 47:09 就是按这个读 −1/2 的：「bending stiffness / uniform compression / and buckling」。
ANCHORS = [
    (0.0,        "等比 isometry"),
    (-1.0 / 4,   "定弯曲刚度 EI/L³ = 常数"),
    (-1.0 / 2,   "定载荷 Euler 屈曲 F ∝ EI/L²"),
    (-2.0 / 3,   "定载荷抗弯强度 σ = Mc/I"),
]


# ---------------------------------------------------------------- 抽样
def pick_designs(blocks, n, nreq, seed):
    """在质量对数轴上分层抽样可行设计，避免全被中间质量占满。"""
    cand = []
    for bi, blk in enumerate(blocks):
        Y = to_Y(blk)
        seen = np.zeros(len(Y), bool)
        for gcap, smax in iter_requirements(blk, nreq):
            seen |= feasible_mask(Y, gcap, smax)
        for j in np.where(seen)[0]:
            cand.append((bi, int(j)))
    if not cand:
        return []
    rng = np.random.default_rng(seed)
    lm = np.array([np.log10(blocks[bi]["m"]) for bi, _ in cand])
    edges = np.linspace(lm.min(), lm.max() + 1e-9, 13)
    bins = np.clip(np.digitize(lm, edges) - 1, 0, len(edges) - 2)
    per = max(1, n // (len(edges) - 1))
    out = []
    for b in range(len(edges) - 1):
        idx = np.where(bins == b)[0]
        if not len(idx):
            continue
        take = idx if len(idx) <= per else rng.choice(idx, per, replace=False)
        out += [cand[i] for i in take]
    rng.shuffle(out)
    return out[:n]


# ---------------------------------------------------------------- 重评
def _eval_one(a):
    """在子进程里重跑一次 eval_v2，只为把 D_mm / seg_len 取回来。

    连 import 一起包在 try 里：环境不对时要安静地返回 None，
    让主进程统计失败数并早停，而不是抛一个 traceback 把整个 map 打断
    —— 这个脚本是给无人值守的批处理用的。
    """
    try:
        import physics_v2 as P
        x9, m, v0, kc, zc = a
        base = {**P.SCEN_BIRD_X, "hip_damp_unified": True, "foot_mode": "bearing"}
        r = P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zc, npass=2, base=base)
    except Exception:
        return None
    if r is None or r.get("fail"):
        return None
    sl = r.get("seg_len")
    if sl is None:
        return None
    return dict(m=m, v0=v0, kc=kc,
                L_mm=[float(v) * 1e3 for v in sl],
                D_mm=[float(v) for v in r["D_mm"]],
                D_max_mm=[float(v) for v in r["D_max_mm"]],
                governs=list(r["governs"]),
                peak_a=float(r["peak_a"]), leg_mass_kg=float(r["leg_mass_kg"]))


def re_evaluate(blocks, picks, workers, cache_fp):
    from concurrent.futures import ProcessPoolExecutor
    done = set()
    if os.path.exists(cache_fp):
        for ln in open(cache_fp, encoding="utf-8"):
            try:
                done.add(json.loads(ln)["key"])
            except Exception:
                pass
        print("[e22] 缓存里已有 %d 条，续跑" % len(done))
    jobs, keys = [], []
    for bi, j in picks:
        k = "%d:%d" % (blocks[bi]["cid"], j)
        if k in done:
            continue
        b = blocks[bi]
        jobs.append((b["X"][j], b["m"], b["v0"], b["kc"], b["zeta_c"])); keys.append(k)
    if not jobs:
        print("[e22] 缓存已完整，跳过重评"); return
    print("[e22] 需要重评 %d 个设计（多进程 %d）" % (len(jobs), workers))
    t0 = time.time()
    nbad = 0
    with ProcessPoolExecutor(max_workers=workers) as ex, open(cache_fp, "a") as f:
        for i, (k, r) in enumerate(zip(keys, ex.map(_eval_one, jobs, chunksize=4)), 1):
            if r is None:
                nbad += 1
            else:
                r["key"] = k
                f.write(json.dumps(r) + "\n")
            if i % 200 == 0 or i == len(jobs):
                el = time.time() - t0
                f.flush()
                print("      %d/%d  失败 %d  (%.0fs, ~%.3f s/个, 预计剩 %.0fs)"
                      % (i, len(jobs), nbad, el, el / i, el / i * (len(jobs) - i)), flush=True)
            if i == 200 and nbad == i:
                print("\n[e22] 前 200 个全部失败 —— 多半是 exudyn 没装好或 physics_v2 导不进来。")
                print("      先单独试一次：python -c \"import sys;sys.path.insert(0,'src/stage10_v2');"
                      "import physics_v2 as P;print(P.eval_v2.__name__)\"")
                print("      提前收工，不浪费机时。\n")
                return


# ---------------------------------------------------------------- 分析
def ols(x, y):
    A = np.c_[np.ones(len(x)), x]
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    yh = A @ b
    ss = 1.0 - np.sum((y - yh) ** 2) / max(np.sum((y - y.mean()) ** 2), 1e-30)
    n = len(x)
    se = np.sqrt(np.sum((y - yh) ** 2) / max(n - 2, 1) /
                 max(np.sum((x - x.mean()) ** 2), 1e-30))
    return float(b[1]), float(b[0]), float(ss), float(1.96 * se), n


def analyze(rows, args, out_dir):
    L = np.array([r["L_mm"] for r in rows])          # (n, 3)
    D = np.array([r["D_mm"] for r in rows])
    DM = np.array([r["D_max_mm"] for r in rows])
    M = np.array([r["m"] for r in rows])
    GOV = np.array([r["governs"] for r in rows])

    # --- 夹逼诊断：这一步不过关，后面的指数一律不能信 ---
    at_min = D <= 4.0 + 1e-6                       # physics_v2.D_MIN = 0.004 m
    at_max = D >= DM - 1e-6                        # D_MAX_RATIO · 段长
    clamp = at_min | at_max
    print("=" * 74)
    print("① 夹逼诊断（D 被 D_MIN=4mm / D_MAX_RATIO=0.25 顶住的比例）")
    for s in range(3):
        print("   %-12s 触下界 %5.1f%%   触上界 %5.1f%%   合计 %5.1f%%"
              % (SEGS[s], 100 * at_min[:, s].mean(), 100 * at_max[:, s].mean(),
                 100 * clamp[:, s].mean()))
    tot = 100 * clamp.mean()
    print("   全部段合计 %.1f%%" % tot)
    verdict_ok = tot <= args.clamp_cap * 100
    if not verdict_ok:
        print("   → 超过 %.0f%% —— **这条结论不能要**。D 主要由几何上下界决定，不是力学。"
              % (args.clamp_cap * 100))
        print("     要么放宽 D_MIN / D_MAX_RATIO 重跑，要么把这条实验降级为附录里的负结果。")
    else:
        print("   → 在 %.0f%% 以内，指数可用。" % (args.clamp_cap * 100))
    print("   受哪条判据支配：" + "  ".join(
        "%s=%.0f%%" % (g, 100 * (GOV == g).mean())
        for g in sorted(set(GOV.ravel().tolist()))))

    # --- 三种口径的指数，生物侧用哪种都能对上 ---
    print("=" * 74)
    print("② 模型长出来的标度指数（只用未被夹住的样本）")
    res = {}
    for tag, xs, ys, lab in (
            ("D_vs_L", np.log10(L), np.log10(D), "log D ~ α·log L      （逐段）"),
            ("D_vs_m", np.log10(np.repeat(M[:, None], 3, 1)), np.log10(D),
             "log D ~ α·log m      （逐段，对体重）"),
            ("DL_vs_L", np.log10(L), np.log10(D / L), "log (D/L) ~ α·log L  （细长比）")):
        print("   " + lab)
        res[tag] = {}
        for s in range(3):
            k = ~clamp[:, s]
            if k.sum() < 30:
                print("      %-12s 有效样本不足（%d）" % (SEGS[s], k.sum())); continue
            a, b, r2, ci, n = ols(xs[k, s], ys[k, s])
            res[tag][SEGS[s]] = dict(alpha=a, intercept=b, r2=r2, ci95=ci, n=n)
            print("      %-12s α = %+.3f ± %.3f   R² = %.3f   n = %d"
                  % (SEGS[s], a, ci, r2, n))
        k = ~clamp.ravel()
        a, b, r2, ci, n = ols(xs.ravel()[k], ys.ravel()[k])
        res[tag]["pooled"] = dict(alpha=a, intercept=b, r2=r2, ci95=ci, n=n)
        print("      %-12s α = %+.3f ± %.3f   R² = %.3f   n = %d" % ("合并三段", a, ci, r2, n))

    # --- 跨质量口径：与生物图问的是同一个问题 ---
    # 生物那张图是**跨物种**回归 d/L ~ L，物种之间 L 的差异来自体重差异。
    # 我们的数据里 L 有两个来源：体重，以及 u_L 给的 ±2.5σ 设计自由度。
    # 直接对 L 回归会把两者混在一起。这里按体重分箱取中位数再回归，
    # 只保留**跨质量**那一份变化 —— 这才是能和生物图逐位比的数。
    print("=" * 74)
    print("②c 跨质量口径（分箱取中位后回归，与生物图同问）")
    lm = np.log10(M)
    edges = np.linspace(lm.min(), lm.max() + 1e-9, args.mbins + 1)
    bidx = np.clip(np.digitize(lm, edges) - 1, 0, args.mbins - 1)
    res["DL_vs_L_bymass"] = {}
    res["L_vs_m"] = {}
    for sname, sel in [(SEGS[s], s) for s in range(3)]:
        k = ~clamp[:, sel]
        xs, ys, ms = [], [], []
        for b in range(args.mbins):
            q = k & (bidx == b)
            if q.sum() < 20:
                continue
            xs.append(np.median(np.log10(L[q, sel])))
            ys.append(np.median(np.log10(D[q, sel] / L[q, sel])))
            ms.append(np.median(lm[q]))
        if len(xs) < 5:
            print("   %-12s 有效质量箱不足" % sname); continue
        a, b0, r2, ci, n = ols(np.array(xs), np.array(ys))
        aL, _, r2L, ciL, _ = ols(np.array(ms), np.array(xs))
        res["DL_vs_L_bymass"][sname] = dict(alpha=a, intercept=b0, r2=r2, ci95=ci, n=n)
        res["L_vs_m"][sname] = dict(alpha=aL, r2=r2L, ci95=ciL)
        print("   %-12s d/L ~ L^α  α = %+.3f ± %.3f (R² %.3f, %d 个质量箱)   "
              "| L ~ m^%.3f" % (sname, a, ci, r2, n, aL))
    print("   注意：上面②里对全部样本直接回归 L 的那组，混进了同质量下换腿长的设计自由度，")
    print("        与生物图不是同一个问题；**比生物用这一组**。")

    # --- 与生物侧比 ---
    print("=" * 74)
    print("②b 理论锚点（同为 d/L ∝ L^α 口径）")
    for a_, lab in ANCHORS:
        mv = res["DL_vs_L"].get("pooled")
        z = (abs(mv["alpha"] - a_) / max(mv["ci95"] / 1.96, 1e-9)) if mv else float("nan")
        print("   α = %+.3f   %-28s   合并三段距此 %.1fσ" % (a_, lab, z))
    print("   （−1/2 就是会上 47:09 Hsia 说的那个 negative half：定载荷下的 Euler 屈曲）")

    print("=" * 74)
    bio = {}
    if args.bio and os.path.exists(args.bio):
        bio = json.load(open(args.bio, encoding="utf-8"))
    if not bio.get("values"):
        bio = {}
    if not bio:
        print("③ 未提供生物侧参照 —— 跳过比较。")
        print("   把实测指数写成 JSON 传给 --bio，格式：")
        print('   {"source": "Shizhuo 会议 45:45 / 出处", "convention": "DL_vs_L",')
        print('    "values": {"tarso (L1)": -0.38, "tibio (L2)": -0.57}}')
        print("   ⚠ 会议里那组 −0.38 / −0.57 / −0.12 / +0.02 **尚未核对原始帧**，")
        print("     所以本脚本不把它写死。核实后再传进来。")
        print("   提示：那几个数是**负的**，而 log D ~ log L 在任何合理力学下都必然为正，")
        print("        所以它们多半是**细长比口径**（D/L 随 L 下降 = 大骨相对更细），")
        print("        对应上面第三组 DL_vs_L。核帧时顺便确认一下横轴到底是 L 还是体重。")
    else:
        conv = bio.get("convention", "D_vs_L")
        key = "DL_vs_L_bymass" if (conv == "DL_vs_L" and res.get("DL_vs_L_bymass")) else conv
        print("③ 与生物实测比较（口径 %s → 用%s，来源：%s）"
              % (conv, "跨质量分箱" if key.endswith("bymass") else "全样本", bio.get("source", "未注明")))
        R = res.get(key, {})
        for nm, bv in bio.get("values", {}).items():
            mv = R.get(nm)
            if mv is None:
                print("   %-12s 生物 %+.3f   模型：无对应" % (nm, bv)); continue
            z = abs(mv["alpha"] - bv) / max(mv["ci95"] / 1.96, 1e-9)
            print("   %-12s 生物 %+.3f   模型 %+.3f ± %.3f   相差 %.1fσ  %s"
                  % (nm, bv, mv["alpha"], mv["ci95"], z,
                     "一致" if z < 2 else ("接近" if z < 4 else "不一致")))
        for alt in bio.get("alternatives", []):
            line = []
            for nm, bv in alt.get("values", {}).items():
                mv = R.get(nm)
                if mv is None:
                    continue
                z = abs(mv["alpha"] - bv) / max(mv["ci95"] / 1.96, 1e-9)
                line.append("%s 生物 %+.2f vs 模型 %+.3f (%.1fσ)" % (nm.split()[0], bv, mv["alpha"], z))
            if line:
                print("   [备选] %-52s %s" % (alt.get("source", "")[:52], "   ".join(line)))
        print("   解读提示：一致 → 纯力学反推自发长成了生物的形状，这是非循环的证据；")
        print("             不一致 → 也是结果，说明生物在别的目标下选形（抗扭/骨髓/生长），")
        print("             正好接 Bigoni 33:04 那条「鸟骨是桁架不是实心管」。")

    print("=" * 74)
    blob = dict(n_rows=len(rows), clamp_pct=float(tot),
                clamp_ok=bool(verdict_ok), clamp_cap=args.clamp_cap,
                at_min_pct=[float(100 * at_min[:, s].mean()) for s in range(3)],
                at_max_pct=[float(100 * at_max[:, s].mean()) for s in range(3)],
                fits=res, bio=bio)
    fp = os.path.join(out_dir, "e22_dl_scaling.json")
    json.dump(blob, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % fp)
    if args.fig:
        _fig(L, D, clamp, res, bio, out_dir, tot)
    return blob


def _fig(L, D, clamp, res, bio, out_dir, tot):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    COLS = [STEEL, GREEN, ORANGE]
    fig, ax = plt.subplots(1, 2, figsize=(12.6, 4.6),
                           gridspec_kw=dict(width_ratios=[1.15, .85], wspace=.26))
    a = ax[0]
    for s in range(3):
        k = ~clamp[:, s]
        a.scatter(L[k, s], D[k, s], s=7, color=COLS[s], alpha=.35, lw=0, label=SEGS[s])
        a.scatter(L[~k, s], D[~k, s], s=7, color="#c9c9c9", alpha=.5, lw=0,
                  label="被夹住" if s == 0 else None)
        f = res.get("D_vs_L", {}).get(SEGS[s])
        if f and k.sum() > 30:
            xs = np.linspace(np.log10(L[k, s].min()), np.log10(L[k, s].max()), 20)
            a.plot(10 ** xs, 10 ** (f["intercept"] + f["alpha"] * xs),
                   color=COLS[s], lw=2.2)
    a.set_xscale("log"); a.set_yscale("log")
    a.set_xlabel("段长 L / mm"); a.set_ylabel("外径 D / mm")
    a.set_title("(a) 纯力学反推出来的 D–L 标度\n被上下界夹住的样本 %.1f%%（灰点，不参与拟合）" % tot,
                fontsize=11)
    a.legend(fontsize=8.5, frameon=False, loc="upper left"); a.grid(alpha=.2, which="both")

    b = ax[1]
    R = res.get(bio.get("convention", "D_vs_L"), {})
    names = [n for n in SEGS if n in R] + (["pooled"] if "pooled" in R else [])
    y = np.arange(len(names))
    b.errorbar([R[n]["alpha"] for n in names], y,
               xerr=[R[n]["ci95"] for n in names], fmt="o", color=STEEL,
               ms=7, capsize=4, lw=2, label="本模型（力学反推）")
    if bio.get("values"):
        for i, n in enumerate(names):
            if n in bio["values"]:
                b.scatter([bio["values"][n]], [i], marker="D", s=70, color=CRIM,
                          zorder=5, label="生物实测" if i == 0 else None)
    b.set_yticks(y); b.set_yticklabels(names, fontsize=10); b.invert_yaxis()
    if bio.get("convention", "D_vs_L") == "DL_vs_L":
        for a_, lab in ANCHORS:
            b.axvline(a_, color="#999", lw=1.0, ls=":" if a_ else "-")
            b.text(a_, -0.72, lab.split()[0], rotation=90, fontsize=7.2,
                   color="#777", ha="center", va="bottom")
    b.axvline(0, color="k", lw=.8, ls=":")
    b.set_xlabel("标度指数 α"); b.grid(alpha=.22, axis="x")
    b.set_title("(b) 模型 vs 生物\n%s" % ("（生物侧未提供）" if not bio.get("values") else
                                      bio.get("source", "")), fontsize=11)
    b.legend(fontsize=9, frameon=False)
    fig.tight_layout()
    fp = os.path.join(out_dir, "e22_dl_scaling.png")
    fig.savefig(fp, dpi=170); print("→ %s" % fp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--factory", default=default_factory(ROOT))
    ap.add_argument("--out", default=None)
    ap.add_argument("--n", type=int, default=3000, help="抽样重评多少个可行设计")
    ap.add_argument("--nreq", type=int, default=4)
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 1))
    ap.add_argument("--seed", type=int, default=22)
    ap.add_argument("--clamp-cap", type=float, default=0.20, help="夹逼比例上限")
    ap.add_argument("--mbins", type=int, default=12, help="跨质量口径的分箱数")
    ap.add_argument("--bio", default=os.path.join(HERE, "e22_bio_ref.json"))
    ap.add_argument("--analyze-only", action="store_true")
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()
    out_dir = args.out or os.path.dirname(args.factory)
    cache_fp = os.path.join(out_dir, "e22_resize_cache.jsonl")

    if not args.analyze_only:
        selfcheck()
        blocks = load_blocks(args.factory)
        if not blocks:
            raise SystemExit("[e22] 空工厂：%s" % args.factory)
        picks = pick_designs(blocks, args.n, args.nreq, args.seed)
        print("[e22] 分层抽到 %d 个可行设计" % len(picks))
        re_evaluate(blocks, picks, args.workers, cache_fp)

    if not os.path.exists(cache_fp):
        raise SystemExit("[e22] 没有缓存 %s —— 先不带 --analyze-only 跑一遍" % cache_fp)
    rows = [json.loads(ln) for ln in open(cache_fp, encoding="utf-8") if ln.strip()]
    print("[e22] 分析 %d 条重评结果" % len(rows))
    if len(rows) < 30:
        print("[e22] 有效结果太少（<30），无法做标度回归。")
        print("      多半是重评那一步全挂了：exudyn 没装好，或 physics_v2 导不进来。")
        print("      先单独试：python -c \"import sys;sys.path.insert(0,'src/stage10_v2');"
              "import physics_v2\"")
        return
    analyze(rows, args, out_dir)


if __name__ == "__main__":
    main()

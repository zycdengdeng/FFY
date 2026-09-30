# -*- coding: utf-8 -*-
"""
生物先验 b=0.365  vs  物理涌现 b_eff=0.238  的并排落震动画

两条标度律在 5 kg 处对齐（同一条腿），到 30 kg 就分开：
    bio  : log10 L1 = 0.631 + 0.365·log10 m(g)          （PGLS 校正后的水鸟先验）
    phys : 同点出发，斜率换成 0.238                       （数据工厂涌现的 b_eff）
→ 30 kg 时 bio 184 mm vs phys 147 mm。同一套关节刚度/阻尼，只变几何。

用法（A100）:
    python src/stage10_v2/anim_b_compare.py --m 30 --out outputs/anim_b
产出: hist_b.npz + b_compare_30kg.mp4 + b_compare_30kg_首帧.png
"""
import os, sys, json, argparse
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physics_v2 as P
from factory_v2 import zeta_of_kc

A_BIO, B_BIO, M_ANCHOR = 0.6307, 0.365, 5.0
# 机器侧指数由 --b-machine 给；r12 实测：硬地 0.263、草地 0.343、湿沙 0.354
GCAP, SMAX = 10 * 9.81, 0.024

def L1_of(m_kg, b, anchor=M_ANCHOR):
    """两条律在 anchor 处重合；b 为标度指数。返回 mm。"""
    lg_a = np.log10(anchor * 1000.0)
    logL_a = A_BIO + B_BIO * lg_a                 # 锚点腿长（两条律共用）
    return 10 ** (logL_a + b * (np.log10(m_kg * 1000.0) - lg_a))

def search_kappa(L1, m, v0, kc, r2, r3, tha, thk, n=64, seed=0):
    """给这条腿找一组可行的 κ踝/κ膝/κ髋/τ（随机搜索，取峰值最低的可行解）。"""
    rng = np.random.default_rng(seed); best = None
    for _ in range(n):
        kap = [rng.uniform(0.75, 8.0), rng.uniform(1.5, 8.0), rng.uniform(3.0, 32.0)]
        tau = rng.uniform(0.005, 0.1)
        x = [L1, r2, r3] + kap + [tau, tha, thk]
        r = P.eval_v2(tuple(x), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2)
        if r.get("fail"): continue
        ok, _ = P.feasible_v2(r, GCAP, SMAX)
        if ok and (best is None or r["peak_a"] < best[1]):
            best = (kap + [tau], r["peak_a"])
    return None if best is None else best[0]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m", type=float, default=30.0, help="机身质量 kg")
    ap.add_argument("--v0", type=float, default=1.2)
    ap.add_argument("--kc", type=float, default=1.0e5, help="地面刚度（1e5=草地）")
    ap.add_argument("--r2", type=float, default=1.75); ap.add_argument("--r3", type=float, default=1.05)
    ap.add_argument("--tha", type=float, default=126.0); ap.add_argument("--thk", type=float, default=137.0)
    ap.add_argument("--slow", type=float, default=150.0, help="慢放倍数")
    ap.add_argument("--b-machine", type=float, default=0.263,
                    help="机器侧标度指数（硬地 0.263 / 草地 0.343 / 湿沙 0.354）")
    ap.add_argument("--tag", default="hard", help="产出文件名后缀")
    ap.add_argument("--out", default="outputs/anim_b")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    B_EFF = a.b_machine
    Lb, Lp = L1_of(a.m, B_BIO), L1_of(a.m, B_EFF)
    print(f"[几何] m={a.m:g} kg   生物先验 b=0.365 → L1 {Lb:.1f} mm"
          f"   |   物理涌现 b=0.238 → L1 {Lp:.1f} mm   （差 {100*(Lb/Lp-1):+.0f}%）")

    print("[搜索] 为生物先验腿找一组可行关节参数（两条腿共用，只变几何）…")
    kap = search_kappa(Lb, a.m, a.v0, a.kc, a.r2, a.r3, a.tha, a.thk)
    if kap is None:
        print("  ✗ 生物腿在该工况下搜不到可行解；改用物理腿来定 κ")
        kap = search_kappa(Lp, a.m, a.v0, a.kc, a.r2, a.r3, a.tha, a.thk)
    assert kap is not None, "两条腿都搜不到可行解，请换工况"
    print(f"  κ踝={kap[0]:.2f} κ膝={kap[1]:.2f} κ髋={kap[2]:.2f} τ={kap[3]*1e3:.1f} ms")

    CASES = [("bio",  f"生物先验  b = {B_BIO:.3f}", Lb, "#2E7D5B"),
             ("phys", f"机器选择  b = {B_EFF:.3f}", Lp, "#1b6ca8")]
    out, meta = {}, {}
    for k, lab, L1, col in CASES:
        x = [L1, a.r2, a.r3] + kap + [a.tha, a.thk]
        r = P.eval_v2(tuple(x), a.m, a.v0, kc=a.kc, zeta_c=zeta_of_kc(a.kc),
                      npass=2, keep_history=True)
        assert not r.get("fail"), (k, r)
        ok, why = P.feasible_v2(r, GCAP, SMAX)
        h = r.pop("hist"); out[k] = h
        meta[k] = dict(label=lab, color=col, L1_mm=L1,
                       peak_g=r["peak_a"]/9.81, leg_stroke_mm=r["leg_stroke_mm"],
                       leg_mass_g=r["leg_mass_kg"]*1e3, ok=bool(ok), why=list(why))
        print(f"  {lab:<26} L1 {L1:6.1f} mm  峰值 {meta[k]['peak_g']:5.2f} g"
              f"  行程 {meta[k]['leg_stroke_mm']:5.1f} mm  腿重 {meta[k]['leg_mass_g']:5.0f} g"
              f"  {'✓ 可行' if ok else '✗ ' + ','.join(why)}")
    npz = os.path.join(a.out, f"hist_{a.tag}.npz")
    np.savez_compressed(npz, meta=json.dumps(meta),
        **{f"{k}__{n}": np.asarray(v) for k, h in out.items() for n, v in h.items()})
    json.dump(dict(m_kg=a.m, v0=a.v0, kc=a.kc, kappa=kap, b_bio=B_BIO, b_machine=B_EFF, meta=meta),
              open(os.path.join(a.out, f"b_compare_{a.tag}.json"), "w"), ensure_ascii=False, indent=1)
    print(f"[存] {npz}")
    render(npz, meta, a)

def render(npz, meta, a):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FFMpegWriter
    for f in ("Noto Sans CJK SC","Noto Sans CJK JP","WenQuanYi Zen Hei"):
        plt.rcParams["font.sans-serif"] = [f] + plt.rcParams["font.sans-serif"]
    plt.rcParams["axes.unicode_minus"] = False
    D = np.load(npz, allow_pickle=True); K = ["bio", "phys"]
    FPS, PRE, POST = 30, 0.004, 0.076
    S = {}
    for k in K:
        t = D[k+"__t"]; rf = float(D[k+"__r_foot"]); foot = D[k+"__foot"]; z = D[k+"__z"]
        i0 = int(np.argmax(foot[:, 2] <= rf)); sink = np.maximum(0., rf-foot[:, 2])
        S[k] = dict(tr=t-t[i0], g=D[k+"__az"]/9.81, stroke=1e3*((z[0]-z)-sink), rf=rf,
                    J=[foot, D[k+"__ankle_p"], D[k+"__knee_p"], D[k+"__hip_p"]])
    TR = np.linspace(-PRE, POST, int(round(FPS*a.slow*(PRE+POST))))
    sm = lambda k, n: np.interp(TR, S[k]["tr"], S[k][n])
    G = {k: sm(k, "g") for k in K}; ST = {k: sm(k, "stroke") for k in K}
    JT = {k: np.stack([[np.interp(TR, S[k]["tr"], Q[:, j]) for j in (0, 2)]
                       for Q in S[k]["J"]], 0).transpose(0, 2, 1) for k in K}
    fig = plt.figure(figsize=(13.2, 7.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1], height_ratios=[1.55, 1], hspace=.30, wspace=.22)
    axL = [fig.add_subplot(gs[0, i]) for i in range(2)]
    axG = fig.add_subplot(gs[1, 0]); axS = fig.add_subplot(gs[1, 1])
    lines = []
    for i, k in enumerate(K):
        A = axL[i]; m = meta[k]
        allx = JT[k][:, :, 0]; ally = JT[k][:, :, 1]
        A.set_xlim(allx.min()-.05, allx.max()+.05); A.set_ylim(-.03, ally.max()*1.12)
        A.set_aspect("equal"); A.axis("off")
        A.axhline(0, color="#8a7f6d", lw=3)
        A.set_title(f"{m['label']}\nL1 = {m['L1_mm']:.0f} mm", fontsize=15,
                    fontweight="bold", color=m["color"], pad=10)
        ln, = A.plot([], [], "-o", color=m["color"], lw=5, ms=9, mec="k", mew=.8)
        txt = A.text(.5, .04, "", transform=A.transAxes, ha="center", fontsize=13, fontweight="bold")
        lines.append((ln, txt))
    for A, dat, lab, lim, ll in ((axG, G, "峰值过载 / g", 10.0, "10 g 上限"),
                                 (axS, ST, "落震行程 / mm", 24.0, "24 mm 预算")):
        A.set_xlim(TR[0]*1e3, TR[-1]*1e3); A.set_xlabel("触地后时间 / ms", fontsize=12)
        A.set_ylabel(lab, fontsize=12.5)
        A.axhline(lim, color="#c0392b", lw=2, ls="--"); A.grid(alpha=.25)
        A.text(TR[-1]*1e3, lim, " "+ll, color="#c0392b", fontsize=11, va="bottom", ha="right")
        top = max(max(abs(dat[k])) for k in K)
        A.set_ylim(0, max(lim*1.25, top*1.15))
        for sp in ("top", "right"): A.spines[sp].set_visible(False)
    cur = {}
    for k in K:
        cur[k] = (axG.plot([], [], color=meta[k]["color"], lw=3)[0],
                  axS.plot([], [], color=meta[k]["color"], lw=3)[0])
    def upd(i):
        for j, k in enumerate(K):
            lines[j][0].set_data(JT[k][:, i, 0], JT[k][:, i, 1])
            lines[j][1].set_text(f"{G[k][i]:.1f} g   {ST[k][i]:.0f} mm")
            lines[j][1].set_color(meta[k]["color"])
            cur[k][0].set_data(TR[:i+1]*1e3, G[k][:i+1])
            cur[k][1].set_data(TR[:i+1]*1e3, ST[k][:i+1])
        return []
    sub = " · ".join(f"{meta[k]['label']}：峰值 {meta[k]['peak_g']:.1f} g，"
                     f"行程 {meta[k]['leg_stroke_mm']:.0f} mm，"
                     f"{'可行' if meta[k]['ok'] else '不可行'}" for k in K)
    fig.suptitle(f"同一套关节刚度，只变腿长标度律 —— {a.m:g} kg 落震（{a.slow:g}× 慢放）\n{sub}",
                 fontsize=15, fontweight="bold", y=.99)
    upd(len(TR)-1)
    png = os.path.join(a.out, f"b_compare_{a.tag}_{a.m:g}kg_末帧.png")
    fig.savefig(png, dpi=140, bbox_inches="tight"); print(f"[存] {png}")
    mp4 = os.path.join(a.out, f"b_compare_{a.tag}_{a.m:g}kg.mp4")
    try:
        w = FFMpegWriter(fps=FPS, bitrate=5200)
        with w.saving(fig, mp4, dpi=110):
            for i in range(len(TR)):
                upd(i); w.grab_frame()
                if i % 90 == 0: print(f"  {i}/{len(TR)}", flush=True)
        print(f"[存] {mp4}")
    except Exception as e:
        print(f"[跳过 mp4] {e}\n  → 末帧 PNG 已存，PPT 可先用它")

if __name__ == "__main__":
    main()

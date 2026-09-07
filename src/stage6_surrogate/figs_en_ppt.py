# -*- coding: utf-8 -*-
"""英文版 PPT 配图（6 张）。所有数字均从原始 CSV 现算，不硬编码。"""
import os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{U}/outputs/ppt_en"; os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "savefig.facecolor": "white", "figure.facecolor": "white"})
BLU, ORA, RED, GRN, GRY = "#1b6ca8", "#d98032", "#c0392b", "#2E7D5B", "#6b6b6b"
box = dict(boxstyle="round,pad=0.45", fc="white", ec=BLU, lw=1.6, alpha=.95)

D = pd.read_csv(f"{U}/data/birdtree/pgls_data_full.csv")
D = D[D.HWI.notna() & D.u.notna()].copy()
D["logL"] = D.logL.astype(float)
n = len(D)

def ols(y, X):
    cols = [np.asarray(c, float) for c in X]
    X = np.column_stack([np.ones(len(cols[0]))] + cols)
    b = np.linalg.lstsq(X, y, rcond=None)[0]; r = y - X @ b
    return b, 1 - r.var() / y.var(), r

# ---------- F1 allometry ----------
def F1():
    x, y = D.log_m.values, D.logL.values
    b, r2, res = ols(y, [x])
    P5 = pd.read_csv(f"{U}/outputs/phylo/S5_prior.csv")
    a_p, b_p = float(np.median(P5.a)), float(np.median(P5.b)); sd = float(np.median(P5.sd_res))
    nsp = int(P5.n.iloc[0])
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.4), gridspec_kw=dict(width_ratios=[1.35, 1], wspace=.24))
    A = ax[0]
    A.hexbin(x, y, gridsize=58, cmap="Blues", mincnt=1, bins="log", linewidths=0)
    xs = np.linspace(x.min(), x.max(), 100)
    wm = D.loc[D.is_water == 1, "log_m"]
    xw = np.linspace(float(wm.min()), float(wm.max()), 60)
    A.fill_between(xw, a_p + b_p*xw - sd, a_p + b_p*xw + sd, color=GRN, alpha=.20, zorder=3,
                   label=f"waterbirds, ±1 SD band  (SD = {sd:.3f} dex)")
    A.plot(xw, a_p + b_p*xw, color=GRN, lw=3.4, zorder=4,
           label=f"waterbirds   $b$ = {b_p:.3f}   (n = {nsp})")
    A.plot(xs, b[0] + b[1]*xs, color=ORA, lw=2.2, ls="--", zorder=4,
           label=f"all birds   $b$ = {b[1]:.3f}")
    A.set_xlabel("log$_{10}$  body mass  [g]", fontsize=12.5)
    A.set_ylabel("log$_{10}$  tarsometatarsus $L_1$  [mm]", fontsize=12.5)
    A.set_title(f"Law 1 — leg length scales with body mass\n{n:,} bird species",
                fontsize=14, fontweight="bold")
    A.legend(loc="upper left", fontsize=10.5, framealpha=.95)
    A.grid(alpha=.2)
    B = ax[1]
    B.hist(res/res.std(), bins=70, color=BLU, alpha=.8)
    B.axvline(0, color="k", lw=1.5)
    B.set_xlabel("standardised residual   $u$", fontsize=12.5)
    B.set_ylabel("species", fontsize=12.5)
    B.set_title("The residual is the signal\n"
                r"$u=\left[\log_{10}L_1-(a+b\log_{10}m)\right]/\sigma$", fontsize=13.5, fontweight="bold")
    B.text(.03, .97, "u < 0 : shorter leg than the prior\n           expects for that mass\n"
                     "u > 0 : longer leg",
           transform=B.transAxes, va="top", ha="left", fontsize=11.5, bbox=box)
    for s in ("top", "right"): B.spines[s].set_visible(False)
    fig.savefig(f"{OUT}/en_F1_allometry.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"F1  OLS b={b[1]:.3f} R2={r2:.3f} | phylo b={b_p:.3f} sd={sd:.3f}")

# ---------- F2 HWI enters the law ----------
def F2():
    y = D.logL.values
    b1, r1, _ = ols(y, [D.log_m.values])
    b2, r2, _ = ols(y, [D.log_m.values, D.HWI.values])
    p1 = b1[0] + b1[1]*D.log_m.values
    p2 = b2[0] + b2[1]*D.log_m.values + b2[2]*D.HWI.values
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.6), sharex=True, sharey=True,
                           gridspec_kw=dict(wspace=.14))
    for A, p, r, c, cm, t, eq in (
        (ax[0], p1, r1, ORA, "Oranges", "body mass only",
         f"log$_{{10}}L_1$ = {b1[0]:.3f} + {b1[1]:.3f}·log$_{{10}}m$"),
        (ax[1], p2, r2, BLU, "Blues", "body mass + flight efficiency (HWI)",
         f"log$_{{10}}L_1$ = {b2[0]:.3f} + {b2[1]:.3f}·log$_{{10}}m$ − {abs(b2[2]):.4f}·HWI")):
        A.hexbin(p, y, gridsize=52, cmap=cm, mincnt=1, bins="log", linewidths=0)
        lo, hi = 1.0, 2.5
        A.plot([lo, hi], [lo, hi], "--", color="#444", lw=2)
        A.set_xlim(lo, hi); A.set_ylim(lo, hi)
        A.set_title(t, fontsize=13.5, fontweight="bold", color=c)
        A.text(.04, .95, f"R$^2$ = {r:.3f}", transform=A.transAxes, va="top",
               fontsize=22, fontweight="bold", color=c)
        A.text(.04, .80, eq, transform=A.transAxes, va="top", fontsize=11.5,
               bbox=dict(boxstyle="round,pad=.4", fc="white", ec=c, lw=1.4))
        A.set_xlabel("predicted  log$_{10}L_1$", fontsize=12.5); A.grid(alpha=.2)
    ax[0].set_ylabel("observed  log$_{10}L_1$", fontsize=12.5)
    d10 = 100*(10**(10*b2[2]) - 1); d_sw = 100*(10**((44-17)*b2[2]) - 1)
    ax[1].text(.97, .06, f"HWI +10  →  leg  {d10:+.0f}%\n"
                         f"sparrow (17) → waterbird (44)  →  leg  {d_sw:+.0f}%",
               transform=ax[1].transAxes, ha="right", va="bottom", fontsize=12,
               fontweight="bold", color=BLU,
               bbox=dict(boxstyle="round,pad=.45", fc="#eaf3fa", ec=BLU, lw=1.5))
    fig.suptitle(f"Law 2 — flight efficiency belongs in the leg-length law   (n = {n:,})",
                 fontsize=15.5, fontweight="bold", y=1.01)
    fig.savefig(f"{OUT}/en_F2_hwi.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"F2  R2 {r1:.3f} -> {r2:.3f}   c_HWI={b2[2]:.5f}")

# ---------- F3 phylogenetic control ----------
def F3():
    R = pd.read_csv(f"{U}/outputs/pgls/pgls_p3_results.csv")
    yu = D.u.values
    bo, _, _ = ols(yu, [D.HWI.values, D.log_m.values])
    b_ols = bo[1]
    fig, ax = plt.subplots(1, 3, figsize=(14.4, 4.5), gridspec_kw=dict(width_ratios=[1, 1, 1.15], wspace=.30))
    A = ax[0]
    med = float(R.beta.median()); q = R.beta.quantile([.025, .975]).values
    A.barh([1, 0], [b_ols, med], color=[ORA, BLU], height=.55)
    A.errorbar([med], [0], xerr=[[med-q[0]], [q[1]-med]], color="k", capsize=5, lw=1.8)
    A.set_yticks([1, 0]); A.set_yticklabels(["OLS\n(ignores phylogeny)", "PGLS\n(200 trees)"], fontsize=11.5)
    A.set_xlabel(r"$\beta_{\rm HWI}$   (effect on leg residual $u$)", fontsize=12)
    A.axvline(0, color="k", lw=1)
    A.text(med/2, 0, f"{med:.4f}", ha="center", va="center", fontsize=13,
           fontweight="bold", color="white")
    A.text(b_ols/2, 1, f"{b_ols:.4f}", ha="center", va="center", fontsize=13,
           fontweight="bold", color="white")
    A.set_title(f"Shared ancestry explains\n{100*(1-med/b_ols):.0f}% of the naive effect",
                fontsize=13, fontweight="bold")
    A.set_xlim(min(b_ols, med)*1.22, 0.002)
    for s in ("top", "right"): A.spines[s].set_visible(False)
    B = ax[1]
    B.hist(R.beta, bins=28, color=BLU, alpha=.85)
    B.axvline(0, color=RED, lw=2, ls="--"); B.set_xlabel(r"$\beta_{\rm HWI}$", fontsize=12)
    B.set_ylabel("trees", fontsize=12)
    B.set_title(f"…but it never vanishes\n200 trees, all p < {R.p.max():.0e}", fontsize=13, fontweight="bold")
    for s in ("top", "right"): B.spines[s].set_visible(False)
    C = ax[2]
    C.hist(R["lambda"], bins=28, color=GRN, alpha=.85)
    C.set_xlabel(r"Pagel's  $\lambda$", fontsize=12); C.set_ylabel("trees", fontsize=12)
    C.set_title(f"λ = {R['lambda'].median():.3f}\n(0 = no phylogenetic signal, 1 = Brownian)",
                fontsize=13, fontweight="bold")
    for s in ("top", "right"): C.spines[s].set_visible(False)
    fig.suptitle("Is it just shared ancestry?  —  phylogenetic generalised least squares, "
                 f"{len(R)} trees × 2 backbones, n = {int(R.n.iloc[0]):,} species",
                 fontsize=14.5, fontweight="bold", y=1.04)
    fig.savefig(f"{OUT}/en_F3_pgls.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"F3  OLS {b_ols:.4f} -> PGLS {med:.4f}  lambda {R['lambda'].median():.3f}")

# ---------- F4 within-family ----------
def F4():
    W = pd.read_csv(f"{U}/outputs/phylo/within_科.csv").sort_values("delta")
    G = pd.read_csv(f"{U}/outputs/phylo/within_属.csv")
    ns, nt = int((W.delta < 0).sum()), len(W)
    z = (ns - nt/2) / np.sqrt(nt/4)
    gs, gt = int((G.delta < 0).sum()), len(G)
    gz = (gs - gt/2) / np.sqrt(gt/4)
    fig, A = plt.subplots(figsize=(13.6, 5.8))
    col = [BLU if v < 0 else RED for v in W.delta]
    A.bar(range(nt), W.delta, color=col, width=.82)
    A.axhline(0, color="k", lw=1.4)
    A.set_xlim(-1, nt); A.set_xticks([])
    A.set_xlabel(f"{nt} bird families (≥15 species each), sorted", fontsize=12.5)
    A.set_ylabel("leg-length residual:\nhigh-HWI half − low-HWI half  [SD]", fontsize=12.5)
    A.set_title("Independent replication — inside a single family, the better fliers have shorter legs",
                fontsize=15, fontweight="bold")
    A.axhspan(0, W.delta.max()*1.25, color=RED, alpha=.05)
    A.axhspan(W.delta.min()*1.25, 0, color=BLU, alpha=.05)
    A.set_ylim(W.delta.min()*1.22, W.delta.max()*1.30)
    A.text(.02, .05, f"shorter legs:  {ns} families", transform=A.transAxes, fontsize=17,
           fontweight="bold", color=BLU)
    A.text(.98, .93, f"longer legs:  {nt-ns} families", transform=A.transAxes, fontsize=15,
           fontweight="bold", color=RED, ha="right")
    A.text(.02, .95, f"{ns}/{nt} = {100*ns/nt:.0f}% of families\n"
                     f"(50% expected by chance;  z = {z:.1f},  p < 1e-8)",
           transform=A.transAxes, va="top", fontsize=13.5, fontweight="bold", color=BLU, bbox=box)
    A.text(.60, .16,
           "Method: within each family, remove the body-mass effect,\n"
           "split species into high / low HWI halves, compare residuals.\n"
           "Same family ⇒ near-identical ancestry and lifestyle,\n"
           "so the difference can only come from flight efficiency.",
           transform=A.transAxes, fontsize=11, color="#333",
           bbox=dict(boxstyle="round,pad=.45", fc="#f6f6f6", ec="#bbb"))
    A.text(.98, .05, f"stricter, within genus:  {gs}/{gt} = {100*gs/gt:.0f}%   (z = {gz:.1f})",
           transform=A.transAxes, ha="right", fontsize=12.5, fontweight="bold", color=GRN,
           bbox=dict(boxstyle="round,pad=.4", fc="#eef7f2", ec=GRN, lw=1.4))
    for s in ("top", "right"): A.spines[s].set_visible(False)
    fig.savefig(f"{OUT}/en_F4_within.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"F4  family {ns}/{nt} z={z:.1f} | genus {gs}/{gt} z={gz:.1f}")

# ---------- F5 foraging stratum ----------
def F5():
    S = pd.read_csv(f"{U}/outputs/phylo/forstrat.csv")
    MAP = {"水域": "Water", "地面": "Ground", "树上": "Trees", "空中": "Aerial"}
    S["g"] = S["主用途"].map(MAP)
    order = ["Aerial", "Water", "Trees", "Ground"]
    stat = []
    for g in order:
        d = S[S.g == g]
        stat.append((g, len(d), float(d.e.median()), float(d.HWI.median())))
    # 层位占比(0-100%) 与 腿长残差 的偏相关,控制体重 —— 全样本
    Z = np.column_stack([np.ones(len(S)), S.log_m])
    ez = S.e.values - Z @ np.linalg.lstsq(Z, S.e.values, rcond=None)[0]
    pc = []
    for g, zh in zip(order, ["\u7a7a\u4e2d", "\u6c34\u57df", "\u6811\u4e0a", "\u5730\u9762"]):
        v = S[zh].values.astype(float)
        rv = v - Z @ np.linalg.lstsq(Z, v, rcond=None)[0]
        r = float(np.corrcoef(rv, ez)[0, 1]); t = r*np.sqrt((len(S)-3)/(1-r*r))
        pc.append((g, r, t))
    fig, ax = plt.subplots(1, 2, figsize=(13.6, 5.2), gridspec_kw=dict(width_ratios=[1.25, 1], wspace=.26))
    A = ax[0]
    dat = [S[S.g == g].e.values for g in order]
    bp = A.boxplot(dat, vert=False, patch_artist=True, widths=.6, showfliers=False,
                   medianprops=dict(color="k", lw=2.2))
    for p, c in zip(bp["boxes"], [BLU, "#4a90c4", "#9dbf9e", ORA]):
        p.set_facecolor(c); p.set_alpha(.85)
    A.axvline(0, color="k", lw=1.4, ls="--")
    A.set_yticklabels([f"{g}\nn = {k:,}   HWI {h:.0f}" for g, k, _, h in stat], fontsize=12)
    A.set_xlabel("leg-length residual  (mass removed)  [SD]", fontsize=12.5)
    A.set_title("What is the leg actually used for?", fontsize=14, fontweight="bold")
    A.set_xlim(-4, 4); A.grid(axis="x", alpha=.25)
    for i, (g, k, m, h) in enumerate(stat):
        A.text(m, i+1.34, f"{m:+.2f}", ha="center", fontsize=11.5, fontweight="bold")
    A.text(.02, .02, "aerial feeders → shortest legs\nground feeders → longest legs",
           transform=A.transAxes, fontsize=11.5, bbox=box)
    for s in ("top", "right"): A.spines[s].set_visible(False)
    B = ax[1]
    cols = [BLU if r < 0 else ORA for _, r, _ in pc]
    B.barh(range(4), [p[1] for p in pc], color=cols, height=.6)
    B.set_yticks(range(4)); B.set_yticklabels([p[0] for p in pc], fontsize=12)
    B.axvline(0, color="k", lw=1.2)
    B.set_xlabel("partial correlation:  % of foraging done in that stratum\n"
                 "vs leg-length residual   (body mass held constant)", fontsize=11.5)
    B.set_title(f"Leg length tracks what the leg does\nall {len(S):,} species", fontsize=13.5, fontweight="bold")
    for i, (g, r, t) in enumerate(pc):
        B.text(.60, i, f"r = {r:+.3f}   (t = {t:+.0f})", va="center", ha="right",
               fontsize=11.5, fontweight="bold", color=(ORA if r > 0 else BLU))
    B.set_xlim(-.34, .62)
    for s in ("top", "right"): B.spines[s].set_visible(False)
    fig.savefig(f"{OUT}/en_F5_forstrat.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print("F5 " + " | ".join(f"{g} n={k}" for g, k, _, _ in stat)
          + "  ||  " + " | ".join(f"{g} r={r:+.3f}" for g, r, _ in pc))

# ---------- F6 repeated independent evolution ----------
def F6():
    I = pd.read_csv(f"{U}/outputs/phylo/S4_redo40_indep.csv")
    I = I[I.independent == True].sort_values("delta_u")
    sh = int((I.delta_u < 0).sum()); tot = len(I)
    r = float(np.corrcoef(I.mean_HWI, I.delta_u)[0, 1])
    fig, A = plt.subplots(figsize=(13.6, 5.8))
    col = [BLU if v < 0 else RED for v in I.delta_u]
    A.bar(range(tot), I.delta_u, color=col, width=.75)
    A.axhline(0, color="k", lw=1.4)
    A.set_xticks(range(tot))
    dup = I.label.duplicated(keep=False)
    labs = [f"{l}  (n={t})" if d else l for l, t, d in zip(I.label, I.n_tip, dup)]
    A.set_xticklabels(labs, rotation=48, ha="right", fontsize=10.5)
    A.set_ylabel("shift in leg residual at the clade's stem  $\\Delta u$  [SD]", fontsize=12.5)
    A.set_title(f"Repeated, independent evolution — {tot} mutually non-nested clades "
                f"that changed leg length on their own",
                fontsize=15, fontweight="bold")
    A.text(.02, .06, f"shortened: {sh}", transform=A.transAxes, fontsize=15,
           fontweight="bold", color=BLU)
    A.text(.98, .93, f"lengthened: {tot-sh}", transform=A.transAxes, fontsize=14,
           fontweight="bold", color=RED, ha="right")
    A.text(.28, .96, f"correlation with clade flight efficiency:   r = {r:+.3f}",
           transform=A.transAxes, va="top", fontsize=13.5, fontweight="bold", color=BLU, bbox=box)
    for i, (d, h) in enumerate(zip(I.delta_u, I.mean_HWI)):
        A.text(i, d + (-.10 if d < 0 else .10), f"{h:.0f}", ha="center",
               va="top" if d < 0 else "bottom", fontsize=9, color="#555")
    A.text(.98, .74, "number at each bar = mean HWI of that clade",
           transform=A.transAxes, ha="right", fontsize=11, color="#555",
           bbox=dict(boxstyle="round,pad=.35", fc="#f6f6f6", ec="#ccc"))
    A.set_ylim(I.delta_u.min()*1.30, max(.9, I.delta_u.max()*1.55))
    for s in ("top", "right"): A.spines[s].set_visible(False)
    fig.savefig(f"{OUT}/en_F6_repeat.png", dpi=170, bbox_inches="tight"); plt.close(fig)
    print(f"F6  {sh}/{tot} shortened, r={r:+.3f}")

for f in (F1, F2, F3, F4, F5, F6): f()
print("→", OUT)

# ---------- F46 两个尺度的独立重复（科内 + 跨支系） ----------
def F46():
    W = pd.read_csv(f"{U}/outputs/phylo/within_科.csv").sort_values("delta")
    G = pd.read_csv(f"{U}/outputs/phylo/within_属.csv")
    I = pd.read_csv(f"{U}/outputs/phylo/S4_redo40_indep.csv")
    I = I[I.independent == True].sort_values("delta_u")
    ns, nt = int((W.delta < 0).sum()), len(W); z = (ns - nt/2)/np.sqrt(nt/4)
    gs, gt = int((G.delta < 0).sum()), len(G); gz = (gs - gt/2)/np.sqrt(gt/4)
    sh, tot = int((I.delta_u < 0).sum()), len(I)
    rr = float(np.corrcoef(I.mean_HWI, I.delta_u)[0, 1])
    fig, ax = plt.subplots(2, 1, figsize=(14.0, 8.6), gridspec_kw=dict(hspace=.62, height_ratios=[1, 1.18]))
    A = ax[0]
    A.bar(range(nt), W.delta, color=[BLU if v < 0 else RED for v in W.delta], width=.82)
    A.axhline(0, color="k", lw=1.3); A.set_xlim(-1, nt); A.set_xticks([])
    A.set_xlabel(f"{nt} families (≥15 species each), sorted", fontsize=11.5)
    A.set_ylabel("high-HWI half − low-HWI half\nleg residual  [SD]", fontsize=11.5)
    A.set_title("(a)  Inside one family:  the better-flying half has shorter legs",
                fontsize=14, fontweight="bold", loc="left")
    A.text(.015, .17, f"{ns}/{nt} = {100*ns/nt:.0f}% of families      "
                      f"(50% expected by chance;  z = {z:.1f})",
           transform=A.transAxes, fontsize=13, fontweight="bold", color=BLU, bbox=box)
    A.text(.985, .93, f"stricter, within genus:  {gs}/{gt} = {100*gs/gt:.0f}%   (z = {gz:.1f})",
           transform=A.transAxes, ha="right", va="top", fontsize=12, fontweight="bold", color=GRN,
           bbox=dict(boxstyle="round,pad=.35", fc="#eef7f2", ec=GRN, lw=1.3))
    A.set_ylim(W.delta.min()*1.32, W.delta.max()*1.35)
    for s in ("top", "right"): A.spines[s].set_visible(False)
    B = ax[1]
    B.bar(range(tot), I.delta_u, color=[BLU if v < 0 else RED for v in I.delta_u], width=.75)
    B.axhline(0, color="k", lw=1.3); B.set_xticks(range(tot))
    dup = I.label.duplicated(keep=False)
    B.set_xticklabels([f"{l} (n={t})" if d else l for l, t, d in zip(I.label, I.n_tip, dup)],
                      rotation=46, ha="right", fontsize=10)
    B.set_ylabel("shift in leg residual\nat the clade's stem  $\\Delta u$  [SD]", fontsize=11.5)
    B.set_title(f"(b)  Across the tree:  {tot} mutually non-nested clades that changed leg length independently",
                fontsize=14, fontweight="bold", loc="left")
    B.text(.30, .10, f"clade mean HWI vs $\\Delta u$ :   r = {rr:+.3f}",
           transform=B.transAxes, fontsize=13, fontweight="bold", color=BLU, bbox=box)
    B.text(.30, .97, f"shortened {sh}   ·   lengthened {tot-sh}", transform=B.transAxes,
           ha="left", va="top", fontsize=12.5, fontweight="bold", color="#444")
    B.set_ylim(I.delta_u.min()*1.22, max(1.0, I.delta_u.max()*1.35))
    for s in ("top", "right"): B.spines[s].set_visible(False)
    fig.suptitle("The trade-off is replicated independently at two scales — it is not one ancient accident",
                 fontsize=16, fontweight="bold", y=.985)
    fig.savefig(f"{OUT}/en_F46_replication.png", dpi=165, bbox_inches="tight"); plt.close(fig)
    print(f"F46 fam {ns}/{nt} | genus {gs}/{gt} | clades {sh}/{tot} r={rr:+.3f}")

F46()

# -*- coding: utf-8 -*-
"""Render the swan-vs-generator drop test (English only, no CJK glyphs).
    python anim_swan_render.py [slow=150]
"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.linewidth": .9})
D_ = "outputs/anim_swan"; SLOW = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
D = np.load(f"{D_}/hist_swan.npz", allow_pickle=True)
J = json.load(open(f"{D_}/swan_compare.json")); M = J["meta"]
K = ["bio", "phys"]; COL = {k: M[k]["color"] for k in K}
SMAX, GCAP = J["smax_mm"], J["gcap_g"]

FPS, PRE, POST = 30, 0.004, 0.076
S = {}
for k in K:
    t = D[k + "__t"]; rf = float(D[k + "__r_foot"]); foot = D[k + "__foot"]; z = D[k + "__z"]
    i0 = int(np.argmax(foot[:, 2] <= rf))
    sink_max = float(max(0., rf - np.min(foot[:, 2])))
    S[k] = dict(tr=t - t[i0], g=D[k + "__az"] / 9.81,
                stroke=1e3 * np.clip((z[0] - z) - sink_max, 0, None),
                Jt=[foot, D[k + "__ankle_p"], D[k + "__knee_p"], D[k + "__hip_p"]])
TR = np.linspace(-PRE, POST, int(round(FPS * SLOW * (PRE + POST))))
sm = lambda k, n: np.interp(TR, S[k]["tr"], S[k][n])
G = {k: sm(k, "g") for k in K}; ST = {k: sm(k, "stroke") for k in K}
JT = {k: np.stack([[np.interp(TR, S[k]["tr"], Q[:, j]) for j in (0, 2)] for Q in S[k]["Jt"]],
                  0).transpose(0, 2, 1) for k in K}

fig = plt.figure(figsize=(13.4, 8.5))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=.40, wspace=.20,
                      top=.815, bottom=.185, left=.065, right=.975)
axL = [fig.add_subplot(gs[0, i]) for i in range(2)]
axG = fig.add_subplot(gs[1, 0]); axS = fig.add_subplot(gs[1, 1])
yhi = max(JT[k][:, :, 1].max() for k in K)
lines = []
for i, k in enumerate(K):
    A = axL[i]; m = M[k]
    xc = float(JT[k][:, :, 0].mean()); w = .62 * yhi
    A.set_xlim(xc - w, xc + w); A.set_ylim(-.035, yhi * 1.10)
    A.set_aspect("equal", adjustable="box"); A.axis("off")
    A.axhline(0, color="#8a7f6d", lw=3.5)
    A.set_title(f"{m['label']}\n$L_1$ = {m['L1_mm']:.0f} mm   "
                r"$\theta_A$ = " + f"{m['thA']:.0f}"u"°"
                f"   leg mass {m['leg_mass_g']:.0f} g",
                fontsize=14.5, fontweight="bold", color=COL[k], pad=12)
    ln, = A.plot([], [], "-o", color=COL[k], lw=5, ms=9, mec="k", mew=.8)
    tx = A.text(.5, -.06, "", transform=A.transAxes, ha="center", fontsize=14,
                fontweight="bold", color=COL[k])
    lines.append((ln, tx))
for A, dat, lab, lim, tag in ((axG, G,  "deceleration (g)", GCAP, f"{GCAP:.0f} g"),
                              (axS, ST, "leg stroke (mm)",  SMAX, f"{SMAX:.0f} mm")):
    A.set_xlim(TR[0] * 1e3, TR[-1] * 1e3)
    A.set_xlabel("time after touchdown (ms)", fontsize=12)
    A.set_ylabel(lab, fontsize=12.5)
    A.axhline(lim, color="#c0392b", lw=1.8, ls="--")
    A.text(TR[-1] * 1e3, lim, f" {tag} ", color="#c0392b", fontsize=11, va="bottom", ha="right")
    A.set_ylim(0, max(lim * 1.22, max(max(abs(dat[k])) for k in K) * 1.15))
    A.grid(alpha=.22)
    for sp in ("top", "right"): A.spines[sp].set_visible(False)
cur = {k: (axG.plot([], [], color=COL[k], lw=3, label=M[k]["label"])[0],
           axS.plot([], [], color=COL[k], lw=3)[0]) for k in K}
axG.legend(loc="upper left", fontsize=10.5, framealpha=.9)

def upd(i):
    for j, k in enumerate(K):
        lines[j][0].set_data(JT[k][:, i, 0], JT[k][:, i, 1])
        lines[j][1].set_text(f"{G[k][i]:.1f} g    {ST[k][i]:.0f} mm")
        cur[k][0].set_data(TR[:i + 1] * 1e3, G[k][:i + 1])
        cur[k][1].set_data(TR[:i + 1] * 1e3, ST[k][:i + 1])
    return []

b, p = M["bio"], M["phys"]
fig.suptitle("Trumpeter swan (11.1 kg, measured geometry) vs the generator at the same mass\n"
             f"{J['v0']} m/s  ·  rigid ground  ·  identical joint stiffness  ·  "
             f"{SLOW:g}$\\times$ slow motion",
             fontsize=15, fontweight="bold", y=.985, linespacing=1.5)
fig.text(.5, .058,
         f"peak  {b['peak_g']:.2f} / {p['peak_g']:.2f} g  ({100*(p['peak_g']/b['peak_g']-1):+.0f}%)      "
         f"stroke  {b['leg_stroke_mm']:.1f} / {p['leg_stroke_mm']:.1f} mm      "
         f"leg mass  {b['leg_mass_g']:.0f} / {p['leg_mass_g']:.0f} g "
         f"({100*(p['leg_mass_g']/b['leg_mass_g']-1):+.0f}%)",
         ha="center", fontsize=13.5, fontweight="bold", color="#222")
fig.text(.5, .016,
         "bird geometry: AVONET tarsus (n = 4) + Watanabe 2017 segment ratios + touchdown angles from waterbird video; "
         "joint stiffness is shared - birds have none measured",
         ha="center", fontsize=10, color="#6b6b6b")
upd(int(np.argmax(ST["phys"])))          # 取生成器行程峰值那一帧做静帧
png = f"{D_}/swan_vs_gen_last.png"; fig.savefig(png, dpi=150); print("[png]", png)
try:
    w = FFMpegWriter(fps=FPS, bitrate=5200)
    with w.saving(fig, f"{D_}/swan_vs_gen.mp4", dpi=110):
        for i in range(len(TR)):
            upd(i); w.grab_frame()
    print("[mp4]", f"{D_}/swan_vs_gen.mp4")
except Exception as e:
    print("[no mp4]", e)

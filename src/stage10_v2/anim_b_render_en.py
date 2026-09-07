# -*- coding: utf-8 -*-
"""Side-by-side drop test, English only (no CJK glyphs required).
Renders from an existing hist_<TAG>.npz — no re-simulation.

    python src/stage10_v2/anim_b_render_en.py outputs/anim_b [slow=150] [tag=hard]

Outputs  b_compare_<TAG>_en.mp4  and  b_compare_<TAG>_en_last.png
"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"],
                     "axes.unicode_minus": False, "figure.facecolor": "white",
                     "savefig.facecolor": "white", "axes.linewidth": .9})
D_   = sys.argv[1] if len(sys.argv) > 1 else "outputs/anim_b"
SLOW = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
TAG  = sys.argv[3] if len(sys.argv) > 3 else "hard"
D = np.load(os.path.join(D_, f"hist_{TAG}.npz"), allow_pickle=True)
J = json.load(open(os.path.join(D_, f"b_compare_{TAG}.json")))
M = J["meta"]; K = ["bio", "phys"]
COL = {"bio": "#8E2A34", "phys": "#1b6ca8"}
LAB = {"bio": ("waterbird allometry", J["b_bio"]), "phys": ("generator", J["b_machine"])}
GROUND = {1.0e6: "rigid ground", 1.0e5: "turf", 5.0e4: "wet sand"}.get(J["kc"], f"k_c = {J['kc']:.0e}")
FPS, PRE, POST = 30, 0.004, 0.076
S = {}
for k in K:
    t = D[k+"__t"]; rf = float(D[k+"__r_foot"]); foot = D[k+"__foot"]; z = D[k+"__z"]
    i0 = int(np.argmax(foot[:, 2] <= rf))
    sink_max = float(max(0., rf - np.min(foot[:, 2])))      # same convention as feasible_v2
    S[k] = dict(tr=t-t[i0], g=D[k+"__az"]/9.81,
                stroke=1e3*np.clip((z[0]-z)-sink_max, 0, None),
                Jt=[foot, D[k+"__ankle_p"], D[k+"__knee_p"], D[k+"__hip_p"]])
TR = np.linspace(-PRE, POST, int(round(FPS*SLOW*(PRE+POST))))
sm = lambda k, n: np.interp(TR, S[k]["tr"], S[k][n])
G = {k: sm(k, "g") for k in K}; ST = {k: sm(k, "stroke") for k in K}
JT = {k: np.stack([[np.interp(TR, S[k]["tr"], Q[:, j]) for j in (0, 2)] for Q in S[k]["Jt"]],
                  0).transpose(0, 2, 1) for k in K}
fig = plt.figure(figsize=(13.4, 7.6))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=.38, wspace=.20,
                      top=.885, bottom=.095, left=.065, right=.975)
axL = [fig.add_subplot(gs[0, i]) for i in range(2)]
axG = fig.add_subplot(gs[1, 0]); axS = fig.add_subplot(gs[1, 1])
yhi = max(JT[k][:, :, 1].max() for k in K); lines = []
for i, k in enumerate(K):
    A = axL[i]; m = M[k]; name, b = LAB[k]
    xc = float(JT[k][:, :, 0].mean()); w = .62*yhi
    A.set_xlim(xc-w, xc+w); A.set_ylim(-.035, yhi*1.10)
    A.set_aspect("equal", adjustable="box"); A.axis("off")
    A.axhline(0, color="#8a7f6d", lw=3.5)
    A.set_title(f"$b$ = {b:.3f}   ({name})\n$L_1$ = {m['L1_mm']:.0f} mm    leg mass {m['leg_mass_g']:.0f} g",
                fontsize=14.5, fontweight="bold", color=COL[k], pad=12)
    ln, = A.plot([], [], "-o", color=COL[k], lw=5, ms=9, mec="k", mew=.8)
    tx = A.text(.5, -.06, "", transform=A.transAxes, ha="center", fontsize=14,
                fontweight="bold", color=COL[k])
    lines.append((ln, tx))
for A, dat, lab, lim, tag in ((axG, G, "deceleration (g)", 10.0, "10 g"),
                              (axS, ST, "leg stroke (mm)", 24.0, "24 mm")):
    A.set_xlim(TR[0]*1e3, TR[-1]*1e3)
    A.set_xlabel("time after touchdown (ms)", fontsize=12)
    A.set_ylabel(lab, fontsize=12.5)
    A.axhline(lim, color="#c0392b", lw=1.8, ls="--")
    A.text(TR[-1]*1e3, lim, f" {tag} ", color="#c0392b", fontsize=11, va="bottom", ha="right")
    A.set_ylim(0, max(lim*1.22, max(max(abs(dat[k])) for k in K)*1.15))
    A.grid(alpha=.22)
    for sp in ("top", "right"): A.spines[sp].set_visible(False)
cur = {k: (axG.plot([], [], color=COL[k], lw=3, label=f"$b$ = {LAB[k][1]:.3f}")[0],
           axS.plot([], [], color=COL[k], lw=3)[0]) for k in K}
axG.legend(loc="upper left", fontsize=11, framealpha=.9)
def upd(i):
    for j, k in enumerate(K):
        lines[j][0].set_data(JT[k][:, i, 0], JT[k][:, i, 1])
        lines[j][1].set_text(f"{G[k][i]:.1f} g    {ST[k][i]:.0f} mm")
        cur[k][0].set_data(TR[:i+1]*1e3, G[k][:i+1])
        cur[k][1].set_data(TR[:i+1]*1e3, ST[k][:i+1])
    return []
b, p = M["bio"], M["phys"]
fig.suptitle(f"{J['m_kg']:.0f} kg airframe  ·  {J['v0']} m/s  ·  {GROUND}  ·  identical joints  "
             f"·  {SLOW:g}$\\times$ slow motion", fontsize=15, fontweight="bold", y=.965)
fig.text(.5, .018,
         f"peak  {b['peak_g']:.2f} / {p['peak_g']:.2f} g      "
         f"stroke  {b['leg_stroke_mm']:.1f} / {p['leg_stroke_mm']:.1f} mm      "
         f"leg mass  {b['leg_mass_g']:.0f} / {p['leg_mass_g']:.0f} g "
         f"({100*(p['leg_mass_g']/b['leg_mass_g']-1):+.0f}%)",
         ha="center", fontsize=13.5, fontweight="bold", color="#222")
upd(len(TR)-1)
png = os.path.join(D_, f"b_compare_{TAG}_en_last.png")
fig.savefig(png, dpi=150, bbox_inches="tight"); print("[png]", png)
print(f"[data] b_bio={J['b_bio']:.3f} b_machine={J['b_machine']:.3f}  "
      f"L1 {b['L1_mm']:.0f}/{p['L1_mm']:.0f} mm  peak {b['peak_g']:.2f}/{p['peak_g']:.2f} g  "
      f"stroke {b['leg_stroke_mm']:.1f}/{p['leg_stroke_mm']:.1f} mm  "
      f"mass {b['leg_mass_g']:.0f}/{p['leg_mass_g']:.0f} g")
mp4 = os.path.join(D_, f"b_compare_{TAG}_en.mp4")
try:
    w = FFMpegWriter(fps=FPS, bitrate=5200)
    with w.saving(fig, mp4, dpi=110):
        for i in range(len(TR)):
            upd(i); w.grab_frame()
    print("[mp4]", mp4)
except Exception as e:
    print("[no mp4]", e)

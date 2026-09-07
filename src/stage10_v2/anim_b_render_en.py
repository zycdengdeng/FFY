# -*- coding: utf-8 -*-
"""从 hist_b.npz 重新渲染英文版并排落震（不重跑仿真）。
用法: python src/stage10_v2/anim_b_render_en.py outputs/anim_b [slow=150]"""
import os, sys, json, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
D_ = sys.argv[1] if len(sys.argv) > 1 else "outputs/anim_b"
SLOW = float(sys.argv[2]) if len(sys.argv) > 2 else 150.0
TAG  = sys.argv[3] if len(sys.argv) > 3 else "hard"
npz = os.path.join(D_, f"hist_{TAG}.npz"); D = np.load(npz, allow_pickle=True)
J = json.load(open(os.path.join(D_, f"b_compare_{TAG}.json")))
M = J["meta"]; K = ["bio", "phys"]
LAB = {"bio": ("BIRD'S EXPONENT", f"$b$ = {J['b_bio']:.3f}"),
       "phys": ("MACHINE'S EXPONENT", f"$b$ = {J['b_machine']:.3f}")}
COL = {"bio": "#2E7D5B", "phys": "#1b6ca8"}
FPS, PRE, POST = 30, 0.004, 0.076
S = {}
for k in K:
    t = D[k+"__t"]; rf = float(D[k+"__r_foot"]); foot = D[k+"__foot"]; z = D[k+"__z"]
    i0 = int(np.argmax(foot[:, 2] <= rf))
    sink_max = float(max(0., rf - np.min(foot[:, 2])))   # 与 physics_v2 的判据口径一致
    S[k] = dict(tr=t-t[i0], g=D[k+"__az"]/9.81,
                stroke=1e3*np.clip((z[0]-z)-sink_max, 0, None),
                Jt=[foot, D[k+"__ankle_p"], D[k+"__knee_p"], D[k+"__hip_p"]])
TR = np.linspace(-PRE, POST, int(round(FPS*SLOW*(PRE+POST))))
sm = lambda k, n: np.interp(TR, S[k]["tr"], S[k][n])
G = {k: sm(k, "g") for k in K}; ST = {k: sm(k, "stroke") for k in K}
JT = {k: np.stack([[np.interp(TR, S[k]["tr"], Q[:, j]) for j in (0, 2)] for Q in S[k]["Jt"]],
                  0).transpose(0, 2, 1) for k in K}
fig = plt.figure(figsize=(13.4, 8.0))
gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1], hspace=.40, wspace=.22,
                      top=.865, bottom=.115, left=.065, right=.975)
axL = [fig.add_subplot(gs[0, i]) for i in range(2)]
axG = fig.add_subplot(gs[1, 0]); axS = fig.add_subplot(gs[1, 1])
lines = []
yhi = max(JT[k][:, :, 1].max() for k in K)
for i, k in enumerate(K):
    A = axL[i]; m = M[k]
    xc = float(JT[k][:, :, 0].mean()); w = .62*yhi
    A.set_xlim(xc-w, xc+w); A.set_ylim(-.035, yhi*1.10)
    A.set_aspect("equal", adjustable="box"); A.axis("off")
    A.axhline(0, color="#8a7f6d", lw=3.5)
    n1, n2 = LAB[k]
    A.set_title(f"{n1}    {n2}\n$L_1$ = {m['L1_mm']:.0f} mm       leg mass {m['leg_mass_g']:.0f} g",
                fontsize=15, fontweight="bold", color=COL[k], pad=14)
    ln, = A.plot([], [], "-o", color=COL[k], lw=5, ms=9, mec="k", mew=.8)
    tx = A.text(.5, -.055, "", transform=A.transAxes, ha="center", fontsize=14,
                fontweight="bold", color=COL[k])
    lines.append((ln, tx))
for A, dat, lab, lim, ll in ((axG, G, "deceleration / g", 10.0, "10 g limit"),
                             (axS, ST, "stroke / mm", 24.0, "24 mm budget")):
    A.set_xlim(TR[0]*1e3, TR[-1]*1e3); A.set_xlabel("time after touchdown / ms", fontsize=12)
    A.set_ylabel(lab, fontsize=12.5)
    A.axhline(lim, color="#c0392b", lw=2, ls="--"); A.grid(alpha=.25)
    A.text(TR[-1]*1e3, lim, " "+ll+" ", color="#c0392b", fontsize=11, va="bottom", ha="right")
    A.set_ylim(0, max(lim*1.22, max(max(abs(dat[k])) for k in K)*1.15))
    for sp in ("top", "right"): A.spines[sp].set_visible(False)
cur = {k: (axG.plot([], [], color=COL[k], lw=3, label=LAB[k][0].title())[0],
           axS.plot([], [], color=COL[k], lw=3)[0]) for k in K}
axG.legend(loc="upper left", fontsize=10.5, framealpha=.9)
def upd(i):
    for j, k in enumerate(K):
        lines[j][0].set_data(JT[k][:, i, 0], JT[k][:, i, 1])
        lines[j][1].set_text(f"{G[k][i]:.1f} g      {ST[k][i]:.0f} mm")
        cur[k][0].set_data(TR[:i+1]*1e3, G[k][:i+1])
        cur[k][1].set_data(TR[:i+1]*1e3, ST[k][:i+1])
    return []
b, p = M["bio"], M["phys"]
fig.suptitle(f"Same joints, same {J['m_kg']:.0f} kg airframe, same {J['v0']} m/s touchdown "
             f"— only the leg-length scaling law differs      ({SLOW:g}$\\times$ slow motion)",
             fontsize=15, fontweight="bold", y=.975)
fig.text(.5, .022,
    f"peak  {b['peak_g']:.2f} g  vs  {p['peak_g']:.2f} g  ({100*(p['peak_g']/b['peak_g']-1):+.0f}%)"
    f"      ·      stroke  {b['leg_stroke_mm']:.1f} mm  vs  {p['leg_stroke_mm']:.1f} mm"
    f"      ·      leg mass  {b['leg_mass_g']:.0f} g  vs  {p['leg_mass_g']:.0f} g  "
    f"({100*(p['leg_mass_g']/b['leg_mass_g']-1):+.0f}%)   — both feasible",
    ha="center", fontsize=13.5, fontweight="bold", color="#222")
upd(len(TR)-1)
png = os.path.join(D_, f"b_compare_{TAG}_en_last.png")
fig.savefig(png, dpi=150, bbox_inches="tight"); print("[png]", png)
mp4 = os.path.join(D_, f"b_compare_{TAG}_en.mp4")
try:
    w = FFMpegWriter(fps=FPS, bitrate=5200)
    with w.saving(fig, mp4, dpi=110):
        for i in range(len(TR)):
            upd(i); w.grab_frame()
    print("[mp4]", mp4)
except Exception as e:
    print("[no mp4]", e)

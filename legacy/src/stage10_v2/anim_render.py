# -*- coding: utf-8 -*-
"""落地对比动画:同一工况(硬地/12kg/1.2m·s⁻¹)、同一套刚度,只变几何。"""
import json, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.animation import FFMpegWriter

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP"]
plt.rcParams["axes.unicode_minus"] = False

D = np.load("/tmp/anim/hist.npz", allow_pickle=True)
META = json.loads(str(D["meta"]))
KEYS = ["ai", "swan", "duck"]
COL = {"ai": "#0E7C86", "swan": "#2E8B57", "duck": "#C75B00"}
NAME = {"ai": "AI 生成设计", "swan": "真鸟·疣鼻天鹅", "duck": "朴素仿生·绿头鸭腿原样"}
SUB = {"ai": "跗跖 L1 = 100.1 mm(网络自选)", "swan": "跗跖 L1 = 93.6 mm(AVONET 实测)",
       "duck": "跗跖 L1 = 39.75 mm(843 g 的腿,原样搬到 12 kg)"}
GCAP = 10.0

SLOW = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
FPS = 30
PRE, POST = 0.004, 0.076          # 触地前后取窗
OUT = sys.argv[2] if len(sys.argv) > 2 else "/tmp/anim/landing_compare.mp4"


# ---------- 取窗 + 对齐触地时刻 ----------
S = {}
for k in KEYS:
    t = D[k + "__t"]; rf = float(D[k + "__r_foot"])
    foot = D[k + "__foot"]; z = D[k + "__z"]
    i0 = int(np.argmax(foot[:, 2] <= rf))
    tc = t[i0]
    sink = np.maximum(0.0, rf - foot[:, 2])
    S[k] = dict(
        tr=t - tc,                                   # 相对触地时刻
        g=D[k + "__az"] / 9.81,
        stroke=1e3 * ((z[0] - z) - sink),            # 腿自身行程 mm
        M=np.abs(D[k + "__M_hip"]),                  # 髋关节力矩 N·m
        foot=foot, rf=rf,
        joints=[foot, D[k + "__ankle_p"], D[k + "__knee_p"], D[k + "__hip_p"]],
        seg=D[k + "__seg_len"])

TR = np.linspace(-PRE, POST, int(round(FPS * SLOW * (PRE + POST))))
NF = len(TR)


def samp(k, name):
    return np.interp(TR, S[k]["tr"], S[k][name])


G = {k: samp(k, "g") for k in KEYS}
ST = {k: samp(k, "stroke") for k in KEYS}
MM = {k: samp(k, "M") for k in KEYS}
FT = {k: np.stack([np.interp(TR, S[k]["tr"], S[k]["foot"][:, j]) for j in (0, 2)], 1)
      for k in KEYS}
JT = {k: np.stack([[np.interp(TR, S[k]["tr"], P[:, j]) for j in (0, 2)]
                   for P in S[k]["joints"]], 0).transpose(0, 2, 1)   # (4关节, 帧, xz)
      for k in KEYS}

# ---------- 画布 ----------
fig = plt.figure(figsize=(16, 9), dpi=120)
fig.patch.set_facecolor("#FBFAF8")
gs = fig.add_gridspec(3, 2, width_ratios=[1.12, 1.0],
                      left=0.035, right=0.975, top=0.845, bottom=0.055,
                      wspace=0.15, hspace=0.34)
axL = fig.add_subplot(gs[0:2, 0]); axL.set_facecolor("#FBFAF8")
axC = fig.add_subplot(gs[2, 0]); axC.axis("off")
axs = [fig.add_subplot(gs[i, 1]) for i in range(3)]

fig.text(0.035, 0.955, "同一工况下的落地对比:AI 生成设计 vs 真实水鸟骨架",
         fontsize=21, color="#8E2A34", fontweight="bold")
fig.text(0.035, 0.912,
         "m = 12 kg    v0 = 1.2 m/s    硬地 k_c = 1e6 N/m    "
         "三者关节刚度 κ 与阻尼比 ζ 完全相同,唯一变量是几何(腿长与比例)",
         fontsize=11.5, color="#666666")
fig.text(0.975, 0.955, f"慢放 {SLOW:.0f}×", fontsize=13, color="#999999", ha="right")

# ---------- 左:三条腿 ----------
LANE = {"ai": 0.0, "swan": 0.335, "duck": 0.655}
axL.set_xlim(-0.165, 0.815); axL.set_ylim(-0.088, 0.428)
axL.set_aspect("equal"); axL.axis("off")
axL.fill_between([-0.165, 0.815], -0.088, 0, color="#D8D2C8", zorder=0)
axL.plot([-0.165, 0.815], [0, 0], color="#8A8378", lw=2.0, zorder=1)
axL.text(-0.157, -0.062, "硬地面", fontsize=9.5, color="#7A7368")

rods, foots, bodies, tags, lanetxt = {}, {}, {}, {}, {}
for k in KEYS:
    x0 = LANE[k]; c = COL[k]
    rods[k] = [axL.plot([], [], color=c, lw=lw, solid_capstyle="round", zorder=4)[0]
               for lw in (5.5, 6.5, 7.5)]
    foots[k] = Circle((0, 0), S[k]["rf"], color=c, zorder=5); axL.add_patch(foots[k])
    bodies[k] = FancyBboxPatch((x0 - 0.062, 0), 0.124, 0.038,
                               boxstyle="round,pad=0.004,rounding_size=0.009",
                               fc=c, ec="none", alpha=0.88, zorder=3)
    axL.add_patch(bodies[k])
    axL.text(x0, 0.398, NAME[k], fontsize=12.5, color=c,
             fontweight="bold", ha="center")
    axL.text(x0, 0.375, SUB[k], fontsize=8.6, color="#777777", ha="center")
    tags[k] = axL.text(x0, -0.052, "", fontsize=12.5, ha="center",
                       fontweight="bold", color=c)
    lanetxt[k] = axL.text(x0, 0.352, "", fontsize=9.2, ha="center", color="#555555")


def draw_leg(k, i):
    """用实测关节位置复原骨架:足 → 踝 → 膝 → 髋(不依赖转角符号约定)。"""
    x0 = LANE[k]
    pts = JT[k][:, i, :].copy()
    pts[:, 0] += x0
    for j, ln in enumerate(rods[k]):
        ln.set_data(pts[j:j + 2, 0], pts[j:j + 2, 1])
    foots[k].center = (pts[0, 0], pts[0, 1])
    bodies[k].set_x(pts[3, 0] - 0.062); bodies[k].set_y(pts[3, 1])


# ---------- 右:三条曲线 ----------
SPEC = [("机体加速度  |a| / g", G, "#8E2A34"),
        ("起落架行程 / mm", ST, None),
        ("髋关节力矩 / N·m", MM, None)]
lines, dots, now = {}, {}, []
for ax, (ttl, dat, _) in zip(axs, SPEC):
    ax.set_facecolor("#FFFFFF")
    ax.set_xlim(-PRE * 1e3, POST * 1e3)
    hi = max(np.max(v) for v in dat.values())
    ax.set_ylim(-0.06 * hi, 1.16 * hi)
    ax.set_title(ttl, fontsize=13, color="#333333", loc="left", pad=6)
    ax.grid(alpha=0.25, lw=0.6)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for k in KEYS:
        ax.plot(TR * 1e3, dat[k], color=COL[k], lw=1.0, alpha=0.16, zorder=1)
        lines[(id(ax), k)] = ax.plot([], [], color=COL[k], lw=2.3, zorder=3)[0]
        dots[(id(ax), k)] = ax.plot([], [], "o", color=COL[k], ms=6, zorder=4)[0]
    now.append(ax.axvline(0, color="#999999", lw=1.1, ls="--", zorder=2))
axs[0].axhline(GCAP, color="#8E2A34", lw=1.4, ls=":", zorder=2)
axs[0].text(POST * 1e3, GCAP, " g_cap = 10 g", fontsize=10, color="#8E2A34",
            va="bottom", ha="right")
axs[2].set_xlabel("触地后时间 / ms", fontsize=11, color="#555555")

ROWS = [("峰值加速度", "peak_g", "{:.2f} g"), ("起落架行程", "leg_stroke_mm", "{:.1f} mm"),
        ("股骨管外径", "_D", "{:.1f} mm"), ("单腿结构质量", "leg_mass_g", "{:.0f} g")]
axC.set_xlim(0, 1); axC.set_ylim(0, 1)
axC.text(0.0, 0.93, "落震结束后的判定(同一套可行性尺子)",
         fontsize=12.5, color="#8E2A34", fontweight="bold")
XC = {"ai": 0.335, "swan": 0.615, "duck": 0.895}
for k in KEYS:
    mt = META[k]; mt["_D"] = mt["D_mm"][2]
    axC.text(XC[k], 0.77, NAME[k].split("·")[-1], fontsize=10.5, color=COL[k],
             ha="center", fontweight="bold")
for r, (lbl, key, fm) in enumerate(ROWS):
    y = 0.61 - 0.145 * r
    axC.text(0.0, y, lbl, fontsize=10.5, color="#555555")
    for k in KEYS:
        v = META[k][key]
        bad = (key == "peak_g" and v > GCAP)
        axC.text(XC[k], y, fm.format(v), fontsize=11, ha="center",
                 color="#8E2A34" if bad else "#333333",
                 fontweight="bold" if bad else "normal")
WHY = {"gcap": "过载超限", "slenderness": "细长比超限", "smax": "行程超限",
       "mass": "质量超预算", "stress": "强度不足", "buckling": "屈曲"}
for k in KEYS:
    mt = META[k]
    txt = "✓ 可行" if mt["ok"] else "× " + "、".join(WHY.get(w, w) for w in mt["why"])
    axC.text(XC[k], 0.035, txt, fontsize=11.5, ha="center", fontweight="bold",
             color="#2E8B57" if mt["ok"] else "#8E2A34")

verdict = axL.text(0.805, -0.062, "", fontsize=11, color="#555555", ha="right")
PEAK = {k: float(np.max(G[k])) for k in KEYS}


def update(i):
    tnow = TR[i] * 1e3
    for k in KEYS:
        draw_leg(k, i)
        tags[k].set_text(f"{G[k][i]:.1f} g")
        tags[k].set_color("#8E2A34" if G[k][i] > GCAP else COL[k])
        lanetxt[k].set_text(f"行程 {ST[k][i]:5.1f} mm     髋力矩 {MM[k][i]:5.1f} N·m")
    for ax, (_, dat, _) in zip(axs, SPEC):
        for k in KEYS:
            lines[(id(ax), k)].set_data(TR[:i + 1] * 1e3, dat[k][:i + 1])
            dots[(id(ax), k)].set_data([tnow], [dat[k][i]])
    for ln in now:
        ln.set_xdata([tnow, tnow])
    if TR[i] < 0:
        verdict.set_text("自由下落")
        verdict.set_color("#AAAAAA")
    else:
        verdict.set_text(f"触地后 {tnow:5.1f} ms")
        verdict.set_color("#555555")
    return []


w = FFMpegWriter(fps=FPS, bitrate=5200,
                 metadata=dict(title="landing compare"))
with w.saving(fig, OUT, dpi=120):
    for i in range(NF):
        update(i); w.grab_frame()
        if i % 60 == 0:
            print(f"  {i}/{NF}", flush=True)
print("done", OUT)

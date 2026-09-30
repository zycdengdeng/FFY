# -*- coding: utf-8 -*-
"""生成能力展示：三个工况 → 三条生成的腿 → 各自的落震结果（同一比例尺）。
数据 = E20@r12 的原始产物；每个工况取可行集里峰值过载的中位设计（不挑最好的）。
用法: python src/stage6_surrogate/fig_generate.py [--lang cn|en]"""
import os, json, argparse, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from matplotlib.patches import Circle, FancyBboxPatch
U = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ap = argparse.ArgumentParser(); ap.add_argument("--lang", default="cn", choices=["cn", "en"])
LANG = ap.parse_args().lang
OUT = f"{U}/outputs/ppt_{LANG}"; os.makedirs(OUT, exist_ok=True)
_have = {f.name for f in fm.fontManager.ttflist}
_cjk = next((f for f in ("Noto Sans CJK SC", "Noto Sans CJK JP", "WenQuanYi Zen Hei") if f in _have), None)
plt.rcParams.update({"font.sans-serif": ([_cjk] if (_cjk and LANG == "cn") else []) + ["DejaVu Sans"],
                     "font.family": "sans-serif", "axes.unicode_minus": False,
                     "figure.facecolor": "white", "savefig.facecolor": "white"})
INK, BONE, GRN, RED, GRY = "#111", "#20628c", "#2E7D5B", "#c0392b", "#666"
COLS = ["#2E7D5B", "#1b6ca8", "#7a4bbf"]
T = {"cn": dict(sup="给定工况，生成器直接给出一条腿 —— 每一条都经落震仿真验证",
                cond=["5 kg · 草地 · 1.2 m/s", "12 kg · 硬地 · 1.2 m/s", "30 kg · 湿沙 · 1.2 m/s"],
                inp="输入", outp="输出（9 维设计）", res="落震结果",
                seg="三段长度", stiff="关节刚度 κ（踝/膝/髋）", tau="松弛时间 τ", pose="触地姿态 θA / θK",
                peak="峰值过载", stroke="行程", mass="腿重", ok="通过  ≤10 g · ≤24 mm",
                scale="同一比例尺", foot="可行集 {k}/{n}，此处取峰值中位的那条",
                strip="全包线首发命中率：{a:.0f}–{b:.0f}%（19 个质量 × 6 种工况 × 2 种子）· "
                      "在真实水鸟的体重与工况下：生成腿 95% 通过，真鸟腿 71%"),
     "en": dict(sup="Given an operating point, the generator returns a leg — each verified by drop simulation",
                cond=["5 kg · turf · 1.2 m/s", "12 kg · rigid ground · 1.2 m/s", "30 kg · wet sand · 1.2 m/s"],
                inp="input", outp="output (9-dim design)", res="drop test",
                seg="segment lengths", stiff="joint stiffness κ (ankle/knee/hip)", tau="relaxation time τ",
                pose="touchdown posture θA / θK",
                peak="peak deceleration", stroke="stroke", mass="leg mass", ok="pass  ≤10 g · ≤24 mm",
                scale="common scale", foot="feasible {k}/{n}; median-peak design shown",
                strip="first-shot feasibility across the envelope: {a:.0f}–{b:.0f}% · "
                      "at real-waterbird conditions: generated legs pass 95%, real bird legs 71%")}[LANG]

A = json.load(open(f"{U}/outputs/v23_e20_r12/e20_bio.json")); MET = A["met"]
Z = np.load(f"{U}/outputs/v23_e20_r12/e20_bio_raw.npz", allow_pickle=True)
Z1 = np.load(f"{U}/outputs/v23_e20_r12_s1/e20_bio_raw.npz", allow_pickle=True)
m_all = Z["m_all"]
ig, ist, imass = MET.index("peak_g"), MET.index("leg_stroke_mm"), MET.index("leg_mass_g")
PICK = [("turf1.2", 16), ("concrete1.2", 17), ("wetsand1.2", 18)]
# 全包线命中率（两个种子）
feas = []
for ZZ in (Z, Z1):
    v = [ZZ[c + "__ok"].mean() for c in A["conds"] if c + "__ok" in ZZ and c != "concrete2.0"]
    feas.append(100 * np.mean(v))

def leg_pts(L1, r2, r3, thA, thK, a1=np.radians(50.)):
    a2 = a1 + np.radians(180 - thA); a3 = a2 - np.radians(180 - thK)
    P0 = np.array([0., 0.]); P1 = P0 + L1 * np.array([np.cos(a1), np.sin(a1)])
    P2 = P1 + r2 * L1 * np.array([np.cos(a2), np.sin(a2)]); P3 = P2 + r3 * L1 * np.array([np.cos(a3), np.sin(a3)])
    return P0, P1, P2, P3

fig = plt.figure(figsize=(16.0, 8.2))
gs = fig.add_gridspec(1, 3, wspace=.06, left=.02, right=.98, top=.86, bottom=.12)
YMAX = 0; designs = []
for (c, i) in PICK:
    X = Z[c + "__x7"][i]; M = Z[c + "__met"][i]; ok = Z[c + "__ok"][i]
    g = M[ok][:, ig]; j = np.argsort(g)[len(g) // 2]; idx = np.where(ok)[0][j]
    designs.append((X[idx], M[idx], int(ok.sum()), len(ok)))
    pts = leg_pts(X[idx][0], X[idx][1], X[idx][2], X[idx][7], X[idx][8])
    YMAX = max(YMAX, pts[3][1])
XW = 0.95 * YMAX
for k, ((c, i), (x, met, nok, ntot)) in enumerate(zip(PICK, designs)):
    ax = fig.add_subplot(gs[0, k]); col = COLS[k]
    ax.set_xlim(-XW * .62, XW * .62); ax.set_ylim(-YMAX * 0.92, YMAX * 1.16)
    ax.set_aspect("equal"); ax.axis("off")
    ax.axhline(0, color="#8a7f6d", lw=3, zorder=1)
    P0, P1, P2, P3 = leg_pts(x[0], x[1], x[2], x[7], x[8])
    xc = (P0[0] + P3[0]) / 2; sh = -xc
    for A_, B_ in ((P0, P1), (P1, P2), (P2, P3)):
        ax.plot([A_[0] + sh, B_[0] + sh], [A_[1], B_[1]], color=col, lw=9, solid_capstyle="round", zorder=3)
    for P in (P1, P2, P3):
        ax.add_patch(Circle((P[0] + sh, P[1]), YMAX * .018, fc="white", ec=INK, lw=1.4, zorder=5))
    ax.add_patch(Circle((P0[0] + sh, P0[1]), x[0] * .18, fc="#f1eee6", ec=INK, lw=1.3, zorder=4))
    ax.add_patch(FancyBboxPatch((P3[0] + sh - YMAX * .16, P3[1] + YMAX * .03), YMAX * .32, YMAX * .10,
                                boxstyle="round,pad=2", fc="#e6ecf6", ec=INK, lw=1.2, zorder=2))
    ax.text(P3[0] + sh, P3[1] + YMAX * .08, f"{m_all[i]:g} kg", ha="center", va="center", fontsize=12, fontweight="bold")
    # 标题：工况
    ax.set_title(T["cond"][k], fontsize=15.5, fontweight="bold", color=col, pad=14)
    # 输出向量
    L1, r2, r3 = x[0], x[1], x[2]
    box = dict(boxstyle="round,pad=.45", fc="white", ec=col, lw=1.5)
    ax.text(.03, .41,
            f"{T['outp']}\n"
            f"{T['seg']}: {L1:.0f} / {r2*L1:.0f} / {r3*L1:.0f} mm\n"
            f"{T['stiff']}: {x[3]:.1f} / {x[4]:.1f} / {x[5]:.1f}\n"
            f"{T['tau']}: {x[6]*1e3:.0f} ms\n"
            f"{T['pose']}: {x[7]:.0f}° / {x[8]:.0f}°",
            transform=ax.transAxes, va="top", ha="left", fontsize=11, linespacing=1.6, bbox=box, zorder=6)
    ax.text(.03, .17,
            f"{T['res']}   ✓ {T['ok']}\n"
            f"{T['peak']} {met[ig]:.1f} g  ·  {T['stroke']} {met[ist]:.1f} mm  ·  {T['mass']} {met[imass]:.0f} g",
            transform=ax.transAxes, va="top", ha="left", fontsize=11, linespacing=1.6,
            bbox=dict(boxstyle="round,pad=.45", fc="#eef7f2", ec=GRN, lw=1.5), color="#1a3d2b", zorder=6)
    ax.text(.5, .015, T["foot"].format(k=nok, n=ntot) + ("  ·  " + T["scale"] if k == 1 else ""),
            transform=ax.transAxes, ha="center", fontsize=10, color=GRY)
fig.suptitle(T["sup"], fontsize=17, fontweight="bold", y=.97)
fig.text(.5, .045, T["strip"].format(a=min(feas), b=max(feas)), ha="center", fontsize=12.5,
         fontweight="bold", color="#222",
         bbox=dict(boxstyle="round,pad=.5", fc="#f6f6f6", ec="#bbb"))
out = f"{OUT}/{LANG}_F_generate.png"
fig.savefig(out, dpi=165, bbox_inches="tight"); print("→", out, f"| 命中率 {feas[0]:.1f}% / {feas[1]:.1f}%")

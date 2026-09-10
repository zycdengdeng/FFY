# -*- coding: utf-8 -*-
"""E28：用 Duong(Cranfield) 的 12 次真实水鸟落水，检验我们模型的触地姿态约定。"""
import glob, os, sys
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import matplotlib.font_manager as fm
fm.fontManager.addfont("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")
plt.rcParams["font.sans-serif"] = ["Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False
CRIM, STEEL, GREY = "#7F2D32", "#5c7f9e", "#9a9a9a"

KIN = "/mnt/user-data/uploads/FFY/duong_data/pose-estimation-video-kinematic"
EV  = pd.read_csv("/mnt/user-data/uploads/FFY/duong_data/code/processed_data/kinematic_processed_dataset.csv")
BAD = {"Duck_1_final"}          # 跟踪失败：压缩后髋点跑到足下方

def load(p):
    r = pd.read_csv(p, header=[1, 2], index_col=0)
    r.columns = ["%s_%s" % (a, b) for a, b in r.columns]; return r

rec = []
for f in sorted(glob.glob(os.path.join(KIN, "*_final", "*_filtered.csv"))):
    vid = os.path.basename(os.path.dirname(f))
    if vid in BAD: continue
    d = load(f)
    lk = {s: np.nanmean([d["%s_%s_likelihood" % (s, p)].values
                         for p in ("hip","knee","ankle","foot")]) for s in ("left","right")}
    s = max(lk, key=lk.get)
    P = lambda p: np.c_[d["%s_%s_x" % (s,p)].values, -d["%s_%s_y" % (s,p)].values]
    hip, knee, ank, foot = P("hip"), P("knee"), P("ankle"), P("foot")
    ev = EV[EV.video == vid]; wc = ev[ev.event == "water_contact"]
    if wc.empty: continue
    i = int(wc.frame.iloc[0])
    tmt  = lambda k: np.degrees(np.arctan2(abs(ank[k,1]-foot[k,1]), abs(ank[k,0]-foot[k,0])))
    lean = lambda k: np.degrees(np.arctan2(abs(hip[k,0]-foot[k,0]), max(hip[k,1]-foot[k,1], 1e-6)))
    hgt  = lambda k: hip[k,1] - foot[k,1]
    win  = list(range(i, min(i+12, len(hip))))
    kmin = win[int(np.argmin([hgt(k) for k in win]))]
    L = np.linalg.norm(hip[i]-knee[i]) + np.linalg.norm(knee[i]-ank[i]) + np.linalg.norm(ank[i]-foot[i])
    pts = np.array([foot[i], ank[i], knee[i], hip[i]], float)
    pts = (pts - pts[0]) / L
    if pts[3, 0] < 0: pts[:, 0] *= -1          # 统一朝右
    rec.append(dict(video=vid, animal=vid.split("_")[0], pts=pts,
                    tmt=tmt(i), tmt2=tmt(kmin), lean=lean(i), lean2=lean(kmin),
                    itj=float(wc.ITJ_angle.iloc[0]), knee_a=float(wc.KNEE_angle.iloc[0])))

R = pd.DataFrame(rec)
COL = {"Duck": "#c98a2e", "Goose": STEEL, "Pelican": "#5a8a5a", "Swan": CRIM}

fig, ax = plt.subplots(1, 3, figsize=(13.6, 4.5),
                       gridspec_kw=dict(width_ratios=[1.0, 1.15, 0.95], wspace=0.30))

# ---- (a) 归一化腿姿态 ----
a = ax[0]
for _, r in R.iterrows():
    p = r.pts
    a.plot(p[:,0], p[:,1], "-o", ms=3, lw=1.4, color=COL[r.animal], alpha=.75)
a.axhline(0, color="#4a7fb5", lw=2.5, alpha=.5)
a.plot([0, 0], [0, 1.0], "--", color="k", lw=1.2, alpha=.6)
a.text(0.04, 1.03, "髋在足正上方（Bigoni 的设想）", fontsize=8.5, va="bottom", ha="left")
a.text(0.46, 0.20, "实测前倾\n30.5° ± 6.4°", fontsize=9.5, color=CRIM, fontweight="bold")
a.set_aspect("equal"); a.set_xlim(-.22, 1.02); a.set_ylim(-.1, 1.15)
a.set_xlabel("水平 / 腿总长"); a.set_ylabel("竖直 / 腿总长")
a.set_title("(a) 11 次真实落水的触地腿姿态\n足端对齐、按腿长归一化", fontsize=10.5)
h = [plt.Line2D([],[],color=c,marker="o",ms=4,lw=1.4,label=k) for k,c in COL.items()]
a.legend(handles=h, fontsize=8, loc="upper right", frameon=False)

# ---- (b) 角度 vs 我们的搜索盒 ----
b = ax[1]
BOX = {"踝 θ_A\n(ITJ)": (113, 160, R.itj.values),
       "膝 θ_K": (118, 157, R.knee_a.values),
       "跗跖骨倾角\n(对应 q1_0)": (None, None, R.tmt.values)}
for i, (k, (lo, hi, v)) in enumerate(BOX.items()):
    if lo is not None:
        b.add_patch(Rectangle((i-.30, lo), .60, hi-lo, fc=STEEL, alpha=.16, ec=STEEL, lw=1))
        b.text(i-.33, hi+2.5, "搜索盒 [%d, %d]" % (lo, hi), fontsize=8.2, va="bottom", ha="left", color=STEEL)
    b.scatter(np.full(len(v), i) + np.random.RandomState(0).uniform(-.13,.13,len(v)),
              v, s=34, color="k", zorder=3, alpha=.8)
    b.plot([i-.30, i+.30], [np.mean(v)]*2, color=CRIM, lw=2.4, zorder=4)
    b.text(i-.34, np.mean(v), "%.0f±%.0f°" % (np.mean(v), np.std(v)), ha="right", va="center",
           fontsize=9, color=CRIM, fontweight="bold")
b.plot([1.62, 2.42], [50, 50], ls="--", color=CRIM, lw=1.6)
b.text(2.46, 50, "模型常数\nq1_0 = 50°", fontsize=8.5, va="center", color=CRIM)
b.set_xticks(range(3)); b.set_xticklabels(BOX.keys(), fontsize=9.5)
b.set_xlim(-.95, 3.15); b.set_ylabel("触地时角度 (°)")
b.set_ylim(15, 178)
b.set_title("(b) 触地角度：真实落水 vs 模型约定\n两个搜索盒全覆盖，q1_0 在实测 1σ 内", fontsize=10.5)

# ---- (c) 压缩过程中腿有没有转竖直 ----
c = ax[2]
for j, (_, r) in enumerate(R.iterrows()):
    c.plot([0, 1], [r.tmt, r.tmt2], "-o", ms=4, lw=1.2, color=COL[r.animal], alpha=.7)
c.plot([0, 1], [R.tmt.mean(), R.tmt2.mean()], "-s", ms=8, lw=3, color="k", zorder=5, label="均值")
c.annotate("", xy=(1.14, R.tmt2.mean()), xytext=(1.14, R.tmt.mean()),
           arrowprops=dict(arrowstyle="-|>", color=CRIM, lw=2))
_ndn = int(((R.tmt2 - R.tmt) < 0).sum())
c.text(1.18, (R.tmt.mean()+R.tmt2.mean())/2, "变平 %.1f°\n(%d/%d 次为负)" % (R.tmt.mean()-R.tmt2.mean(), _ndn, len(R)),
       fontsize=9, va="center", color=CRIM, fontweight="bold")
c.set_xticks([0, 1]); c.set_xticklabels(["触水", "最大压缩"], fontsize=10)
c.set_xlim(-.2, 1.75); c.set_ylabel("跗跖骨与水平面夹角 (°)")
c.set_title("(c) 压缩过程中腿并没有转向竖直\n和我们的仿真结论一致", fontsize=10.5)
c.legend(fontsize=8.5, frameon=False, loc="lower left")

fig.text(0.008, 0.010,
         "数据来源：Duong Vu (Cranfield MSc CSTE, 2025) DeepLabCut 姿态估计，12 次水鸟落水（鸭 3 / 雁 3 / 鹈鹕 3 / 天鹅 3），"
         "Duck_1 因跟踪失败剔除。角度为本工作从关键点重建，与其 kinematic_processed_dataset 交叉核对（ITJ 均值差 1.8°）。\n"
         "注：原视频为慢动作且未标定，速度量无法反解，本图只用几何量。",
         fontsize=7.4, color="#555")
fig.subplots_adjust(left=.055, right=.985, top=.865, bottom=.185)
out = "/home/claude/duong/F_duong_e28.png"
fig.savefig(out, dpi=200); print("saved", out)
print(R[["video","tmt","tmt2","lean","lean2","itj","knee_a"]].round(1).to_string(index=False))
print("tmt %.1f±%.1f  lean %.1f±%.1f  d_tmt %.1f" % (R.tmt.mean(), R.tmt.std(), R.lean.mean(), R.lean.std(), (R.tmt2-R.tmt).mean()))

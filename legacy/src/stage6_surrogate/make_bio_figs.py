# -*- coding: utf-8 -*-
"""给夏老师的生物分析图组(4 张,纸面风格白底):
A 异速生长主图:214 种散点(按科着色)+ 合并拟合 + 三条假设斜率 + 先验带 + 五个 AI 设计
B 各科 u 地段图:回应"为什么不按科建先验"
C Watanabe 比例窄带 + 尺寸趋势检验(欠账 A2)
D 标度指数森林图:自测/分科/文献/理论/涌现 一图对齐
"""
import argparse, csv, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 9, "axes.linewidth": .7,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "font.sans-serif": ["WenQuanYi Zen Hei", "Noto Sans CJK SC", "DejaVu Sans"],
                     "axes.unicode_minus": False})
FAM_COL = {"Anatidae": "#3A6EA5", "Phalacrocoracidae": "#4E8D7C",
           "Podicipedidae": "#B0713F", "Gaviidae": "#7A5FA0", "Pelecanidae": "#C05B6B"}
FAM_CN = {"Anatidae": "雁鸭科", "Phalacrocoracidae": "鸬鹚科",
          "Podicipedidae": "䴙䴘科", "Gaviidae": "潜鸟科", "Pelecanidae": "鹈鹕科"}
CRIM = "#8E2A34"

rows = list(csv.DictReader(open("/tmp/bio/avonet_waterbirds.csv")))
seen, R = set(), []
for r in rows:
    if r["scientificNameStd"] in seen: continue
    seen.add(r["scientificNameStd"])
    R.append((r["scientificNameStd"], r["Family"], float(r["tarsus_mm"]), float(r["BodyMass.Value"])))
fam = np.array([r[1] for r in R]); L = np.array([r[2] for r in R]); M = np.array([r[3] for r in R])
x, y = np.log10(M), np.log10(L)
b, a = np.polyfit(x, y, 1)
resid = y - (a + b * x); sigma = float(np.std(resid, ddof=2))
n = len(R)
se_b = sigma / (np.std(x, ddof=1) * np.sqrt(n - 1)); ci = 1.96 * se_b
r2f = float(np.corrcoef(x, y)[0, 1] ** 2)
print(f"合并拟合: b={b:.3f}±{ci:.3f}  a={a:.3f}  σ={sigma:.4f}  r²={r2f:.3f}  n={n}")

ap = argparse.ArgumentParser()
ap.add_argument("--no-ai", action="store_true", help="A 图不画 AI 设计的星")
ap.add_argument("--no-abl", action="store_true",
                help="A 图不画几何/弹性相似两条参考线(不讲消融时用)")
ap.add_argument("--out", default="/tmp/bio", help="输出目录")
ARGS, _ = ap.parse_known_args()
designs = [] if ARGS.no_ai else json.load(open("/tmp/designs.json"))

# ============ A 异速生长主图 ============
fig, ax = plt.subplots(figsize=(7.4, 5.0))
for f in FAM_COL:
    s = fam == f
    ax.scatter(M[s]/1000, L[s], s=16, color=FAM_COL[f], alpha=.75, lw=0,
               label=f"{FAM_CN[f]} ({s.sum()})")
mg = np.linspace(np.log10(90), np.log10(12000), 50)
ax.plot(10**mg/1000, 10**(a + b*mg), "-", color=CRIM, lw=2.2,
        label=f"合并拟合 b={b:.3f}±{ci:.3f}")
ax.fill_between(10**mg/1000, 10**(a + b*mg - 2.5*sigma), 10**(a + b*mg + 2.5*sigma),
                color=CRIM, alpha=.08, lw=0, label="条件先验带 ±2.5σ")
mref = np.log10(3464)
lref = a + b*mref
if not ARGS.no_abl:
    for bh, ls, lab in [(1/3, "--", "几何相似 b=1/3"), (0.25, ":", "弹性相似 b=1/4")]:
        ax.plot(10**mg/1000, 10**(lref + bh*(mg-mref)), ls, color="#555", lw=1.3,
                label=lab)
if designs:
    dm = np.array([d["m"] for d in designs]); dl = np.array([d["L_mm"][0] for d in designs])
    ax.plot(dm, dl, "*", ms=17, mfc="#F0A030", mec="k", mew=.8, zorder=5,
            label="AI 生成设计(实测可行)")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks([0.1,0.3,1,3,10]); ax.set_xticklabels(["0.1","0.3","1","3","10"])
ax.set_yticks([20,40,80,120]); ax.set_yticklabels(["20","40","80","120"])
ax.set_xlabel("体重 (kg)"); ax.set_ylabel("跗跖骨长 L₁ (mm)")
ax.set_title("水鸟腿长异速生长律(AVONET 213 种)与 AI 设计的落点" if designs
             else f"水鸟腿长异速生长律:{n} 种水鸟实测与条件先验带",
             loc="left", fontsize=11)
ax.legend(frameon=False, fontsize=7.6, loc="upper left",
          ncol=2 if (designs or not ARGS.no_abl) else 1)
fig.tight_layout(); fig.savefig(f"{ARGS.out}/figA_allometry.png", dpi=250, bbox_inches="tight")
plt.close(fig)

# ============ B 各科 u 地段图 ============
u = resid / sigma
fig, ax = plt.subplots(figsize=(7.0, 3.6))
order = ["Gaviidae", "Phalacrocoracidae", "Anatidae", "Podicipedidae", "Pelecanidae"]
stats = []
for i, f in enumerate(order):
    s = fam == f
    uf = u[s]
    stats.append((f, float(np.median(uf)), float(np.mean(uf)), int(s.sum())))
    ax.scatter(uf, np.full(uf.sum() if False else len(uf), i) + np.random.default_rng(i).uniform(-.14,.14,len(uf)),
               s=15, color=FAM_COL[f], alpha=.75, lw=0)
    ax.plot([np.median(uf)]*2, [i-.28, i+.28], color=FAM_COL[f], lw=2.6)
ax.axvline(0, color="#999", lw=.8, ls="--")
ax.set_yticks(range(len(order)))
ax.set_yticklabels([f"{FAM_CN[f]} (n={int((fam==f).sum())})" for f in order])
ax.set_xlabel("u = 相对同体重典型腿长的偏离(σ 单位)")
ax.set_title("各科在 u 轴上的地段:科身份 ≈ 连续坐标上的一段(竖线=科中位)", loc="left", fontsize=10.5)
ax.set_xlim(-3.4, 3.4)
fig.tight_layout(); fig.savefig("/tmp/bio/figB_family_u.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print("各科 u 中位:", {f: round(md,2) for f,md,_,_ in stats})

# ============ C Watanabe 比例 + 尺寸趋势检验 ============
w = list(csv.DictReader(open("/tmp/pipeline_code/data/skeletal/watanabe2017_anatidae.csv")))
wv = [r for r in w if r["group"] == "Volant"]
r2a = np.array([float(r["r2"]) for r in wv]); r3a = np.array([float(r["r3"]) for r in wv])
tmt = np.array([float(r["tmt_mm"]) for r in wv]); lt = np.log10(tmt)
tib = np.array([float(r["tib_mm"]) for r in wv]); fem = np.array([float(r["fem_mm"]) for r in wv])

# (b) 用**原始段长**做 log-log 回归,不用比值。
# 比值 r=L_seg/L1 与 L1 共用分母,回归会产生伪负相关(Pearson 1897);
# 直接回归 log L_seg ~ log L1 没有这个问题,而且天然有参照点:几何相似 = 1.0。
out = {}
for nm, arr in (("L2", tib), ("L3", fem)):
    pw, ic = np.polyfit(lt, np.log10(arr), 1)
    res = np.log10(arr) - (ic + pw*lt)
    se = np.std(res, ddof=2) / (np.std(lt, ddof=0)*np.sqrt(len(arr)))
    out[nm] = (pw, ic, 1.96*se, abs((pw-1.0)/se))

fig, axs = plt.subplots(1, 2, figsize=(9.8, 3.9), width_ratios=[1, 1.12])
ax = axs[0]
ax.scatter(r2a, r3a, s=18, color="#3A6EA5", alpha=.8, lw=0, label=f"Watanabe 2017 会飞种 (n={len(wv)})")
ax.add_patch(plt.Rectangle((1.49,0.84), 2.09-1.49, 1.28-0.84, fill=False,
             ec=CRIM, lw=1.8, label="设计盒(实测全距)"))
ax.plot(1.764, 0.951, "D", ms=9, mfc="#F0A030", mec="k", label="天鹅(课题组前期蓝本)")
ax.set_xlabel("r₂ = 胫跗骨 / 跗跖骨"); ax.set_ylabel("r₃ = 股骨 / 跗跖骨")
ax.set_title("(a) 腿骨比例的生物窄带", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=7.6, loc="lower right")
ax.set_xlim(1.38, 2.22); ax.set_ylim(0.72, 1.42)

ax = axs[1]
xs = np.linspace(lt.min(), lt.max(), 20)
for nm, arr, c, cn in (("L2", tib, "#3A6EA5", "L₂ 胫跗骨"), ("L3", fem, "#4E8D7C", "L₃ 股骨")):
    pw, ic, ci_, tv = out[nm]
    ax.scatter(tmt, arr, s=16, color=c, alpha=.75, lw=0,
               label=f"{cn}   p = {pw:.3f} ± {ci_:.3f}")
    ax.plot(10**xs, 10**(ic + pw*xs), "-", color=c, lw=1.8, zorder=4)
    ic1 = np.mean(np.log10(arr)) - 1.0*np.mean(lt)          # 同质心的 p=1 参照
    ax.plot(10**xs, 10**(ic1 + 1.0*xs), "--", color="#999", lw=1.2, zorder=3)
ax.plot([], [], "--", color="#999", lw=1.2, label="几何相似 p = 1.0(形状不变)")
ax.set_xscale("log"); ax.set_yscale("log")
from matplotlib.ticker import NullFormatter, NullLocator
ax.set_xticks([30,60,120]); ax.set_xticklabels(["30","60","120"])
ax.set_yticks([30,60,120,240]); ax.set_yticklabels(["30","60","120","240"])
ax.xaxis.set_minor_locator(NullLocator()); ax.xaxis.set_minor_formatter(NullFormatter())
ax.yaxis.set_minor_locator(NullLocator()); ax.yaxis.set_minor_formatter(NullFormatter())
ax.set_xlabel("跗跖骨长 L₁ (mm)"); ax.set_ylabel("该段骨长 (mm)")
ax.set_title("(b) 分段骨长的相对生长:$L_{seg} \\propto L_1^{\\,p}$", loc="left", fontsize=10.5)
ax.legend(frameon=False, fontsize=7.8, loc="upper left")
ax.text(.98, .04, f"两段均显著低于 1.0(|t| = {out['L2'][3]:.1f} / {out['L3'][3]:.1f},n = {len(wv)})\n"
        "→ 跗跖骨每长 1 倍,胫跗骨只长 %.2f 倍、股骨只长 %.2f 倍" % (out["L2"][0], out["L3"][0]),
        transform=ax.transAxes, fontsize=7.8, color="#555", ha="right", va="bottom",
        linespacing=1.6)
fig.tight_layout(); fig.savefig(f"{ARGS.out}/figC_ratios.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print(f"相对生长: L2 p={out['L2'][0]:.3f}±{out['L2'][2]:.3f} |t|={out['L2'][3]:.1f}  "
      f"L3 p={out['L3'][0]:.3f}±{out['L3'][2]:.3f} |t|={out['L3'][3]:.1f}")

# ============ D 标度指数森林图 ============
fits = json.load(open("/tmp/pipeline_code/outputs/bird_pareto/avonet_allometry.json"))["fits"]
items = [
    ("本工作 · 214 种水鸟", b, ci, CRIM, True),
    ("  雁鸭科 (152)", fits["Anatidae"]["b"], fits["Anatidae"]["ci95"], "#3A6EA5", False),
    ("  䴙䴘科 (19)", fits["Podicipedidae"]["b"], fits["Podicipedidae"]["ci95"], "#B0713F", False),
    ("  鸬鹚科 (30)", fits["Phalacrocoracidae"]["b"], fits["Phalacrocoracidae"]["ci95"], "#4E8D7C", False),
    ("  潜鸟科 (5)", fits["Gaviidae"]["b"], fits["Gaviidae"]["ci95"], "#7A5FA0", False),
    ("全部鸟类 9198 种 (AVONET)", fits["all_birds"]["b"], fits["all_birds"]["ci95"], "#555", False),
    ("文献:后肢长度指数(Kilbourne 2014 引/复核)", 0.39, 0.02, "#555", False),
    ("理论:几何相似", 1/3, 0.0, "#999", False),
    ("理论:弹性相似(McMahon)", 0.25, 0.0, "#999", False),
    ("涌现:AI 训练后实际执行(bio 臂,2 种子)", 0.249, 0.005, "#F0A030", True),
]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
for i, (nm, bv, cv, c, bold) in enumerate(items):
    yy = len(items) - 1 - i
    if cv > 0:
        ax.plot([bv-cv, bv+cv], [yy, yy], "-", color=c, lw=2.2)
    ax.plot(bv, yy, "o" if not bold else "D", ms=7 if not bold else 9, color=c,
            mec="k" if bold else c, mew=.8 if bold else 0)
    ax.text(-0.02, yy, nm, ha="right", va="center", fontsize=8.4,
            fontweight="bold" if bold else "normal", color=c)
ax.axvline(b, color=CRIM, lw=.8, ls="--", alpha=.5)
ax.axvline(0.249, color="#F0A030", lw=.8, ls="--", alpha=.6)
ax.set_xlim(-0.02, 0.52); ax.set_ylim(-0.7, len(items)-0.3)
ax.set_yticks([]); ax.set_xlabel("腿长-体重标度指数 b")
ax.set_title("标度指数全景:实测(按科)· 文献 · 理论 · AI 涌现值", loc="left", fontsize=11)
ax.spines["left"].set_visible(False)
fig.tight_layout(); fig.savefig("/tmp/bio/figD_forest.png", dpi=250, bbox_inches="tight")
plt.close(fig)
print("四张图完成")

# -*- coding: utf-8 -*-
"""飞行感知的腿长标度律 —— 论文 A 的控制组 v3。

## 为什么要有这一版(2026-09-03)

v2 按"着陆基底"分组,得到的最陡的类群是**鸵鸟(b=0.502)**。但鸵鸟**不会飞**,
它的腿从来不需要接住从空中来的冲击 —— 放进"着陆"的统计里是错的,而且它在往上拉斜率。

更深的一点:与起落架真正同构的生物条件不是"你在什么地面上活动",而是
**"你会不会飞,以及你以多快的速度从空中回到地面"**。所以本版做两件事:

  1. **剔除不会飞的**(古颚类、企鹅、以及水鸟里的船鸭/无翼鸬鹚/短翅䴙䴘等)
     —— 这与项目既有做法一致:r₂/r₃ 窄带取自 Watanabe 2017 的 **91 种会飞**雁鸭科,
     但 b 的拟合此前用了全部水鸟,**口径本来就不一致**,本版一并修正。
  2. **把飞行能力做成协变量**,而不只是筛子:
       · HWI  手翼指数 = Kipp距离/翼长 —— 飞行效率/扩散能力的标准代理(AVONET)
       · WL*  翼载荷代理 = m / 翼长²  —— **与着陆速度直接相关**:翼载荷越高失速越快,
              回到地面的速度越大 → 与我们的工况变量 v₀ 同构。这是本分析最贴任务的量。

## 判据

  · 若剔除不会飞的之后,各类群 b 仍与水鸟无显著差异 → "着陆基底不驱动腿长标度"更稳
  · 若 u 与翼载荷显著相关 → **落得快的鸟腿更长/更短**,这是可直接接进设计的生物学结论

## 诚实边界

  · 不会飞的名单是**人工整理**的(按目 + 已知属),不是数据库字段。可能漏掉个别物种;
    名单见 FLIGHTLESS_*,论文里须作为附录列出。
  · 翼载荷代理 m/翼长² 不是真实翼载荷(缺翼面积),只在类群间做**相对**比较。
  · 系统发育非独立性仍未解决(见 v2 的 L4 说明)。
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

A_W, B_W, SIG_W = 0.479, 0.391, 0.0784
WATER_FAMILIES = ["Anatidae", "Gaviidae", "Podicipedidae", "Pelecanidae", "Phalacrocoracidae"]
REF_SLOPES = {"着陆物理 b_eff (v2.2)": 0.238, "弹性相似 McMahon": 0.250,
              "着陆物理 b_eff (v2.1)": 0.281, "几何相似": 1/3, "水鸟(旧口径)": 0.391}

# --- 不会飞的类群(人工整理,论文附录须列出) ---
FLIGHTLESS_ORDERS = {"Struthioniformes", "Rheiformes", "Casuariiformes",
                     "Apterygiformes", "Sphenisciformes"}   # 古颚类 + 企鹅
# 水鸟及其它类群里已知无飞行能力的属(部分属内种间有差异,此处从严整属剔除)
FLIGHTLESS_GENERA = {
    "Tachyeres",      # 船鸭:4 种中 3 种不能飞
    "Nannopterum",    # 加岛无翼鸬鹚(旧置于 Phalacrocorax)
    "Rollandia",      # 短翅䴙䴘(R. microptera 不能飞)
    "Centropelma",    # 同上异名
    "Podilymbus",     # 阿蒂特兰䴙䴘(已灭绝种不能飞)
    "Cnemiornis", "Chendytes",     # 化石类群,通常不在库中
}
FLIGHTLESS_SPECIES = {
    "Anas aucklandica", "Anas nesiotis", "Anas chlorotis",   # 新西兰各岛短翅鸭
    "Podiceps taczanowskii",                                  # 秘鲁短翅䴙䴘
    "Phalacrocorax harrisi", "Nannopterum harrisi",
    "Tachyeres pteneres", "Tachyeres brachypterus", "Tachyeres leucocephalus",
}

CLADES = {
    "水鸟(5 科)":            ("Family", set(WATER_FAMILIES),  "水面"),
    "雁鸭科":                ("Family", {"Anatidae"},         "水面+硬地"),
    "鸡形目":                ("Order",  {"Galliformes"},      "地面"),
    "  ├ 雉科":              ("Family", {"Phasianidae"},      "地面"),
    "  └ 凤冠雉科":           ("Family", {"Cracidae"},         "树栖+地面"),
    "鹰形目":                ("Order",  {"Accipitriformes"},  "抓握猎物"),
    "隼形目":                ("Order",  {"Falconiformes"},    "岩壁/树枝"),
    "鸮形目":                ("Order",  {"Strigiformes"},     "抓握猎物"),
    "鸽形目":                ("Order",  {"Columbiformes"},    "硬地/树枝"),
    "鹤形目":                ("Order",  {"Gruiformes"},       "涉水/地面"),
    "鸻形目":                ("Order",  {"Charadriiformes"},  "滩涂/水面"),
    "鹭科":                  ("Family", {"Ardeidae"},         "涉水"),
    "雀形目":                ("Order",  {"Passeriformes"},    "树枝"),
    "鹦形目":                ("Order",  {"Psittaciformes"},   "树枝"),
}


def ols(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    n = len(x); X = np.column_stack([np.ones(n), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    s2 = float(res @ res) / (n - 2)
    se = np.sqrt(s2 * np.linalg.inv(X.T @ X)[1, 1])
    sst = float(((y - y.mean()) ** 2).sum())
    return dict(a=float(beta[0]), b=float(beta[1]), ci95=float(1.96 * se),
                r2=float(1 - float(res @ res) / sst) if sst > 0 else np.nan, n=int(n))


def build(a):
    """从 .rda 建**含翅膀形态**的物种级表(旧缓存只有腿长,不够用)。"""
    import pyreadr, pandas as pd
    if os.path.exists(a.cache) and not a.refresh:
        print(f"[数据] 读缓存 {a.cache}")
        return pd.read_csv(a.cache)
    av = pyreadr.read_r(a.avonet)["avonet"]
    el = pyreadr.read_r(a.elton)["elton_birds"]
    keep = ["Tarsus.Length", "Wing.Length", "Kipps.Distance", "Hand.wing.Index"]
    av = av.dropna(subset=["Tarsus.Length", "scientificNameStd"])
    g = av.groupby("scientificNameStd")
    sp = g[keep].median()
    sp["n_spec"] = g.size()
    sp = sp[sp["n_spec"] >= a.min_specimens]
    fs = [c for c in el.columns if c.startswith("ForStrat.") and c.split(".")[1][0].islower()]
    el2 = (el.dropna(subset=["BodyMass.Value", "scientificNameStd"])
             .set_index("scientificNameStd")[["BodyMass.Value", "Family", "Order"] + fs])
    j = sp.join(el2, how="inner").dropna(subset=["BodyMass.Value", "Family", "Order"]).reset_index()
    j = j[j["BodyMass.Value"] > 0].copy()
    # EltonTraits 同名多条记录会让 join 膨胀(118 名 / 433 行,内容完全相同)→ 去重
    j = j.drop_duplicates("scientificNameStd")
    os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
    j.to_csv(a.cache, index=False)
    print(f"[数据] 已写缓存 {a.cache}({len(j)} 种,含翅膀形态)")
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avonet", default="/tmp/traitdata/data/avonet.rda")
    ap.add_argument("--elton", default="/tmp/traitdata/data/elton_birds.rda")
    ap.add_argument("--cache", default="data/avonet_joined_flight.csv")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="outputs/bird_pareto")
    ap.add_argument("--min_specimens", type=int, default=2)
    ap.add_argument("--min_species", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import pandas as pd

    j = build(a)
    gen = j["scientificNameStd"].str.split().str[0]
    j["flightless"] = (j["Order"].isin(FLIGHTLESS_ORDERS)
                       | gen.isin(FLIGHTLESS_GENERA)
                       | j["scientificNameStd"].isin(FLIGHTLESS_SPECIES))
    j["log_m"] = np.log10(j["BodyMass.Value"])
    j["log_L"] = np.log10(j["Tarsus.Length"])
    j["u"] = (j["log_L"] - (A_W + B_W * j["log_m"])) / SIG_W
    # 翼载荷代理:m / 翼长^2(缺翼面积,只作类群间相对比较)
    j["WL"] = j["BodyMass.Value"] / (j["Wing.Length"] ** 2)
    j["logWL"] = np.log10(j["WL"])

    print(f"\n联接 {len(j)} 种;标记为不会飞 {int(j['flightless'].sum())} 种")
    fl = j[j["flightless"]]
    print("  按目:", fl["Order"].value_counts().to_dict())
    flw = fl[fl["Family"].isin(WATER_FAMILIES)]
    print(f"  其中在水鸟 5 科内: {len(flw)} 种 → {sorted(flw['scientificNameStd'].tolist())}")

    V = j[~j["flightless"]].copy()      # volant only
    OUT = {"prior": dict(a=A_W, b=B_W, sigma=SIG_W), "ref": REF_SLOPES,
           "n_total": int(len(j)), "n_flightless": int(j["flightless"].sum())}

    # ---------- ① 剔除不会飞的,对 b 有多大影响 ----------
    print(f"\n{'='*94}\n① 剔除不会飞的物种前后对比\n{'='*94}")
    print(f"{'类群':<16}{'含不会飞 b':>14}{'仅会飞 b':>12}{'Δb':>9}{'剔除数':>8}")
    rows1 = []
    for name, (field, vals, _s) in CLADES.items():
        s_all = j[j[field].isin(vals)]; s_v = V[V[field].isin(vals)]
        if len(s_v) < a.min_species: continue
        f_all = ols(s_all["log_m"], s_all["log_L"]); f_v = ols(s_v["log_m"], s_v["log_L"])
        nrm = len(s_all) - len(s_v)
        print(f"{name:<16}{f_all['b']:>14.3f}{f_v['b']:>12.3f}{f_v['b']-f_all['b']:>+9.3f}{nrm:>8}")
        rows1.append(dict(clade=name, b_all=f_all["b"], b_volant=f_v["b"], n_removed=int(nrm)))
    fa = ols(j["log_m"], j["log_L"]); fv = ols(V["log_m"], V["log_L"])
    print(f"{'全部鸟类':<16}{fa['b']:>14.3f}{fv['b']:>12.3f}{fv['b']-fa['b']:>+9.3f}"
          f"{len(j)-len(V):>8}")
    OUT["step1_flightless_effect"] = rows1

    # ---------- ② 只用会飞的重排 ----------
    print(f"\n{'='*94}\n② 仅会飞物种的标度律(按 b 升序)\n{'='*94}")
    print(f"{'类群':<16}{'n':>5}{'b':>8}{'±95CI':>8}{'R²':>6}{'跨度dex':>9}{'u中位':>8}"
          f"{'翼载荷中位':>11}   基底")
    rows2 = []
    for name, (field, vals, sub_) in CLADES.items():
        s = V[V[field].isin(vals)]
        if len(s) < a.min_species: continue
        f = ols(s["log_m"], s["log_L"])
        rows2.append(dict(name=name, **f, u_med=float(s["u"].median()),
                          span=float(s["log_m"].max() - s["log_m"].min()),
                          wl_med=float(s["WL"].median()), substrate=sub_))
    rows2.append(dict(name="全部会飞鸟类", **fv, u_med=float(V["u"].median()),
                      span=float(V["log_m"].max() - V["log_m"].min()),
                      wl_med=float(V["WL"].median()), substrate="—"))
    rows2.sort(key=lambda r: r["b"])
    for r in rows2:
        w = "" if r["span"] >= 1.0 else " ⚠"
        print(f"{r['name']:<16}{r['n']:>5}{r['b']:>8.3f}{r['ci95']:>8.3f}{r['r2']:>6.2f}"
              f"{r['span']:>9.2f}{r['u_med']:>+8.2f}{r['wl_med']:>11.3f}   {r['substrate']}{w}")
    OUT["step2_volant_clades"] = rows2

    # ---------- ③ 飞行能力作协变量 ----------
    print(f"\n{'='*94}\n③ 飞行能力与腿长的关系(仅会飞物种)\n{'='*94}")
    for lab, col, expl in (("手翼指数 HWI", "Hand.wing.Index", "飞行效率/扩散能力"),
                           ("翼载荷代理 log(m/翼长²)", "logWL", "与着陆速度同向")):
        s = V.dropna(subset=[col])
        if len(s) < 50: print(f"  {lab}: 有效样本不足"); continue
        # u ~ x(u 已按体重归一,故这是"控制体重后"的关系)
        X = np.column_stack([np.ones(len(s)), s[col].values])
        beta, *_ = np.linalg.lstsq(X, s["u"].values, rcond=None)
        res = s["u"].values - X @ beta
        se = np.sqrt(float(res @ res) / (len(s) - 2) * np.linalg.inv(X.T @ X)[1, 1])
        r = float(np.corrcoef(s[col], s["u"])[0, 1])
        print(f"  u ~ {lab:<26} 斜率 {beta[1]:+.3f} ± {1.96*se:.3f}   r = {r:+.3f}   n = {len(s)}   ({expl})")
        OUT[f"step3_{col}"] = dict(slope=float(beta[1]), ci95=float(1.96*se), r=r, n=int(len(s)))
    # 按翼载荷分档看 b
    s = V.dropna(subset=["logWL"])
    qs = s["logWL"].quantile([0, .25, .5, .75, 1.0]).values
    print(f"\n  按翼载荷四分位分档(每档内重新拟合 b):")
    print(f"  {'档':<22}{'n':>6}{'b':>8}{'±95CI':>8}{'u中位':>8}")
    wl_rows = []
    for i, (lo, hi) in enumerate(zip(qs[:-1], qs[1:])):
        sub = s[(s["logWL"] >= lo) & (s["logWL"] <= hi if i == 3 else s["logWL"] < hi)]
        if len(sub) < 50: continue
        f = ols(sub["log_m"], sub["log_L"])
        lab = f"Q{i+1} WL {10**lo:.2f}–{10**hi:.2f}"
        print(f"  {lab:<22}{f['n']:>6}{f['b']:>8.3f}{f['ci95']:>8.3f}{sub['u'].median():>+8.2f}")
        wl_rows.append(dict(q=i+1, lo=float(10**lo), hi=float(10**hi), **f,
                            u_med=float(sub["u"].median())))
    OUT["step3_wl_quartiles"] = wl_rows

    print(f"\n参考斜率: " + "  ".join(f"{k}={v:.3f}" for k, v in sorted(REF_SLOPES.items(), key=lambda t: t[1])))
    p = os.path.join(a.out, "allometry_flight.json")
    json.dump(OUT, open(p, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n→ {p}")


if __name__ == "__main__":
    main()

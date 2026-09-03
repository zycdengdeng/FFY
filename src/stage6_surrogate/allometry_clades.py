# -*- coding: utf-8 -*-
"""跨类群 / 跨着陆基底的腿长标度律 —— 论文 A 的正面控制组。

## 问题

我们用**水鸟**提取先验,设计的却是**硬地**起落架。E21 已表明真鸟腿是为着水优化的
(水面允许更长的减速距离);E17 给出着陆物理偏好的涌现指数 b_eff ≈ 0.24。
而水鸟的 b = 0.391 站在全鸟类(0.312)的长腿一侧。

**那么:在硬基底着陆的鸟,腿的标度律更接近哪个?**

  · 若接近 0.24–0.31 且 u<0(同体重腿更短) → 着陆物理偏好的指数在自然界被实现了,
    水鸟是"为水面超配"的特例 —— 论文 A 闭合。
  · 若不接近 → 鸟类腿长标度律不主要由着陆驱动 —— 同样是可写的结论,而且更谨慎。

## 本脚本做四层分析(逐层更硬)

  L1 分类群:目/科水平的 b 与 u(鸡形目额外拆到科:雉/松鸡/珠鸡/凤冠雉)
  L2 分**觅食基底**(EltonTraits ForStrat.*):比分类群更机制化 —— 直接问
     "你在什么基底上活动"而不是"你姓什么";并对 u 做 ground% 的连续回归
  L3 **体重匹配**:只在共同体重区间内比,排除"b 差异其实是体重跨度差异"的伪影
  L4 **科级聚合**:每科取中位数再拟合,粗略控制系统发育非独立性(见下方诚实说明)

## 诚实边界(必须写进论文)

  · **系统发育非独立性**:物种不是独立样本,近缘种共享祖先性状。严格做法是 PGLS
    (需 Jetz et al. 2012 鸟类系统树)。本脚本的 L4 科级聚合只是**粗略**的部分控制,
    不能替代 PGLS —— 论文里必须这么写,或者补做 PGLS。
  · **觅食基底 ≠ 着陆基底**:ForStrat 描述的是觅食时所处层位,与着陆基底高度相关
    但不等同(如猛禽在空中觅食、在树枝着陆)。这是本分析最大的代理误差。
  · 腿长受多功能驱动(涉水深度、潜水杠杆、奔跑、栖握、体温调节),着陆只是其一。

## 用法

  # 数据(traitdata R 包;首次运行后会自动缓存,以后不再需要 .rda)
  git clone --depth 1 https://github.com/EcologicalTraitData/traitdata /tmp/traitdata
  python src/stage6_surrogate/allometry_clades.py --out outputs/bird_pareto

  # 缓存建好后(data/avonet_joined_all.csv),任何机器上直接:
  python src/stage6_surrogate/allometry_clades.py --out outputs/bird_pareto
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from allometry_data import ols_loglog, WATER_FAMILIES

# 项目既定的水鸟先验(与 bioprior.py 同源);u 以它为零点,可与既有五科直接并列
A_W, B_W, SIG_W = 0.479, 0.391, 0.0784
REF_SLOPES = {
    "着陆物理涌现 b_eff (v2.2)": 0.238,
    "弹性相似 (McMahon 1973)": 0.250,
    "着陆物理涌现 b_eff (v2.1)": 0.281,
    "几何相似": 1 / 3,
    "水鸟实测 (本项目先验)": 0.391,
}

CLADES = {
    "水鸟(既有 5 科)":        ("Family", set(WATER_FAMILIES),      "水面"),
    "雁鸭科 Anatidae":        ("Family", {"Anatidae"},             "水面+硬地"),
    "鸡形目 Galliformes":     ("Order",  {"Galliformes"},          "地面(陆栖腿力学模型生物)"),
    "  ├ 雉科 Phasianidae":   ("Family", {"Phasianidae"},          "地面"),
    "  ├ 珠鸡科 Numididae":   ("Family", {"Numididae"},            "地面"),
    "  └ 凤冠雉科 Cracidae":  ("Family", {"Cracidae"},             "树栖+地面"),
    "鹰形目 Accipitriformes": ("Order",  {"Accipitriformes"},      "树枝/地面,抓握"),
    "隼形目 Falconiformes":   ("Order",  {"Falconiformes"},        "岩壁/树枝"),
    "鸮形目 Strigiformes":    ("Order",  {"Strigiformes"},         "树枝/地面,抓握"),
    "鸽形目 Columbiformes":   ("Order",  {"Columbiformes"},        "硬地/树枝"),
    "鹤形目 Gruiformes":      ("Order",  {"Gruiformes"},           "涉水/地面"),
    "鸻形目 Charadriiformes": ("Order",  {"Charadriiformes"},      "滩涂/水面"),
    "鹭科 Ardeidae":          ("Family", {"Ardeidae"},             "涉水(长腿极端)"),
    "雀形目 Passeriformes":   ("Order",  {"Passeriformes"},        "树枝栖握"),
    "鹦形目 Psittaciformes":  ("Order",  {"Psittaciformes"},       "树枝攀握"),
    "鸵鸟等古颚类":            ("Order",  {"Struthioniformes", "Rheiformes",
                                          "Casuariiformes", "Apterygiformes"}, "地面奔跑"),
}
RAPTOR_ORDERS = {"Accipitriformes", "Falconiformes", "Strigiformes"}
CACHE_DEFAULT = "data/avonet_joined_all.csv"


def slope_diff_z(f1, f2):
    """两条 OLS 斜率之差的 z 检验(CI95 → SE)。返回 (Δb, z, p 近似)。"""
    se1, se2 = f1["ci95"] / 1.96, f2["ci95"] / 1.96
    d = f1["b"] - f2["b"]
    se = np.hypot(se1, se2)
    z = d / se if se > 0 else np.nan
    from math import erfc
    p = erfc(abs(z) / np.sqrt(2)) if np.isfinite(z) else np.nan
    return d, z, p


def load_joined(a):
    """优先读缓存;没有则读 .rda 并写缓存(以后任何机器都不再需要 traitdata)。"""
    import pandas as pd
    if os.path.exists(a.cache) and not a.refresh:
        j = pd.read_csv(a.cache)
        print(f"[数据] 读缓存 {a.cache}  ({len(j)} 种)")
        return j
    for p, nm in ((a.avonet, "AVONET"), (a.elton, "EltonTraits")):
        if not os.path.exists(p):
            sys.exit(f"[数据] 找不到 {nm}: {p}\n"
                     f"  缓存也不存在({a.cache})。请先取数据:\n"
                     f"    git clone --depth 1 https://github.com/EcologicalTraitData/traitdata /tmp/traitdata\n"
                     f"  然后重跑本脚本(会自动建缓存,以后不再需要)。\n"
                     f"  若 .rda 文件名不同,用 --avonet/--elton 指定;"
                     f"  ls /tmp/traitdata/data/ 看实际文件名。")
    import pyreadr
    av = pyreadr.read_r(a.avonet); av = av[list(av.keys())[0]]
    el = pyreadr.read_r(a.elton);  el = el[list(el.keys())[0]]
    print(f"[数据] AVONET 列: {[c for c in av.columns][:8]} …")
    print(f"[数据] Elton  列: {[c for c in el.columns][:12]} …")
    av = av.dropna(subset=["Tarsus.Length", "scientificNameStd"])
    g = av.groupby("scientificNameStd")["Tarsus.Length"]
    sp = g.median().to_frame("tarsus_mm"); sp["n_spec"] = g.size()
    sp = sp[sp["n_spec"] >= a.min_specimens]
    # 觅食层位(有就带上,没有就算了 —— 不同版本列名可能不同)
    forstrat = [c for c in el.columns if c.lower().startswith("forstrat")]
    keep = ["BodyMass.Value", "Family", "Order"] + forstrat
    keep = [c for c in keep if c in el.columns]
    el2 = el.dropna(subset=["BodyMass.Value", "scientificNameStd"]).set_index("scientificNameStd")[keep]
    j = sp.join(el2, how="inner").dropna(subset=["BodyMass.Value", "Family", "Order"]).reset_index()
    j = j[j["BodyMass.Value"] > 0].copy()
    os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
    j.to_csv(a.cache, index=False)
    print(f"[数据] 已写缓存 {a.cache}  ({len(j)} 种,含 {len(forstrat)} 个 ForStrat 列)")
    return j


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avonet", default="/tmp/traitdata/data/avonet.rda")
    ap.add_argument("--elton", default="/tmp/traitdata/data/elton_birds.rda")
    ap.add_argument("--cache", default=CACHE_DEFAULT)
    ap.add_argument("--refresh", action="store_true", help="忽略缓存,重新从 .rda 构建")
    ap.add_argument("--out", default="outputs/bird_pareto")
    ap.add_argument("--min_specimens", type=int, default=2)
    ap.add_argument("--min_species", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import pandas as pd

    j = load_joined(a)
    j["u"] = (np.log10(j["tarsus_mm"]) - (A_W + B_W * np.log10(j["BodyMass.Value"]))) / SIG_W
    j["log_m"] = np.log10(j["BodyMass.Value"])
    print(f"\n联接后 {len(j)} 种 · {j['Order'].nunique()} 目 · {j['Family'].nunique()} 科")
    fs = [c for c in j.columns if c.lower().startswith("forstrat")]
    print(f"可用觅食层位列: {fs if fs else '无(L2 将跳过)'}\n")

    OUT = {"prior": dict(a=A_W, b=B_W, sigma=SIG_W), "ref_slopes": REF_SLOPES}

    def fit_group(sub):
        f = ols_loglog(sub["BodyMass.Value"], sub["tarsus_mm"])
        return dict(n=int(len(sub)), b=f["b"], ci95=f["ci95"], r2=f["r2"],
                    u_med=float(sub["u"].median()),
                    u_q1=float(sub["u"].quantile(.25)), u_q3=float(sub["u"].quantile(.75)),
                    log_m_span=float(sub["log_m"].max() - sub["log_m"].min()),
                    m_kg=(float(sub["BodyMass.Value"].min() / 1000),
                          float(sub["BodyMass.Value"].max() / 1000)))

    def table(rows, title, note=""):
        print(f"\n{'='*100}\n{title}\n{'='*100}")
        if note: print(note)
        print(f"{'类群/组':<26}{'n':>5}{'b':>8}{'±95CI':>8}{'R²':>6}{'跨度dex':>9}"
              f"{'u中位':>8}{'u IQR':>15}   基底")
        for r in rows:
            flag = "" if r["log_m_span"] >= 1.0 else "  ⚠跨度<1dex"
            print(f"{r['name']:<26}{r['n']:>5}{r['b']:>8.3f}{r['ci95']:>8.3f}{r['r2']:>6.2f}"
                  f"{r['log_m_span']:>9.2f}{r['u_med']:>+8.2f}  [{r['u_q1']:+.2f},{r['u_q3']:+.2f}]"
                  f"   {r.get('substrate','')}{flag}")

    # ---------------- L1 分类群 ----------------
    rows = []
    for name, (field, vals, sub_) in CLADES.items():
        sub = j[j[field].isin(vals)]
        if len(sub) < a.min_species:
            print(f"  ⚠ {name}: 仅 {len(sub)} 种,跳过"); continue
        rows.append({**fit_group(sub), "name": name, "substrate": sub_})
    for name, sel, sub_ in (("猛禽合并(鹰隼鸮)", j["Order"].isin(RAPTOR_ORDERS), "硬冲击着陆"),
                            ("全部鸟类", j["Order"].notna(), "—")):
        rows.append({**fit_group(j[sel]), "name": name, "substrate": sub_})
    rows.sort(key=lambda r: r["b"])
    table(rows, "L1 · 分类群标度律(按 b 升序)")
    OUT["L1_clades"] = rows

    # 与两个锚点的显著性
    wb = next(r for r in rows if r["name"].startswith("水鸟"))
    print(f"\n与水鸟(b={wb['b']:.3f})的斜率差异检验:")
    for r in rows:
        if r["name"].startswith("水鸟") or r["log_m_span"] < 1.0:
            continue
        d, z, p = slope_diff_z(r, wb)
        star = "***" if p < .001 else "**" if p < .01 else "*" if p < .05 else "n.s."
        print(f"  {r['name']:<26} Δb={d:+.3f}  z={z:+5.2f}  p={p:.2g}  {star}")

    # ---------------- L2 觅食基底 ----------------
    if fs:
        gcol = next((c for c in fs if "ground" in c.lower()), None)
        wcols = [c for c in fs if "wat" in c.lower()]
        arb = [c for c in fs if any(k in c.lower() for k in ("canopy", "midhigh", "understory"))]
        if gcol and wcols:
            j["_ground"] = j[gcol].astype(float)
            j["_water"] = j[wcols].astype(float).sum(axis=1)
            j["_arboreal"] = j[arb].astype(float).sum(axis=1) if arb else 0.0
            grp = []
            for nm, sel, sub_ in (
                ("地面为主 ≥70%", j["_ground"] >= 70, "硬地"),
                ("地面 40–70%",   (j["_ground"] >= 40) & (j["_ground"] < 70), "混合"),
                ("水面为主 ≥70%", j["_water"] >= 70, "水面"),
                ("树栖为主 ≥70%", j["_arboreal"] >= 70, "树枝"),
            ):
                sub = j[sel]
                if len(sub) >= a.min_species:
                    grp.append({**fit_group(sub), "name": nm, "substrate": sub_})
            table(grp, "L2 · 按觅食基底分组(比分类群更机制化)",
                  f"层位列: ground={gcol}  water={wcols}  arboreal={arb or '—'}\n"
                  "⚠ 觅食层位 ≠ 着陆基底,是代理量(见文件头诚实边界)")
            OUT["L2_substrate"] = grp
            # u 对 ground% 的连续回归(控制体重)
            X = np.column_stack([np.ones(len(j)), j["_ground"] / 100.0])
            beta, *_ = np.linalg.lstsq(X, j["u"].values, rcond=None)
            res = j["u"].values - X @ beta
            se = np.sqrt(np.sum(res**2) / (len(j) - 2) *
                         np.linalg.inv(X.T @ X)[1, 1])
            print(f"\nu ~ 地面觅食比例 的斜率: {beta[1]:+.3f} ± {1.96*se:.3f} (95%CI), n={len(j)}")
            print(f"  解读:比例每升到 100%,腿长残差 u 变化 {beta[1]:+.2f} σ。"
                  f"{'负值 = 越在地面活动,腿相对越短 ✓ 与假设一致' if beta[1] < 0 else '正值 = 与假设相反'}")
            OUT["L2_u_vs_ground"] = dict(slope=float(beta[1]), ci95=float(1.96*se), n=int(len(j)))
        else:
            print("\n[L2] 未识别出 ground/water 层位列,跳过")

    # ---------------- L3 体重匹配 ----------------
    wbm = j[j["Family"].isin(WATER_FAMILIES)]
    lo, hi = wbm["log_m"].quantile(.10), wbm["log_m"].quantile(.90)
    print(f"\n{'='*100}\nL3 · 体重匹配(限制在水鸟 10–90 分位体重区间: "
          f"{10**lo/1000:.2f}–{10**hi/1000:.2f} kg)\n{'='*100}")
    print("排除「b 差异其实来自体重跨度差异」的伪影。样本变少,CI 会变宽。")
    m3 = []
    for name, (field, vals, sub_) in CLADES.items():
        sub = j[j[field].isin(vals) & j["log_m"].between(lo, hi)]
        if len(sub) >= a.min_species:
            m3.append({**fit_group(sub), "name": name, "substrate": sub_})
    m3.sort(key=lambda r: r["b"])
    if m3: table(m3, "(体重匹配后)")
    OUT["L3_mass_matched"] = m3

    # ---------------- L4 科级聚合 ----------------
    print(f"\n{'='*100}\nL4 · 科级聚合(每科取中位数再拟合,粗略控制系统发育非独立性)\n{'='*100}")
    print("⚠ 这不是 PGLS,只是部分控制。严格做法需 Jetz et al. 2012 鸟类系统树。")
    fam = j.groupby(["Order", "Family"]).agg(
        tarsus_mm=("tarsus_mm", "median"), **{"BodyMass.Value": ("BodyMass.Value", "median")},
        n_sp=("tarsus_mm", "size")).reset_index()
    fam = fam[fam["n_sp"] >= 3]
    fam["u"] = (np.log10(fam["tarsus_mm"]) - (A_W + B_W * np.log10(fam["BodyMass.Value"]))) / SIG_W
    fam["log_m"] = np.log10(fam["BodyMass.Value"])
    f_all = fit_group(fam)
    print(f"  全部科 (n={f_all['n']} 科): b = {f_all['b']:.3f} ± {f_all['ci95']:.3f}, "
          f"R² = {f_all['r2']:.2f}   ← 对比物种级全鸟类")
    l4 = [{**f_all, "name": "全部科(聚合)", "substrate": "—"}]
    for nm, sel in (("水鸟科", fam["Family"].isin(WATER_FAMILIES)),
                    ("鸡形目各科", fam["Order"] == "Galliformes"),
                    ("猛禽各科", fam["Order"].isin(RAPTOR_ORDERS))):
        sub = fam[sel]
        if len(sub) >= 3:
            print(f"  {nm} (n={len(sub)} 科): u 中位 {sub['u'].median():+.2f}")
            l4.append(dict(name=nm, n=int(len(sub)), u_med=float(sub["u"].median()),
                           b=np.nan, ci95=np.nan, r2=np.nan, log_m_span=np.nan,
                           u_q1=float(sub["u"].quantile(.25)), u_q3=float(sub["u"].quantile(.75)),
                           m_kg=(np.nan, np.nan), substrate="—"))
    OUT["L4_family_level"] = l4

    # ---------------- 结论提示 ----------------
    print(f"\n{'='*100}\n参考斜率\n{'='*100}")
    for k, v in sorted(REF_SLOPES.items(), key=lambda t: t[1]):
        print(f"  {k:<28} {v:.3f}")
    print("\n读法:")
    print("  · b 越小 = 腿随体重涨得越慢。b_eff≈0.24 是我们的着陆物理自己选的指数。")
    print("  · u<0 = 同体重下比水鸟先验腿短。可与既有五科并列(雁鸭 −0.43 … 䴙䴘 +1.72)。")
    print("  · 跨度 <1 dex 的类群 b 不可信(CI 宽),看 u。")
    print("  · L1→L4 逐层更保守;若结论在四层都稳,才可写进论文。")

    p = os.path.join(a.out, "allometry_clades.json")
    json.dump(OUT, open(p, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n→ {p}")


if __name__ == "__main__":
    main()

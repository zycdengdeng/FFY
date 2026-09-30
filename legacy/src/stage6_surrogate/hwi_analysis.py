# -*- coding: utf-8 -*-
"""腿长 vs 飞行能力(HWI)与食性 —— 把 r=−0.63 这条关系查到底。

## 背景

v3 发现:控制体重后,腿长残差 u 与手翼指数 HWI 的相关 r = −0.625(n=9146),
远强于"着陆基底"分组(r ≈ 0.1)。若成立,它给出一个比"基底"更根本的解释:
**腿长与飞行能力是权衡** —— 腿是必须带上天的死重与阻力。
这与起落架同构(起落架也是死重),并解释了为何我们的 b_eff=0.238 比所有会飞的鸟都平:
优化器只优化着陆,不必付飞行的代价。

## 但 r=−0.63 有四个可能的假象来源,本脚本逐个排除

  A **体型混杂**   u 已按体重归一,但 HWI 本身随体型变 → 做偏相关,控制 log(m)
  B **类群驱动**   雀形目占 58%,可能整条关系只是"雀形目 vs 其它" → 各目内部分别拟合
  C **系统发育**   近缘种既共享翼型也共享腿型 → 科级/属级聚合(粗控;PGLS 才是正解)
  D **食性混杂**   猛禽要抓握(腿长)、也要高效飞行 → 把食性做协变量与分层

## 输出

  ① 全样本偏相关(控制 log m)与多元回归 u ~ HWI + logm + 食性
  ② 目内 / 科内分别拟合(固定效应式的粗控)
  ③ 科级、属级聚合后的关系
  ④ 食性 5 类分层 + 食性百分比作连续协变量
  ⑤ 我们关心的水鸟/鸡形目在这张图上的位置

## 诚实边界

  · 属级/科级聚合是**粗略**的系统发育控制,不能替代 PGLS(需 Jetz et al. 2012 系统树)。
    若本关系要作为论文主结论,PGLS 是必需项而非可选项。
  · HWI 是飞行效率/扩散能力的形态代理,不是直接的飞行性能测量。
  · 相关不是因果;腿长受多功能驱动,本分析只能说"HWI 是最强的单一相关量"。
"""
from __future__ import annotations
import argparse, json, os
import numpy as np

A_W, B_W, SIG_W = 0.479, 0.391, 0.0784
WATER = ["Anatidae", "Gaviidae", "Podicipedidae", "Pelecanidae", "Phalacrocoracidae"]
FLIGHTLESS_ORDERS = {"Struthioniformes", "Rheiformes", "Casuariiformes",
                     "Apterygiformes", "Sphenisciformes"}
FLIGHTLESS_GENERA = {"Tachyeres", "Nannopterum", "Rollandia", "Centropelma", "Podilymbus"}
FLIGHTLESS_SPECIES = {"Anas aucklandica", "Anas nesiotis", "Anas chlorotis",
                      "Podiceps taczanowskii", "Phalacrocorax harrisi"}
DIET = ["Diet.Inv", "Diet.Vend", "Diet.Vect", "Diet.Vfish", "Diet.Vunk",
        "Diet.Scav", "Diet.Fruit", "Diet.Nect", "Diet.Seed", "Diet.PlantO"]


def olsm(X, y, names):
    """多元 OLS,返回系数、95CI、t。X 不含截距(内部加)。"""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    res = y - X @ beta
    dof = len(y) - X.shape[1]
    s2 = float(res @ res) / dof
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    sst = float(((y - y.mean()) ** 2).sum())
    return dict(names=["const"] + list(names),
                beta=beta.tolist(), ci95=(1.96 * se).tolist(),
                t=(beta / se).tolist(), n=int(len(y)),
                r2=float(1 - float(res @ res) / sst) if sst > 0 else np.nan)


def partial_corr(x, y, z):
    """x,y 在控制 z(可多列)后的偏相关。"""
    Z = np.column_stack([np.ones(len(x)), np.asarray(z, float)])
    rx = x - Z @ np.linalg.lstsq(Z, x, rcond=None)[0]
    ry = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
    return float(np.corrcoef(rx, ry)[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avonet", default="/tmp/traitdata/data/avonet.rda")
    ap.add_argument("--elton", default="/tmp/traitdata/data/elton_birds.rda")
    ap.add_argument("--cache", default="data/avonet_hwi.csv")
    ap.add_argument("--refresh", action="store_true")
    ap.add_argument("--out", default="outputs/bird_pareto")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import pandas as pd

    if os.path.exists(a.cache) and not a.refresh:
        j = pd.read_csv(a.cache); print(f"[数据] 读缓存 {a.cache}")
    else:
        import pyreadr
        av = pyreadr.read_r(a.avonet)["avonet"]
        el = pyreadr.read_r(a.elton)["elton_birds"]
        av = av.dropna(subset=["Tarsus.Length", "scientificNameStd"])
        g = av.groupby("scientificNameStd")
        sp = g[["Tarsus.Length", "Wing.Length", "Kipps.Distance", "Hand.wing.Index"]].median()
        sp["n_spec"] = g.size()
        sp = sp[sp["n_spec"] >= 2]
        keep = ["BodyMass.Value", "Family", "Order", "Diet.5Cat", "Nocturnal",
                "PelagicSpecialist"] + DIET
        keep = [c for c in keep if c in el.columns]
        el2 = (el.dropna(subset=["BodyMass.Value", "scientificNameStd"])
                 .set_index("scientificNameStd")[keep])
        j = sp.join(el2, how="inner").dropna(subset=["BodyMass.Value", "Family", "Order"]).reset_index()
        j = j[j["BodyMass.Value"] > 0]
        # EltonTraits 同名多条记录会让 join 膨胀(实测 118 名 / 433 行,
        # 且内容完全相同,如 Aerodramus brevirostris 重复 65 次)→ 去重
        j = j.drop_duplicates("scientificNameStd")
        os.makedirs(os.path.dirname(a.cache) or ".", exist_ok=True)
        j.to_csv(a.cache, index=False); print(f"[数据] 写缓存 {a.cache}")

    gen = j["scientificNameStd"].str.split().str[0]
    fl = (j["Order"].isin(FLIGHTLESS_ORDERS) | gen.isin(FLIGHTLESS_GENERA)
          | j["scientificNameStd"].isin(FLIGHTLESS_SPECIES))
    j = j[~fl].copy()                                  # 只留会飞的
    j["genus"] = gen[~fl].values
    j["log_m"] = np.log10(j["BodyMass.Value"])
    j["u"] = (np.log10(j["Tarsus.Length"]) - (A_W + B_W * j["log_m"])) / SIG_W
    V = j.dropna(subset=["Hand.wing.Index"]).copy()
    print(f"会飞且有 HWI 的物种: {len(V)} / {len(j)}\n")
    OUT = {}

    # ---------- ① 偏相关 + 多元回归 ----------
    print("=" * 92); print("① 控制体型后,HWI 与腿长残差 u 的关系"); print("=" * 92)
    r_raw = float(np.corrcoef(V["Hand.wing.Index"], V["u"])[0, 1])
    r_pm = partial_corr(V["Hand.wing.Index"].values, V["u"].values, V[["log_m"]].values)
    print(f"  原始相关            r = {r_raw:+.3f}")
    print(f"  控制 log(体重) 后    r = {r_pm:+.3f}   ← A 体型混杂检验")
    m1 = olsm(V[["Hand.wing.Index", "log_m"]].values, V["u"].values, ["HWI", "log_m"])
    for nm, b, c, t in zip(m1["names"], m1["beta"], m1["ci95"], m1["t"]):
        print(f"    {nm:<10} β = {b:+.4f} ± {c:.4f}   t = {t:+7.1f}")
    print(f"    R² = {m1['r2']:.3f}  n = {m1['n']}")
    OUT["step1"] = dict(r_raw=r_raw, r_partial_mass=r_pm, model=m1)

    # 加食性
    dc = [c for c in DIET if c in V.columns]
    Vd = V.dropna(subset=dc)
    m2 = olsm(Vd[["Hand.wing.Index", "log_m"] + dc[:-1]].values, Vd["u"].values,
              ["HWI", "log_m"] + dc[:-1])
    print(f"\n  加入食性百分比后(去掉一列防共线):")
    for nm, b, c, t in zip(m2["names"][:4], m2["beta"][:4], m2["ci95"][:4], m2["t"][:4]):
        print(f"    {nm:<12} β = {b:+.4f} ± {c:.4f}   t = {t:+7.1f}")
    big = sorted(zip(m2["names"][3:], m2["beta"][3:], m2["t"][3:]),
                 key=lambda x: -abs(x[2]))[:4]
    print(f"    食性中 |t| 最大的四项: " + ", ".join(f"{n}({t:+.0f})" for n, _, t in big))
    print(f"    R² = {m2['r2']:.3f}  n = {m2['n']}   ← D 食性混杂检验")
    OUT["step2_with_diet"] = m2

    # ---------- ② 目内 / 科内 ----------
    print("\n" + "=" * 92); print("② 各目内部分别拟合(B 类群驱动检验)"); print("=" * 92)
    print(f"  {'目':<24}{'n':>6}{'r(HWI,u|m)':>13}{'β_HWI':>10}")
    rows = []
    for o, sub in V.groupby("Order"):
        if len(sub) < 40: continue
        rp = partial_corr(sub["Hand.wing.Index"].values, sub["u"].values, sub[["log_m"]].values)
        mm = olsm(sub[["Hand.wing.Index", "log_m"]].values, sub["u"].values, ["HWI", "log_m"])
        rows.append(dict(order=o, n=int(len(sub)), r=rp, beta=mm["beta"][1], t=mm["t"][1]))
    rows.sort(key=lambda d: d["r"])
    for d in rows:
        print(f"  {d['order']:<24}{d['n']:>6}{d['r']:>+13.3f}{d['beta']:>+10.4f}")
    neg = sum(1 for d in rows if d["r"] < 0)
    print(f"\n  → {neg}/{len(rows)} 个目内部为负相关"
          f"{'  ✓ 不是雀形目单独驱动' if neg > len(rows)*0.7 else '  ⚠ 方向不一致'}")
    OUT["step3_within_order"] = rows

    # ---------- ③ 系统发育粗控 ----------
    print("\n" + "=" * 92); print("③ 聚合后的关系(C 系统发育粗控;非 PGLS)"); print("=" * 92)
    agg_rows = []
    for lab, key in (("属级", "genus"), ("科级", "Family"), ("目级", "Order")):
        g2 = V.groupby(key).agg(u=("u", "median"), HWI=("Hand.wing.Index", "median"),
                                log_m=("log_m", "median"), n=("u", "size"))
        g2 = g2[g2["n"] >= 3]
        if len(g2) < 10: continue
        rp = partial_corr(g2["HWI"].values, g2["u"].values, g2[["log_m"]].values)
        mm = olsm(g2[["HWI", "log_m"]].values, g2["u"].values, ["HWI", "log_m"])
        print(f"  {lab}聚合  单元 {len(g2):>4}   r(HWI,u|m) = {rp:+.3f}   "
              f"β_HWI = {mm['beta'][1]:+.4f} ± {mm['ci95'][1]:.4f}   t = {mm['t'][1]:+.1f}")
        agg_rows.append(dict(level=lab, n_units=int(len(g2)), r=rp,
                             beta=mm["beta"][1], ci95=mm["ci95"][1], t=mm["t"][1]))
    OUT["step4_aggregated"] = agg_rows

    # ---------- ④ 食性分层 ----------
    print("\n" + "=" * 92); print("④ 按食性 5 类分层(D 食性混杂)"); print("=" * 92)
    print(f"  {'食性':<16}{'n':>6}{'u中位':>9}{'HWI中位':>10}{'r(HWI,u|m)':>13}{'b(腿长标度)':>13}")
    diet_rows = []
    for d5, sub in V.groupby("Diet.5Cat"):
        if len(sub) < 40: continue
        rp = partial_corr(sub["Hand.wing.Index"].values, sub["u"].values, sub[["log_m"]].values)
        X = np.column_stack([np.ones(len(sub)), sub["log_m"].values])
        bb = np.linalg.lstsq(X, np.log10(sub["Tarsus.Length"].values), rcond=None)[0][1]
        print(f"  {d5:<16}{len(sub):>6}{sub['u'].median():>+9.2f}"
              f"{sub['Hand.wing.Index'].median():>10.1f}{rp:>+13.3f}{bb:>13.3f}")
        diet_rows.append(dict(diet=d5, n=int(len(sub)), u_med=float(sub["u"].median()),
                              hwi_med=float(sub["Hand.wing.Index"].median()), r=rp, b=float(bb)))
    OUT["step5_diet"] = diet_rows
    # 食性百分比与 u 的单变量相关(控制体重)
    print(f"\n  各食性成分与 u 的偏相关(控制体重):")
    dr = []
    for c in dc:
        s = V.dropna(subset=[c])
        if len(s) < 100: continue
        rp = partial_corr(s[c].values.astype(float), s["u"].values, s[["log_m"]].values)
        dr.append((c, rp, len(s)))
    for c, rp, n in sorted(dr, key=lambda t: -abs(t[1]))[:6]:
        print(f"    {c:<14} r = {rp:+.3f}  (n={n})")
    OUT["step5_diet_components"] = [dict(comp=c, r=r, n=n) for c, r, n in dr]

    # ---------- ⑤ 我们关心的类群在哪 ----------
    print("\n" + "=" * 92); print("⑤ 本项目相关类群的位置"); print("=" * 92)
    print(f"  {'类群':<20}{'n':>6}{'HWI中位':>10}{'u中位':>9}{'b':>9}")
    foc = {"水鸟(5科)": V["Family"].isin(WATER), "雁鸭科": V["Family"] == "Anatidae",
           "鸡形目": V["Order"] == "Galliformes",
           "猛禽3目": V["Order"].isin(["Accipitriformes", "Falconiformes", "Strigiformes"]),
           "雀形目": V["Order"] == "Passeriformes", "全部会飞": V["Order"].notna()}
    focus = []
    for nm, sel in foc.items():
        s = V[sel]
        X = np.column_stack([np.ones(len(s)), s["log_m"].values])
        bb = np.linalg.lstsq(X, np.log10(s["Tarsus.Length"].values), rcond=None)[0][1]
        print(f"  {nm:<20}{len(s):>6}{s['Hand.wing.Index'].median():>10.1f}"
              f"{s['u'].median():>+9.2f}{bb:>9.3f}")
        focus.append(dict(name=nm, n=int(len(s)), hwi=float(s["Hand.wing.Index"].median()),
                          u=float(s["u"].median()), b=float(bb)))
    OUT["step6_focus"] = focus

    print("\n判读:")
    print("  · ① 控制体重后 r 若仍强 → A 排除;② 多数目内为负 → B 排除")
    print("  · ③ 聚合后 r 若塌掉 → 关系主要是系统发育的,须做 PGLS 才能主张")
    print("  · ④ 各食性层内 r 若同向 → D 排除")
    p = os.path.join(a.out, "hwi_analysis.json")
    json.dump(OUT, open(p, "w"), indent=1, ensure_ascii=False, default=float)
    print(f"\n→ {p}")


if __name__ == "__main__":
    main()

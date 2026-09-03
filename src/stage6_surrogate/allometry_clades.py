# -*- coding: utf-8 -*-
"""跨类群腿长标度律:水鸟是不是"长腿的极端"?硬地着陆的鸟更接近物理偏好的指数吗?

动机(2026-09-03):我们用水鸟提取先验,但设计对象是**硬地**起落架;而 E21 已表明
真鸟腿是为着水优化的(水面允许更长减速距离)。E17 给出着陆物理偏好的涌现指数
b_eff ≈ 0.24–0.28,远低于水鸟的 0.39。**那么在硬地/树枝上着陆的鸟,腿的标度律更接近哪个?**

这是论文 A("生物标度律能否迁移到着陆任务")的正面控制组:
  · 若猛禽/地栖鸟的 b 更接近 b_eff、u < 0(同体重腿更短) → 着陆物理偏好的指数在自然界被实现了,
    水鸟是"为水面超配"的特例 —— 故事闭合。
  · 若不是 → 鸟类腿长标度律不主要由着陆驱动 —— 同样是可写的结论。

输出两个量:
  b   该类群内 log10(跗跖) ~ log10(体重) 的 OLS 斜率(带 95% CI)
  u   每个物种相对**水鸟先验**(a=0.479, b=0.391, σ=0.0784)的标准化残差 —— 与既有五科 u 同尺度,可直接比

用法(A100,traitdata 已拉过):
  python src/stage6_surrogate/allometry_clades.py \
      --avonet /tmp/traitdata/data/avonet.rda --elton /tmp/traitdata/data/elton_birds.rda \
      --out outputs/bird_pareto
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from allometry_data import ols_loglog, WATER_FAMILIES

# 项目既定的水鸟先验(与 bioprior.py 一致),u 以它为零点
A_W, B_W, SIG_W = 0.479, 0.391, 0.0784
# 参考斜率
REF = {"物理涌现 b_eff (v2.2)": 0.238, "物理涌现 b_eff (v2.1)": 0.281,
       "弹性相似 McMahon 1/4": 0.25, "几何相似 1/3": 0.333}

# 类群 → 着陆基底 / 运动生态。用 EltonTraits 的 Order/Family(traitdata 标准化名)。
CLADES = {
    # 名称: (筛选字段, 取值集合, 着陆基底描述)
    "水鸟(既有 5 科)":     ("Family", set(WATER_FAMILIES), "水面"),
    "雁鸭科 Anatidae":     ("Family", {"Anatidae"}, "水面/硬地兼有"),
    "鹰形目 Accipitriformes": ("Order", {"Accipitriformes"}, "树枝/地面/猎物,硬冲击"),
    "隼形目 Falconiformes":   ("Order", {"Falconiformes"}, "岩壁/树枝,高速俯冲后落地"),
    "鸮形目 Strigiformes":    ("Order", {"Strigiformes"}, "树枝/地面,猎物冲击"),
    "鸡形目 Galliformes":     ("Order", {"Galliformes"}, "地面为主,火鸡/珍珠鸡=陆栖腿力学模型"),
    "鸽形目 Columbiformes":   ("Order", {"Columbiformes"}, "硬地/树枝"),
    "鹭科 Ardeidae":          ("Family", {"Ardeidae"}, "涉水,长腿"),
    "鹤形目 Gruiformes":      ("Order", {"Gruiformes"}, "涉水/地面"),
    "鸻形目 Charadriiformes": ("Order", {"Charadriiformes"}, "滩涂/水面,混合"),
    "雀形目 Passeriformes":   ("Order", {"Passeriformes"}, "树枝,栖息为主"),
    "鹦形目 Psittaciformes":  ("Order", {"Psittaciformes"}, "树枝"),
}
# 猛禽合并组(三目并)
RAPTORS = {"Accipitriformes", "Falconiformes", "Strigiformes"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avonet", default="/tmp/traitdata/data/avonet.rda")
    ap.add_argument("--elton", default="/tmp/traitdata/data/elton_birds.rda")
    ap.add_argument("--out", default="outputs/bird_pareto")
    ap.add_argument("--min_specimens", type=int, default=2)
    ap.add_argument("--min_species", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    import pyreadr, pandas as pd
    av = pyreadr.read_r(a.avonet); av = av[list(av.keys())[0]]
    el = pyreadr.read_r(a.elton); el = el[list(el.keys())[0]]
    av = av.dropna(subset=["Tarsus.Length", "scientificNameStd"])
    g = av.groupby("scientificNameStd")["Tarsus.Length"]
    sp = g.median().to_frame("tarsus_mm"); sp["n_spec"] = g.size()
    sp = sp[sp["n_spec"] >= a.min_specimens]
    el2 = el.dropna(subset=["BodyMass.Value", "scientificNameStd"])
    el2 = el2.set_index("scientificNameStd")[["BodyMass.Value", "Family", "Order"]]
    j = sp.join(el2, how="inner").dropna(subset=["BodyMass.Value", "Family", "Order"])
    j = j[j["BodyMass.Value"] > 0].copy()
    # 相对水鸟先验的标准化残差 u
    j["u"] = (np.log10(j["tarsus_mm"]) - (A_W + B_W * np.log10(j["BodyMass.Value"]))) / SIG_W
    print(f"联接后物种 {len(j)}  ·  目 {j['Order'].nunique()}  ·  科 {j['Family'].nunique()}")
    print("Order 取值样例:", sorted(j["Order"].unique().tolist())[:12], "...\n")

    rows = []
    def add(name, sub, substrate):
        if len(sub) < a.min_species:
            print(f"  ⚠ {name}: 仅 {len(sub)} 种,跳过"); return
        f = ols_loglog(sub["BodyMass.Value"], sub["tarsus_mm"])
        rows.append(dict(clade=name, substrate=substrate, n=int(len(sub)),
                         b=f["b"], ci95=f["ci95"], r2=f["r2"],
                         u_med=float(sub["u"].median()),
                         u_q1=float(sub["u"].quantile(.25)), u_q3=float(sub["u"].quantile(.75)),
                         m_kg_range=(float(sub["BodyMass.Value"].min()/1000),
                                     float(sub["BodyMass.Value"].max()/1000)),
                         log_m_span=float(np.log10(sub["BodyMass.Value"].max()/sub["BodyMass.Value"].min()))))
    for name, (field, vals, sub_) in CLADES.items():
        add(name, j[j[field].isin(vals)], sub_)
    add("猛禽合并(鹰+隼+鸮)", j[j["Order"].isin(RAPTORS)], "硬冲击着陆")
    add("全部鸟类", j, "—")

    # 打印
    print(f"\n{'类群':<26}{'n':>6}{'b':>8}{'±':>6}{'R²':>6}{'体重跨度(dex)':>13}{'u 中位':>8}{'u IQR':>14}   着陆基底")
    for r in sorted(rows, key=lambda r: r["b"]):
        print(f"{r['clade']:<26}{r['n']:>6}{r['b']:>8.3f}{r['ci95']:>6.3f}{r['r2']:>6.2f}"
              f"{r['log_m_span']:>13.2f}{r['u_med']:>+8.2f}  [{r['u_q1']:+.2f},{r['u_q3']:+.2f}]   {r['substrate']}")
    print("\n参考斜率:", "  ".join(f"{k}={v}" for k, v in REF.items()))
    print("\n读法:")
    print("  · b 越小,腿随体重涨得越慢;b_eff≈0.24 是着陆物理自己选的。")
    print("  · u<0 = 同体重下比水鸟先验腿短;u 中位可直接与既有五科(雁鸭 −0.43 … 䴙䴘 +1.72)并列。")
    print("  · 体重跨度 <1 dex 的类群 b 不可信(CI 会很宽),看 u 就好。")
    json.dump(dict(prior=dict(a=A_W, b=B_W, sigma=SIG_W), ref=REF, rows=rows),
              open(os.path.join(a.out, "allometry_clades.json"), "w"), indent=1, ensure_ascii=False)
    print(f"\n→ {a.out}/allometry_clades.json")


if __name__ == "__main__":
    main()

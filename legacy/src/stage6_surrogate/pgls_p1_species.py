# -*- coding: utf-8 -*-
"""PGLS P1 · 导出物种名单并做名称规范化,供 BirdTree/VertLife 取树。

BirdTree(Jetz et al. 2012)用的分类与 EltonTraits 的 scientificNameStd 不完全一致,
名称匹配是 PGLS 流程里最主要的人工量。本脚本:

  1. 从分析用表导出物种名单(BirdTree 的 tip label 格式:属_种)
  2. 做基本规范化(去亚种、去杂交符、统一空格/下划线)
  3. 输出三份:上传用名单、规范化映射表、需人工核对的可疑项

用法:
  python src/stage6_surrogate/pgls_p1_species.py \
      --data data/avonet_hwi.csv --out outputs/pgls

之后:把 outputs/pgls/species_for_tree.txt 上传到 https://birdtree.org/subsets/
     选 Hackett 骨架 / 100 棵 / Nexus → 下载;再取一份 Ericson。
"""
from __future__ import annotations
import argparse, os, re
import numpy as np

FLIGHTLESS_ORDERS = {"Struthioniformes", "Rheiformes", "Casuariiformes",
                     "Apterygiformes", "Sphenisciformes"}
FLIGHTLESS_GENERA = {"Tachyeres", "Nannopterum", "Rollandia", "Centropelma", "Podilymbus"}
FLIGHTLESS_SPECIES = {"Anas aucklandica", "Anas nesiotis", "Anas chlorotis",
                      "Podiceps taczanowskii", "Phalacrocorax harrisi"}


def normalize(name: str):
    """→ (tip_label, 是否可疑, 原因)。BirdTree tip 格式为 Genus_species。"""
    s = str(name).strip()
    flags = []
    if "×" in s or " x " in s.lower():
        flags.append("疑似杂交")
    s = re.sub(r"[×x]\s*", "", s) if "疑似杂交" in flags else s
    s = re.sub(r"\s+", " ", s.replace("_", " ")).strip()
    parts = s.split(" ")
    if len(parts) > 2:
        flags.append(f"三名法(亚种?),截为前两词:原「{s}」")
        parts = parts[:2]
    if len(parts) < 2:
        flags.append("不足两词")
        return None, True, "; ".join(flags)
    g, sp = parts[0], parts[1].lower()
    if not g[:1].isupper():
        flags.append("属名首字母非大写")
    if not re.fullmatch(r"[A-Za-z\-]+", sp):
        flags.append(f"种加词含异常字符「{sp}」")
    return f"{g}_{sp}", bool(flags), "; ".join(flags)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/avonet_hwi.csv")
    ap.add_argument("--out", default="outputs/pgls")
    ap.add_argument("--keep-flightless", action="store_true",
                    help="默认剔除不会飞的物种(与主分析口径一致)")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import pandas as pd

    j = pd.read_csv(a.data)
    n0 = len(j)
    if not a.keep_flightless:
        gen = j["scientificNameStd"].str.split().str[0]
        fl = (j["Order"].isin(FLIGHTLESS_ORDERS) | gen.isin(FLIGHTLESS_GENERA)
              | j["scientificNameStd"].isin(FLIGHTLESS_SPECIES))
        j = j[~fl].copy()
        print(f"剔除不会飞 {int(fl.sum())} 种")
    j = j.drop_duplicates("scientificNameStd")
    j = j.dropna(subset=["Hand.wing.Index", "Tarsus.Length", "BodyMass.Value"])
    print(f"分析样本: {len(j)} 种(原表 {n0})")

    recs = []
    for nm in j["scientificNameStd"]:
        tip, sus, why = normalize(nm)
        recs.append(dict(scientificNameStd=nm, tip_label=tip,
                         suspicious=sus, note=why))
    m = pd.DataFrame(recs)
    ok = m[m["tip_label"].notna()]
    dup = ok[ok.duplicated("tip_label", keep=False)].sort_values("tip_label")

    # 产出
    p_list = os.path.join(a.out, "species_for_tree.txt")
    with open(p_list, "w") as f:
        f.write("\n".join(sorted(ok["tip_label"].dropna().unique())) + "\n")
    m.to_csv(os.path.join(a.out, "name_map.csv"), index=False)
    sus = m[m["suspicious"]]
    sus.to_csv(os.path.join(a.out, "name_review.csv"), index=False)

    # 带上 tip_label 的分析表(P2 直接用)
    j2 = j.merge(m[["scientificNameStd", "tip_label"]], on="scientificNameStd", how="left")
    j2["log_m"] = np.log10(j2["BodyMass.Value"])
    j2["u"] = (np.log10(j2["Tarsus.Length"]) - (0.479 + 0.391 * j2["log_m"])) / 0.0784
    j2 = j2.rename(columns={"Hand.wing.Index": "HWI"})
    cols = ["tip_label", "scientificNameStd", "u", "HWI", "log_m",
            "Tarsus.Length", "BodyMass.Value", "Family", "Order", "Diet.5Cat"]
    j2[[c for c in cols if c in j2.columns]].to_csv(
        os.path.join(a.out, "pgls_data.csv"), index=False)

    print(f"\n可用 tip_label: {ok['tip_label'].nunique()} 个唯一名")
    print(f"需人工核对:      {len(sus)} 条  → {a.out}/name_review.csv")
    if len(dup):
        print(f"⚠ 规范化后重名:  {dup['tip_label'].nunique()} 组 "
              f"({len(dup)} 条) —— 多为亚种被截成同一名,须人工合并或去重")
        for t, g in list(dup.groupby("tip_label"))[:5]:
            print(f"    {t}: {list(g['scientificNameStd'])}")
    if len(sus):
        print("\n可疑项样例:")
        for _, r in sus.head(8).iterrows():
            print(f"    {r['scientificNameStd']:<34} → {r['tip_label']}   [{r['note']}]")
    print(f"\n产出:")
    print(f"  {p_list}                  ← 上传到 birdtree.org/subsets/")
    print(f"  {a.out}/pgls_data.csv     ← P2 用的分析表(已带 tip_label / u / HWI / log_m)")
    print(f"  {a.out}/name_map.csv      ← 完整映射,供回溯")
    print(f"  {a.out}/name_review.csv   ← 需人工核对")
    print(f"\n下一步:取树时选 **Hackett 骨架 / 100 棵 / Nexus**;再取一份 Ericson 骨架。")


if __name__ == "__main__":
    main()

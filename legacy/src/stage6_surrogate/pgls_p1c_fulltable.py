# -*- coding: utf-8 -*-
"""P1c · 生成系统发育全家桶的输入表(在 P1b 匹配逻辑之上补齐分类/食性/原始骨长)
产出 data/birdtree/pgls_data_full.csv:
  tip, u, HWI, log_m, logL, Family, Order, Diet.5Cat, Diet.*(10列), is_water
"""
import io, re, sys, zipfile, unicodedata
import numpy as np, pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
BT   = f"{ROOT}/data/birdtree"
A_W, B_W, SIG_W = 0.479, 0.391, 0.0784
WATER = ["Anatidae", "Gaviidae", "Podicipedidae", "Pelecanidae", "Phalacrocoracidae"]
FL_ORD = {"Struthioniformes","Rheiformes","Casuariiformes","Apterygiformes","Sphenisciformes"}
FL_GEN = {"Tachyeres","Nannopterum","Rollandia","Centropelma","Podilymbus"}
FL_SPP = {"Anas aucklandica","Anas nesiotis","Anas chlorotis",
          "Podiceps taczanowskii","Phalacrocorax harrisi"}
DIET = ["Diet.Inv","Diet.Vend","Diet.Vect","Diet.Vfish","Diet.Vunk",
        "Diet.Scav","Diet.Fruit","Diet.Nect","Diet.Seed","Diet.PlantO"]

# 1 · 树 tip
with zipfile.ZipFile(f"{BT}/HackettStage2_0001_1000.zip") as z:
    with z.open(z.namelist()[0]) as f:
        line1 = io.TextIOWrapper(f, "utf-8", errors="replace").readline()
tips = set(re.findall(r"[(,]([A-Za-z][A-Za-z_.'\-]*?):", line1))
tips_sp = {t.replace("_", " "): t for t in tips}
print(f"[树] {len(tips)} tip")

# 2 · crosswalk
cw = pd.read_excel(f"{BT}/AVONET_Supplementary_dataset1.xlsx",
                   sheet_name="BirdLife–BirdTree crosswalk")
cw.columns = [c.strip() for c in cw.columns]
m13 = {}
for a, b in zip(cw["Species1"], cw["Species3"]):
    if isinstance(a, str) and isinstance(b, str):
        m13.setdefault(a.strip(), b.strip())

# 3 · 性状表
d = pd.read_csv(f"{ROOT}/data/avonet_hwi.csv").drop_duplicates("scientificNameStd")
d = d[(d["BodyMass.Value"] > 0) & d["Tarsus.Length"].notna()].copy()
gen = d["scientificNameStd"].str.split().str[0]
d = d[~(d["Order"].isin(FL_ORD) | gen.isin(FL_GEN)
        | d["scientificNameStd"].isin(FL_SPP))].copy()
d["log_m"] = np.log10(d["BodyMass.Value"])
d["logL"]  = np.log10(d["Tarsus.Length"])
d["u"]     = (d["logL"] - (A_W + B_W * d["log_m"])) / SIG_W
d["is_water"] = d["Family"].isin(WATER).astype(int)
d = d[np.isfinite(d["u"])]

nrm = lambda s: unicodedata.normalize("NFKC", s).strip()
tip, how = [], []
for nm in d["scientificNameStd"]:
    n = nrm(nm)
    if n in tips_sp: tip.append(tips_sp[n]); how.append("direct"); continue
    b = m13.get(n)
    if b and nrm(b) in tips_sp: tip.append(tips_sp[nrm(b)]); how.append("crosswalk"); continue
    tip.append(None); how.append("miss")
d["tip"], d["how"] = tip, how
print(f"[匹配] 直接 {(d.how=='direct').sum()} + crosswalk {(d.how=='crosswalk').sum()}"
      f" = {d.tip.notna().sum()}/{len(d)}({d.tip.notna().mean():.1%})")

ok = d[d.tip.notna()].copy()
ok = ok.loc[ok.assign(a=ok.u.abs()).sort_values("a").drop_duplicates("tip").index].sort_values("tip")
cols = ["tip","u","HWI","log_m","logL","Family","Order","Diet.5Cat","is_water"] + DIET
ok = ok.rename(columns={"Hand.wing.Index":"HWI"})
ok[cols].to_csv(f"{BT}/pgls_data_full.csv", index=False)
print(f"[产出] pgls_data_full.csv  {len(ok)} 种 · 水鸟 {int(ok.is_water.sum())} 种"
      f" · 有 HWI {int(ok.HWI.notna().sum())} 种")

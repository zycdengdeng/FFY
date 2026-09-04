# -*- coding: utf-8 -*-
"""P1b · 名字对齐:我们的 8831 种 ↔ BirdTree 树 tip
级联匹配:①直接同名 ②BirdLife–BirdTree crosswalk ③(报告未匹配)
产出:
  data/birdtree/pgls_data_matched.csv   tip(树格式)/u/HWI/log_m —— P2/P3 的唯一输入
  data/birdtree/unmatched.csv           没对上的名单(供人工核对)
"""
import io, sys, zipfile, unicodedata
import pandas as pd

ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
BT   = f"{ROOT}/data/birdtree"

# ---- 1 · 树的 tip 名单(translate 表,流式读 zip 头部,不解压 464MB) ----
tips = set()
import re
with zipfile.ZipFile(f"{BT}/HackettStage2_0001_1000.zip") as z:
    with z.open(z.namelist()[0]) as f:
        line1 = io.TextIOWrapper(f, "utf-8", errors="replace").readline()
tips = set(re.findall(r"[(,]([A-Za-z][A-Za-z_.'\-]*?):", line1))
print(f"[树] tip 数 = {len(tips)}(应为 9993)")

# ---- 2 · crosswalk 表(BirdLife Species1 ↔ BirdTree Species3) ----
xls = f"{BT}/AVONET_Supplementary_dataset1.xlsx"
cw = pd.read_excel(xls, sheet_name="BirdLife–BirdTree crosswalk")
cw.columns = [c.strip() for c in cw.columns]
c1 = [c for c in cw.columns if "1" in c or "BirdLife" in c][0]
c3 = [c for c in cw.columns if "3" in c or "BirdTree" in c][0]
print(f"[crosswalk] {len(cw)} 行,列 {c1!r} → {c3!r}")
m13 = {}
for a, b in zip(cw[c1], cw[c3]):
    if isinstance(a, str) and isinstance(b, str):
        m13.setdefault(a.strip(), b.strip())

# ---- 3 · 我们的分析表 ----
d = pd.read_csv(f"{ROOT}/outputs/pgls/pgls_data.csv")
name_sp = d["tip_label"].str.replace("_", " ", regex=False)

def norm(s):  # 去附标/统一空白
    return unicodedata.normalize("NFKC", s).strip()

tips_sp = {t.replace("_", " "): t for t in tips}

how, tip_final = [], []
for nm in name_sp:
    nm = norm(nm)
    if nm in tips_sp:                       # ① 直接同名
        how.append("direct"); tip_final.append(tips_sp[nm]); continue
    bt = m13.get(nm)                        # ② crosswalk
    if bt and norm(bt) in tips_sp:
        how.append("crosswalk"); tip_final.append(tips_sp[norm(bt)]); continue
    how.append("miss"); tip_final.append(None)

d["tip"], d["match_how"] = tip_final, how
n = len(d); nd = (d.match_how == "direct").sum(); nc = (d.match_how == "crosswalk").sum()
print(f"[匹配] 直接 {nd} + crosswalk {nc} = {nd+nc}/{n}({(nd+nc)/n:.1%})")

ok = d[d.tip.notna()].copy()
dup = ok.tip.duplicated(keep=False)
if dup.any():   # 两个 BirdLife 种并到同一 BirdTree 种 → 留 |u| 较小者,记档
    print(f"[并名] {dup.sum()} 行映射到 {ok[dup].tip.nunique()} 个重复 tip,按 |u| 最小去重")
    ok = ok.loc[ok.assign(a=ok.u.abs()).sort_values("a").drop_duplicates("tip").index]
ok = ok.sort_values("tip")
ok[["tip", "u", "HWI", "log_m"]].to_csv(f"{BT}/pgls_data_matched.csv", index=False)
d[d.tip.isna()][["tip_label"]].to_csv(f"{BT}/unmatched.csv", index=False)
print(f"[产出] pgls_data_matched.csv {len(ok)} 种 · unmatched.csv {int(d.tip.isna().sum())} 种")

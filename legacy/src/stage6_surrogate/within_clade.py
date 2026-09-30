# -*- coding: utf-8 -*-
"""族内比较:每个科/属内部,飞得更好的一半腿是不是更短
用法: python within_clade.py [根目录,默认 .]
产出 outputs/phylo/within_科.csv · within_属.csv"""
import sys, numpy as np, pandas as pd
ROOT = sys.argv[1] if len(sys.argv) > 1 else "."
D = pd.read_csv(f"{ROOT}/data/birdtree/pgls_data_full.csv")
D = D[D.HWI.notna()].copy()

def within(df):
    """先回归掉体重,再按 HWI 高低对半比较腿长残差"""
    X = np.column_stack([np.ones(len(df)), df.log_m])
    e = df.u.values - X @ np.linalg.lstsq(X, df.u.values, rcond=None)[0]
    h = df.HWI.values > np.median(df.HWI.values)
    if h.sum() < 3 or (~h).sum() < 3: return None
    rh = df.HWI.values - X @ np.linalg.lstsq(X, df.HWI.values, rcond=None)[0]
    return np.median(e[h]) - np.median(e[~h]), float(np.corrcoef(rh, e)[0, 1])

for lvl, key, mn in (("科", "Family", 15), ("属", "genus", 6)):
    d = D.copy()
    if key == "genus": d["genus"] = d.tip.str.split("_").str[0]
    rows = []
    for g, s in d.groupby(key):
        if len(s) < mn: continue
        out = within(s)
        if out is None: continue
        rows.append(dict(unit=g, n=len(s), delta=out[0], r=out[1],
                         hwi_gap=s.HWI.max() - s.HWI.min()))
    R = pd.DataFrame(rows)
    neg = int((R.delta < 0).sum()); z = (neg - len(R)/2) / np.sqrt(len(R)/4)
    print(f"{lvl}级(每单元 ≥{mn} 种) {len(R)} 个 · 高HWI半腿更短 {neg}/{len(R)}"
          f" = {neg/len(R):.1%} · z={z:.1f} · Δ中位 {R.delta.median():+.3f}σ")
    R.to_csv(f"{ROOT}/outputs/phylo/within_{lvl}.csv", index=False)

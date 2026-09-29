# -*- coding: utf-8 -*-
"""E-R1d 数据准备:tip + logL(跗跖) + logm + 层位 guild。

logL 由 u 精确反演:u = (logL - (0.479 + 0.391*logm))/0.0784(历史基线,仅作反演常数,
不影响 M0/M1/M2 的比较——三个模型都在同一 logL 上拟合)。
guild = data/forstrat_dom.csv 主导觅食层位(≥50%),mixed 剔除。
"""
import csv

A0, B0, SIG = 0.479, 0.391, 0.0784
dom = {r['scientificNameStd'].replace(' ', '_'): r['dom_strat'].replace('ForStrat.', '')
       for r in csv.DictReader(open('data/forstrat_dom.csv'))}
out = []
for r in csv.DictReader(open('data/birdtree/pgls_data_matched.csv')):
    g = dom.get(r['tip'])
    if not g or g == 'mixed':
        continue
    u = float(r['u']); lm = float(r['log_m'])
    out.append((r['tip'], u*SIG + A0 + B0*lm, lm, g))
w = csv.writer(open('data/birdtree/pgls_guild_data.csv', 'w', newline=''))
w.writerow(['tip', 'logL', 'logm', 'guild']); w.writerows(out)
from collections import Counter
print(len(out), '种;分组:', dict(Counter(g for *_, g in out)))

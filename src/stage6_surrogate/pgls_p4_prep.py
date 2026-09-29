# -*- coding: utf-8 -*-
"""E-R1d 数据准备:tip + logL(跗跖) + logm + 层位 guild。自给自足:
缺 data/forstrat_dom.csv 时从 data/avonet_joined_all.csv 现场生成(主导层位≥50%)。

logL 由 u 精确反演:u = (logL - (0.479 + 0.391*logm))/0.0784(历史基线仅作反演常数,
不影响 M0/M1/M2 比较——三模型拟合同一 logL)。
"""
import csv, os

STR = ['ForStrat.watbelowsurf', 'ForStrat.wataroundsurf', 'ForStrat.ground',
       'ForStrat.understory', 'ForStrat.midhigh', 'ForStrat.canopy', 'ForStrat.aerial']

def build_dom():
    out = {}
    for r in csv.DictReader(open('data/avonet_joined_all.csv')):
        sp = r.get('scientificNameStd', '')
        if not sp or sp in out:
            continue
        try:
            vals = [float(r[c] or 0) for c in STR]
        except (ValueError, KeyError):
            continue
        if sum(vals) <= 0:
            continue
        out[sp] = STR[vals.index(max(vals))] if max(vals) >= 50 else 'mixed'
    return out

if os.path.exists('data/forstrat_dom.csv'):
    dom = {r['scientificNameStd']: r['dom_strat']
           for r in csv.DictReader(open('data/forstrat_dom.csv'))}
else:
    assert os.path.exists('data/avonet_joined_all.csv'), \
        '缺 data/avonet_joined_all.csv:请从 Windows 端 FFY/data/ 传到本机同路径'
    dom = build_dom()
    with open('data/forstrat_dom.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['scientificNameStd', 'dom_strat'])
        w.writerows(sorted(dom.items()))
    print('已生成 data/forstrat_dom.csv:', len(dom), '种')

A0, B0, SIG = 0.479, 0.391, 0.0784
dom = {k.replace(' ', '_'): v.replace('ForStrat.', '') for k, v in dom.items()}
out = []
for r in csv.DictReader(open('data/birdtree/pgls_data_matched.csv')):
    g = dom.get(r['tip'])
    if not g or g == 'mixed':
        continue
    u = float(r['u']); lm = float(r['log_m'])
    out.append((r['tip'], u*SIG + A0 + B0*lm, lm, g))
with open('data/birdtree/pgls_guild_data.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['tip', 'logL', 'logm', 'guild']); w.writerows(out)
from collections import Counter
print(len(out), '种;分组:', dict(Counter(g for *_, g in out)))

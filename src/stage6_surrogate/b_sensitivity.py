# -*- coding: utf-8 -*-
"""b 口径敏感性矩阵（2026-09-24 定口径的复现脚本）。

结论（论文口径判定，用户 2026-09-24 批准）：
  主口径 = 仅会飞水鸟5科 b=0.407±0.016 (n=203)。
  依据：
   ① 全鸟(会飞) log-log 有曲率：分段局部斜率 0.270(10-30g) → 0.384(0.3-1kg)
      → 0.182(3-12kg)，单一 b 无定义 —— 全鸟口径统计上不成立，不是"没选"。
   ② 1–12 kg 窗内全鸟 pooled b=0.451 是宗族混杂（分目 −0.015…0.660），辛普森悖论。
   ③ 水鸟口径窗稳定：全域 0.407 / 工程窗 1–12kg 0.396 / 雁形目 0.448，互相咬合。
   ④ 机制论据：水鸟腿=着陆承重器官（功能同源），雀形目腿=栖枝抓握器官。
   ⑤ 分目 b 的散布(0.23 猛禽…0.65 鹤形/犀鸟)本身是结果：腿的功能生态位决定
      标度 —— 支撑论文 R2 反事实叙事，做 F2 子图。
  0.391（含 10 个不会飞物种的历史工程口径）在 SI 交代，差异在 ±SE 内，
  已有实验不重做；G10 = b=0.407 回灌管线单种子确认（A100）。

用法： python3 b_sensitivity.py   （需 data/avonet_hwi.csv）
"""
import csv, math, re, os, sys
from collections import defaultdict

ROOT = os.path.join(os.path.dirname(__file__), '..', '..')

def load():
    src = open(os.path.join(os.path.dirname(__file__), 'allometry_flight.py'),
               encoding='utf8').read()
    ns = {}
    for name in ('FLIGHTLESS_ORDERS', 'FLIGHTLESS_GENERA',
                 'FLIGHTLESS_SPECIES', 'WATER_FAMILIES'):
        m = re.search(name + r'\s*=\s*(\{[^}]*\}|\[[^\]]*\])', src, re.S)
        ns[name] = eval(m.group(1))
    rows = csv.DictReader(open(os.path.join(ROOT, 'data', 'avonet_hwi.csv')))
    data = []
    for r in rows:
        try:
            L = float(r['Tarsus.Length']); mg = float(r['BodyMass.Value'])
        except (ValueError, KeyError):
            continue
        if L <= 0 or mg <= 0:
            continue
        sp = r['scientificNameStd']; genus = sp.split()[0]
        fl = (r['Order'] in ns['FLIGHTLESS_ORDERS']
              or genus in ns['FLIGHTLESS_GENERA']
              or sp in ns['FLIGHTLESS_SPECIES'])
        data.append(dict(sp=sp, x=math.log10(mg), y=math.log10(L),
                         fam=r['Family'], odr=r['Order'], fl=fl, m=mg/1e3))
    return data, ns

def ols(sub):
    n = len(sub)
    mx = sum(d['x'] for d in sub)/n; my = sum(d['y'] for d in sub)/n
    sxx = sum((d['x']-mx)**2 for d in sub)
    sxy = sum((d['x']-mx)*(d['y']-my) for d in sub)
    b = sxy/sxx
    sse = sum((d['y']-(my+b*(d['x']-mx)))**2 for d in sub)
    return b, math.sqrt(sse/(n-2)/sxx), n

def main():
    data, ns = load()
    W = set(ns['WATER_FAMILIES'])
    vol = [d for d in data if not d['fl']]
    wat = [d for d in data if d['fam'] in W]
    watv = [d for d in wat if not d['fl']]

    print("== 主矩阵 ==")
    for name, sub in [
        ("水鸟5科·全部(历史口径)", wat),
        ("水鸟5科·仅会飞(主口径)", watv),
        ("水鸟·会飞·1-12kg", [d for d in watv if 1 <= d['m'] <= 12]),
        ("全鸟·仅会飞", vol),
        ("全鸟·会飞·1-12kg", [d for d in vol if 1 <= d['m'] <= 12]),
    ]:
        b, se, n = ols(sub)
        print(f"  {name:24s} n={n:>5} b={b:.3f}±{se:.3f}")

    print("== 全鸟曲率（分质量段局部斜率）==")
    for lo, hi in [(0, .01), (.01, .03), (.03, .1), (.1, .3),
                   (.3, 1), (1, 3), (3, 12)]:
        sub = [d for d in vol if lo <= d['m'] < hi]
        if len(sub) > 20:
            b, se, n = ols(sub)
            print(f"  {lo:g}-{hi:g} kg: n={n} b={b:.3f}±{se:.3f}")

    print("== 1-12kg 窗内分目（n>=15）==")
    byo = defaultdict(list)
    for d in vol:
        if 1 <= d['m'] <= 12:
            byo[d['odr']].append(d)
    for o, sub in sorted(byo.items(), key=lambda kv: -len(kv[1])):
        if len(sub) >= 15:
            b, se, n = ols(sub)
            print(f"  {o:22s} n={n:>4} b={b:.3f}±{se:.3f}")

    print("== 水鸟口径内不会飞物种（历史口径混入的 10 种）==")
    for d in sorted((d for d in wat if d['fl']), key=lambda d: d['fam']):
        print(f"  {d['fam']:20s}{d['sp']}")

if __name__ == '__main__':
    main()

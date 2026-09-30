# -*- coding: utf-8 -*-
"""P4b · 鲁棒被动对照:有没有一副固定刚度的腿能一腿打全部工况?

P4 的 ①被动 是按参考工况(草地 1.2)设计的 —— 那是"中间值被动",不是"最坏情况被动"。
本实验把 P4 搜出的 16 组按工况最优 (κ踝,κ膝,τ) 全部当候选,每组跨全部 16 工况互评:
若某一组(多半是高速工况的最优)能覆盖 15–16/16,则"定制系列被动腿"路线成立,
主动/整定失去必要性;若最好也只有 ~10/16,则整定的价值坐实。

用法:  python src/stage10_v2/p4b_robust.py --p4 outputs/v2_p4/p4_twolevel.json --out outputs/v2_p4
"""
from __future__ import annotations
import argparse, json, os, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import physics_v2 as P
from factory_v2 import zeta_of_kc

GCAP, SMAX = 10 * 9.81, 0.024
BASE = {**P.SCEN_BIRD_X, "hip_damp_unified": True}


def _ev(a):
    x9, m, v0, kc = a
    r = P.eval_v2(tuple(x9), m, v0, kc=kc, zeta_c=zeta_of_kc(kc), npass=2, base=BASE)
    if r is None or r.get("fail"):
        return (None, False)
    ok, _ = P.feasible_v2(r, GCAP, SMAX)
    return (r["peak_a"] / 9.81, bool(ok))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p4", default="outputs/v2_p4/p4_twolevel.json")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default="outputs/v2_p4")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    R = json.load(open(a.p4))
    OUT = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for m, d in sorted(R.items(), key=lambda t: float(t[0])):
            mf = float(m)
            L1, r2, r3, tA, tK = d["geom"]; kh = d["kap_hip"]
            conds = [(c["kc"], c["v0"]) for c in d["conds"]]
            # 12kg 硬地1.8 是"要求物理不可能"格,评它但覆盖统计里剔除
            bad = [(1e6, 1.8)] if mf == 12.0 else []
            good_ci = [i for i, cd in enumerate(conds) if cd not in bad]
            # 候选 = 16 组按工况最优 + 参考被动
            cands = [dict(src=f"opt@kc={c['kc']:.0e},v0={c['v0']}", ka=c["active"]["ka"],
                          kk=c["active"]["kk"], tau=c["active"]["tau"]) for c in d["conds"]] \
                  + [dict(src="P4参考被动(草地1.2)", ka=d["passive_stiff"][0],
                          kk=d["passive_stiff"][1], tau=d["passive_stiff"][2])]
            jobs = [( [L1, r2, r3, cd_["ka"], cd_["kk"], kh, cd_["tau"], tA, tK],
                      mf, v0, kc)
                    for cd_ in cands for (kc, v0) in conds]
            print(f"[m={m}] {len(cands)} 候选 × {len(conds)} 工况 = {len(jobs)} 次", flush=True)
            res = list(ex.map(_ev, jobs, chunksize=1))
            rows = []
            for i, cd_ in enumerate(cands):
                rr = res[i * len(conds):(i + 1) * len(conds)]
                okg = [rr[j][0] for j in good_ci if rr[j][1] and rr[j][0]]
                rows.append(dict(**cd_, cover=sum(1 for j in good_ci if rr[j][1]),
                                 n=len(good_ci),
                                 g_med=(float(np.median(okg)) if okg else None),
                                 g_worst=(float(np.max(okg)) if okg else None),
                                 per=[(rr[j][0], rr[j][1]) for j in range(len(conds))]))
            rows.sort(key=lambda r: (-r["cover"], r["g_worst"] or 99))
            OUT[m] = dict(rows=rows, conds=conds, excluded=bad)
            # 理想整定的覆盖作参照
            ideal = sum(1 for i in good_ci if d["conds"][i]["active"]["ok"])
            print(f"  参照:按工况整定覆盖 {ideal}/{len(good_ci)}")
            print(f"  {'候选来源':<26}{'覆盖':>7}{'g中位':>7}{'g最坏':>7}")
            for r_ in rows[:6] + [r_ for r_ in rows if r_["src"].startswith("P4参考")]:
                print(f"  {r_['src']:<26}{r_['cover']:>4}/{r_['n']:<3}"
                      f"{(f'{r_[chr(103)+chr(95)+chr(109)+chr(101)+chr(100)]:.2f}' if r_['g_med'] else '—'):>7}"
                      f"{(f'{r_[chr(103)+chr(95)+chr(119)+chr(111)+chr(114)+chr(115)+chr(116)]:.2f}' if r_['g_worst'] else '—'):>7}")
    json.dump(OUT, open(os.path.join(a.out, "p4b_robust.json"), "w"),
              indent=1, ensure_ascii=False)
    print(f"[p4b] → {a.out}/p4b_robust.json")


if __name__ == "__main__":
    main()

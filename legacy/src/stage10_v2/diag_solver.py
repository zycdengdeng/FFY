# -*- coding: utf-8 -*-
"""诊断:E20 硬地 2.0 全零 —— 是物理不可能,还是求解器发散?

夜跑里 concrete2.0 的 19 个质量级可行率全是 0,且日志出现 DYNAMIC SOLVER FAILED。
两种解释必须分开,因为结论完全不同:
  (a) 物理不可能 —— 10g 从 v0 停下的最小行程 v0²/(2·g_cap) 已超预算 s_max。
      这是可以写进论文的**要求边界**,不是缺陷。
  (b) 数值发散  —— 9 维 + 高速 + 硬接触把积分器打崩。这是 bug,结果不可用。

判据:
  ① 先算理论最小行程(方波极限),看 (a) 是否成立;
  ② 对同一批设计,把步长逐级缩小重跑,看失败是否消失、峰值是否收敛。
     若缩步长后出现可行解 → (b);若始终不可行但**求解成功** → (a)。

用法:  python src/stage10_v2/diag_solver.py --out outputs/v2_diag
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
import physics_v2 as P
from bioprior import BioPrior
from factory_v2 import zeta_of_kc
from e17_emergent_b import load as load_cvae
from train_cvae import norm

GCAP_G, SMAX = 10.0, 0.024


def theory(v0, gcap_g=GCAP_G, smax=SMAX):
    """方波极限:恒定减速度 a=gcap 停下所需的最小行程,以及所需的行程利用效率。"""
    s_min = v0 ** 2 / (2 * gcap_g * 9.81)
    return s_min, s_min / smax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v22_e5_bio/cvae_r39.pt")
    ap.add_argument("--masses", default="5,12,30")
    ap.add_argument("--out", default="outputs/v2_diag")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    print("=== ① 理论最小行程(方波极限,与设计无关) ===")
    print(f"验收: g_cap = {GCAP_G:.0f} g, s_max = {SMAX*1000:.0f} mm\n")
    print(f"{'v0 (m/s)':>10}{'最小行程 mm':>14}{'需要的行程利用率':>18}   判定")
    for v0 in (0.6, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0):
        sm, eff = theory(v0)
        verdict = ("物理不可能" if eff > 1.0 else
                   "极限(>85%,方波才够)" if eff > 0.85 else "有余量")
        print(f"{v0:>10.1f}{sm*1000:>14.1f}{eff*100:>17.0f}%   {verdict}")
    print("\n注:方波是**理论上限** —— 真实缓冲器达不到 100% 的行程利用率。")
    print("   线性弹簧恰好是 50%;油气式实测 80–90%。所以 >85% 那一档实际就已不可达。\n")

    import torch
    model, meta = load_cvae(a.ckpt); pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
    lo, hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    BASE = {**P.SCEN_BIRD_X, "hip_damp_unified": True}

    def gen(m, v0, kc):
        c = np.array([np.log10(m), v0, np.log10(kc), GCAP_G * 9.81, SMAX]); torch.manual_seed(7)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, lo, hi), dtype=torch.float32), 64).numpy()
        return prior.expand(np.clip(u, 0, 1), m).mean(0)

    print("=== ② 步长收敛检验(硬地 k_c=1e6,v0=2.0 与 1.2 对照) ===")
    print("若缩步长后失败消失/出现可行解 → 数值问题;若求解成功但始终不可行 → 物理边界。\n")
    hs = [float(BASE.get("h", 2e-4)), 1e-4, 5e-5, 2e-5]
    print(f"{'m':>5}{'v0':>5}{'步长 h':>10}{'求解':>7}{'峰值g':>9}{'行程mm':>9}{'利用率':>8}  判定")
    OUT = []
    for m in [float(v) for v in a.masses.split(",")]:
        for v0 in (1.2, 2.0):
            x = gen(m, v0, 1e6)
            for h in hs:
                b = {**BASE, "h": h}
                r = P.eval_v2(tuple(x), m, v0, kc=1e6, zeta_c=zeta_of_kc(1e6),
                              npass=2, base=b)
                if r is None or r.get("fail"):
                    print(f"{m:>5.0f}{v0:>5.1f}{h*1e6:>9.0f}µs{'✗':>7}   "
                          f"求解失败: {(r or {}).get('fail','?')}")
                    OUT.append(dict(m=m, v0=v0, h=h, solved=False,
                                    fail=(r or {}).get("fail", "?")))
                    continue
                ok, why = P.feasible_v2(r, GCAP_G * 9.81, SMAX)
                st = r["leg_stroke_mm"]
                print(f"{m:>5.0f}{v0:>5.1f}{h*1e6:>9.0f}µs{'✓':>7}{r['peak_a']/9.81:>9.2f}"
                      f"{st:>9.1f}{theory(v0)[0]*1000/max(st,1e-9)*100:>7.0f}%  "
                      f"{'可行' if ok else '✗'+','.join(why)}")
                OUT.append(dict(m=m, v0=v0, h=h, solved=True, peak_g=r["peak_a"]/9.81,
                                stroke=st, ok=bool(ok), why=list(why)))
            print()
    json.dump(OUT, open(os.path.join(a.out, "diag_solver.json"), "w"),
              indent=1, ensure_ascii=False)

    print("=== 结论 ===")
    f20 = [o for o in OUT if o["v0"] == 2.0]
    solved = [o for o in f20 if o["solved"]]
    feas = [o for o in solved if o.get("ok")]
    conv = len(solved) == len(f20)
    if feas:
        print("⚠ v0=2.0 存在可行解 → 夜跑的全零至少部分是**采样/数值**问题,需重跑该格。")
    elif conv:
        print("✓ v0=2.0 全部**求解成功但不可行** → 是物理边界,不是 bug。")
        print("  可以据此说:在 10g/24mm 的验收下,硬地 2.0 m/s 超出可行域。")
    else:
        nf = len(f20) - len(solved)
        print(f"⚠ v0=2.0 有 {nf}/{len(f20)} 次求解失败 → 数值问题真实存在。")
        print("  建议:该工况用更小步长重跑,或把求解失败与不可行在统计里分列。")
    print(f"\n[diag] → {a.out}/diag_solver.json")


if __name__ == "__main__":
    main()

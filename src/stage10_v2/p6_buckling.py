# -*- coding: utf-8 -*-
"""P6 · 屈曲校核:我们的「造不出来」判据漏了什么。

现状的结构定尺只查两件事:① 材料强度(σ ≤ σ_y/SF);② 细长比几何上限(D ≤ 0.25·段长)。
**没有查屈曲。** 薄壁圆管(t = 0.1D)在冲击轴压下有两种失稳,都可能先于屈服发生:

  A 整体屈曲(Euler)   —— 细长杆整根压弯。σ_cr = π²E/(L_e/r_g)²
  B 局部屈曲(壳)      —— 管壁起皱。经典 σ_cr = E·t/(R√(3(1−ν²))),
                          但真实圆柱壳对初始缺陷极敏感,必须乘折减系数 γ。
                          本脚本用 NASA SP-8007:γ = 1 − 0.901(1 − e^(−φ)),φ = (1/16)√(R/t)。
  纯弯曲的局部屈曲允许应力略高于轴压(经验 ×1.3),这里保守取轴压值。

判据:若 min(σ_cr,Euler, σ_cr,local) < σ_allow = σ_y/SF,则**屈曲控制**,
现有定尺是不安全的 —— 这正是国际会议上最容易被问到的一条。

用法:  python src/stage10_v2/p6_buckling.py --out outputs/v2_p6
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
sys.path.insert(0, os.path.join(HERE, "..", "stage8_struct"))
import physics_v2 as P
from bioprior import BioPrior
from e8_struct import MATERIALS as _MATS   # (ρ, σ_y, E),项目单一真源
from factory_v2 import zeta_of_kc
from e17_emergent_b import load as load_cvae
from train_cvae import norm

GCAP, SMAX = 10 * 9.81, 0.024
NU = 0.35                      # 泊松比(cfnylon / al7075 都在 0.33–0.35)
T_OVER_D = 0.10                # 壁厚比,与 physics_v2 一致
K_EULER = 1.0                  # 有效长度系数:两端铰接保守取 1.0


def shell_knockdown(R_over_t):
    """NASA SP-8007 圆柱壳轴压折减系数(经典理论 × γ)。"""
    phi = np.sqrt(R_over_t) / 16.0
    return 1.0 - 0.901 * (1.0 - np.exp(-phi))


def check(D_mm, L_mm, E_Pa, sig_allow_Pa):
    """返回 (σ_cr_Euler, σ_cr_local, γ, 控制模式)。全部为 Pa。"""
    D = D_mm * 1e-3; L = L_mm * 1e-3
    t = T_OVER_D * D; Ro = D / 2.0; Ri = Ro - t
    A = np.pi * (Ro**2 - Ri**2)
    I = np.pi / 4.0 * (Ro**4 - Ri**4)
    rg = np.sqrt(I / A)                                   # 回转半径
    lam = K_EULER * L / rg                                # 长细比
    sig_e = np.pi**2 * E_Pa / lam**2                      # Euler 临界应力
    Rm = Ro - t / 2.0                                     # 中面半径
    gam = shell_knockdown(Rm / t)
    sig_l = gam * E_Pa * t / (Rm * np.sqrt(3 * (1 - NU**2)))
    gov = min(sig_e, sig_l, sig_allow_Pa)
    mode = ("屈服" if gov == sig_allow_Pa else
            ("整体屈曲" if gov == sig_e else "局部屈曲"))
    return sig_e, sig_l, gam, lam, Rm / t, mode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="outputs/v22_e5_bio/cvae_r39.pt")
    ap.add_argument("--masses", default="5,8,12,20,30")
    ap.add_argument("--out", default="outputs/v2_p6")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    import torch
    model, meta = load_cvae(a.ckpt); pr = meta["prior"]
    prior = BioPrior("bio", sigma=pr["sigma"], u_max=pr["u_max"], v21=True)
    lo, hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    BASE = {**P.SCEN_BIRD_X, "hip_damp_unified": True}

    def gen(m, v0, kc):
        c = np.array([np.log10(m), v0, np.log10(kc), GCAP, SMAX]); torch.manual_seed(7)
        with torch.no_grad():
            u = model.sample(torch.tensor(norm(c, lo, hi), dtype=torch.float32), 64).numpy()
        return prior.expand(np.clip(u, 0, 1), m).mean(0)

    MATS = {k: dict(E=_MATS[k][2], sy=_MATS[k][1]) for k in ("cfnylon", "al7075")}
    SF = P.SF
    OUT = []
    print(f"壁厚比 t/D = {T_OVER_D}  ·  安全系数 SF = {SF}  ·  ν = {NU}")
    print(f"折减系数按 NASA SP-8007;有效长度系数 K = {K_EULER}(两端铰接,保守)\n")
    for mat, mp in MATS.items():
        sig_allow = mp["sy"] / SF
        print(f"===== 材料 {mat}  E = {mp['E']/1e9:.0f} GPa  σ_y = {mp['sy']/1e6:.0f} MPa  "
              f"→ 许用 {sig_allow/1e6:.0f} MPa =====")
        print(f"{'m':>5}{'地面':>6}{'段':>5}{'D(mm)':>8}{'L(mm)':>8}{'R/t':>7}{'λ':>7}"
              f"{'γ':>7}{'σ_Euler':>10}{'σ_local':>10}   控制")
        for m in [float(v) for v in a.masses.split(",")]:
            for tn, kc in (("草地", 1e5), ("硬地", 1e6)):
                x = gen(m, 1.2, kc)
                r = P.eval_v2(tuple(x), m, 1.2, kc=kc, zeta_c=zeta_of_kc(kc),
                              npass=2, mat=mat, base=BASE)
                if r is None or r.get("fail"):
                    print(f"{m:>5.0f}{tn:>6}   落震失败 {r.get('fail') if r else '?'}"); continue
                L1 = x[0]; Ls = [L1, x[1] * L1, x[2] * L1]        # 跗跖/胫跗/股骨
                for j, nm in enumerate(("跗跖", "胫跗", "股骨")):
                    D = r["D_mm"][j]
                    se, sl, g, lam, rt, mode = check(D, Ls[j], mp["E"], sig_allow)
                    print(f"{m:>5.0f}{tn:>6}{nm:>5}{D:>8.1f}{Ls[j]:>8.1f}{rt:>7.1f}{lam:>7.1f}"
                          f"{g:>7.3f}{se/1e6:>9.0f}M{sl/1e6:>9.0f}M   "
                          f"{mode}{'  ⚠' if mode!='屈服' else ''}")
                    OUT.append(dict(mat=mat, m=m, terr=tn, seg=nm, D_mm=float(D),
                                    L_mm=float(Ls[j]), R_over_t=float(rt), lam=float(lam),
                                    gamma=float(g), sig_euler_MPa=float(se/1e6),
                                    sig_local_MPa=float(sl/1e6),
                                    sig_allow_MPa=float(sig_allow/1e6), governs=mode))
        print()
    json.dump(OUT, open(os.path.join(a.out, "p6_buckling.json"), "w"),
              indent=1, ensure_ascii=False)
    bad = [o for o in OUT if o["governs"] != "屈服"]
    print(f"===== 汇总 =====")
    print(f"共 {len(OUT)} 个段-工况组合,其中 **{len(bad)} 个由屈曲控制**(现有定尺未考虑)。")
    for mat in MATS:
        sub = [o for o in OUT if o["mat"] == mat]
        b = [o for o in sub if o["governs"] != "屈服"]
        if b:
            worst = min(b, key=lambda o: min(o["sig_euler_MPa"], o["sig_local_MPa"]))
            print(f"  {mat}: {len(b)}/{len(sub)} 由屈曲控制;最严重 {worst['m']:.0f}kg {worst['terr']}"
                  f" {worst['seg']} → {worst['governs']} 临界 "
                  f"{min(worst['sig_euler_MPa'],worst['sig_local_MPa']):.0f} MPa "
                  f"vs 许用 {worst['sig_allow_MPa']:.0f} MPa")
        else:
            print(f"  {mat}: 0/{len(sub)} 由屈曲控制 —— 屈服始终先到,现有判据安全。")
    print(f"\n[p6] → {a.out}/p6_buckling.json")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""v2.5 评价配置的唯一出处(2026-09-29,缺陷 D-2 登记后新增)。

D-2 物理配置不一致:三个"生成器无关"实验各用了一套物理,且都不等于生成器臂用的工厂物理:
  outputs/v25_bird_e18b (旧锚点走廊) : --v21 --foot bearing → 9 维,无 mu_from_ground,m 2–40
  outputs/g1b_e18b      (新锚点走廊) : 无参数 → v2.0 物理,7 维,foot=leg,m 0.5–120
  outputs/er3           (ER3 试跑)   : 10 维设计但 base=None → foot=leg,无统一髋阻尼,ζ_c=0.15(≠ζ(k_c)=0.215)
  工厂 factory_v2 --v25 --foot bearing --planar 1(生成器臂、G-0 对标): 第四套。
三者互比、或与生成器臂比,都混进了物理版本。此后所有生成器无关实验(E18c/ER3/ER4b)必须:
  1) 只从本模块取 prior / evaluate / gates / CONDS;
  2) 把 stamp() 写进输出文件;分析脚本读入时先比对 stamp,不一致就拒绝合并。
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)
import physics_v2 as P                      # noqa: E402
import bioprior as BP                       # noqa: E402
from factory_v2 import zeta_of_kc, lhs      # noqa: E402,F401

# 验收要求默认固定(与 e18b 相同);FFY_SMAX / FFY_GCAP_G 可覆盖做敏感性(stamp 会如实记录)
GCAP_G = float(os.environ.get("FFY_GCAP_G", 10.0))
SMAX = float(os.environ.get("FFY_SMAX", 0.024))

# 与 e18b_corridor_multi.CONDS 逐字相同
CONDS = {
    "concrete1.2": dict(kc=1.0e6, v0=1.2, label="硬地 k_c=1e6 · v0=1.2"),
    "turf1.2":     dict(kc=1.0e5, v0=1.2, label="草地 k_c=1e5 · v0=1.2"),
    "wetsand1.2":  dict(kc=5.0e4, v0=1.2, label="湿沙 k_c=5e4 · v0=1.2"),
    "concrete1.5": dict(kc=1.0e6, v0=1.5, label="硬地 k_c=1e6 · v0=1.5"),
    "turf2.0":     dict(kc=1.0e5, v0=2.0, label="草地 k_c=1e5 · v0=2.0"),
    "wetsand2.0":  dict(kc=5.0e4, v0=2.0, label="湿沙 k_c=5e4 · v0=2.0"),
}

# 与 run_v25_main.sh CONFIG=bird 的工厂调用逐项对应:
#   factory_v2.py --v25 --foot bearing --planar 1 --npass 2;Fr=0 → v_x=0(纯垂直,面内平动自由)
CFG = dict(
    physics="v2.5", v21=True, v25=True, ndim=10,
    foot="bearing", planar=True, mu_from_ground=True, hip_damp_unified=True,
    npass=2, v_x=0.0,
    material=P.MAT_DEFAULT,                       # 由 FFY_MATERIAL 决定,默认 cfnylon
    gcap_g=GCAP_G, smax=SMAX,
    reb_cap=float(P.REB_CAP), slip_frac=float(P.SLIP_FRAC), mu_sf=float(P.MU_SF),
    m_ref_kg=float(BP.M_REF_KG), u_max=float(BP.U_MAX),
)


def base():
    return {**P.SCEN_BIRD_X, "hip_damp_unified": True,
            "foot_mode": CFG["foot"], "mu_from_ground": True}


def prior(arm="bio"):
    return BP.BioPrior(arm, v21=True, v25=True)


def evaluate(x10, m, v0, kc):
    """与工厂 _eval_one 同一条调用;返回 physics_v2 的结果字典(可能含 fail)。"""
    return P.eval_v2(tuple(x10), float(m), float(v0), kc=float(kc), zeta_c=zeta_of_kc(kc),
                     npass=CFG["npass"], base=base(), v_x=CFG["v_x"], planar=CFG["planar"])


def gates(r):
    """四闸 + 回弹/打滑软闸(feasible_v2 默认参数)。返回 (ok, 违反项列表)。"""
    return P.feasible_v2(r, GCAP_G * 9.81, SMAX)


def stamp():
    return dict(CFG)


def same_stamp(s1, s2, keys=("physics", "v25", "foot", "planar", "mu_from_ground",
                             "hip_damp_unified", "npass", "material", "gcap_g", "smax",
                             "reb_cap", "slip_frac", "m_ref_kg", "u_max")):
    bad = [k for k in keys if s1.get(k) != s2.get(k)]
    return (not bad), bad


def check_meta(path):
    """与工厂/训练集 meta 比对可比对的字段;不一致抛错,缺字段跳过。"""
    import json
    d = json.load(open(path))
    pr = d.get("prior", {})
    checks = {"foot_mode": CFG["foot"], "planar": CFG["planar"], "v25": CFG["v25"],
              "u_dim": CFG["ndim"]}
    src = {**d, **{k: pr[k] for k in ("v25", "ndim", "u_max") if k in pr}}
    if "ndim" in src:
        src["u_dim"] = src["ndim"]
    bad = {k: (src[k], v) for k, v in checks.items() if k in src and src[k] != v}
    if bad:
        raise SystemExit(f"[evalcfg] 与 {path} 的配置不一致: {bad}")
    print(f"[evalcfg] 与 {path} 可比字段一致: {[k for k in checks if k in src]}")


if __name__ == "__main__":
    import json
    print(json.dumps(stamp(), ensure_ascii=False, indent=1))
    for c, d in CONDS.items():
        print(f"{c:<12} ζ_c={zeta_of_kc(d['kc']):.3f}")

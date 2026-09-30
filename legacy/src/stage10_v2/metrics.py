# -*- coding: utf-8 -*-
"""v4 单一口径源 —— 所有出图 / 出表 / 落盘的唯一入口。

立项背景（2026-09-22 核验）：
  一次核验查出三处口径混用 ——
    · `v3_grid_probe.py` 裸足分支落盘 `a_res_g`（已除 9.81），轮足分支落盘 `a_res`（m/s²）；
    · 三方对比图三根柱子来自三个不同模型 / 三个不同刹车配置；
    · 「载荷系数」与「峰值加速度」两个口径在图之间串味。
  根因不是粗心，是**没有制度**：每个数字都由写图的人临时从 raw dict 里取，
  没有任何东西强迫他说明这个数来自哪个模型、哪个工况、哪个配置。

本模块就是那个制度：

  1. **单位写进键名**。`a_res_g`（g）、`stroke_mm`（mm）、`Lstop_m`（m）、
     `omega_radps`（rad/s）。没有后缀的键一律拒收。
  2. **出处随数据走**。每条记录必带 `Prov(model, cond, cfg)`，落盘、画图、
     做表都带着；取数时可随时回答「这个数是什么模型、什么工况、什么配置」。
  3. **量纲自检**。数值落在物理上不可能的区间时直接抛错，而不是默默画出去。

用法：
    from metrics import Prov, canon, load_factor, MODEL

    p = Prov(model=MODEL.V4_PLANAR, cond=dict(m_kg=4.73, v0_mps=1.372,
             kc=8.68e5, Fr=0.0), cfg=dict(tau_max_Nm=2.0, om_th_radps=40.0))
    rec = canon(raw_from_eval, p)          # → 带单位后缀 + 出处的规范记录
    lf  = load_factor(rec)                 # → 峰值载荷系数 F/mg

自检：  python metrics.py
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Any

G = 9.81                      # m/s²，全项目唯一重力常数


# ──────────────────────────────────────────────────────── 模型标签
class MODEL:
    """物理模型标签。任何数字都必须声明它来自哪个模型 —— 不同模型的数不可直接同图。"""
    V25_RAIL = "v2.5-竖直滑轨-单腿"        # 竖直滑轨，水平力由约束反力承担
    V25_PLANAR = "v2.5-平面-单腿"          # 平面，单腿扛全重，水平力不抵消
    V3_PLANAR = "v3-平面-前后镜像双腿"      # 四腿四轮，水平力对称抵消
    V4_PLANAR = "v4-平面-左右双腿脚掌"      # 两腿四轮，侧视图中左右腿投影重合
    OLEO_IDEAL = "理想恒力油气支柱"          # 解析式，非仿真
    ALL = (V25_RAIL, V25_PLANAR, V3_PLANAR, V4_PLANAR, OLEO_IDEAL)


# ──────────────────────────────────────────────────────── 出处
@dataclass
class Prov:
    """(模型, 工况, 配置) 三元组 —— 纪律的载体。"""
    model: str
    cond: dict = field(default_factory=dict)   # 工况：m_kg / v0_mps / kc / Fr
    cfg: dict = field(default_factory=dict)    # 配置：设计与旋钮
    note: str = ""

    def __post_init__(self):
        if self.model not in MODEL.ALL:
            raise ValueError(f"未登记的模型标签：{self.model!r}；"
                             f"合法值 {MODEL.ALL}")

    def label(self) -> str:
        """一行人话出处，直接印到图上。"""
        c = self.cond
        bits = []
        if "m_kg" in c:
            bits.append(f"m={c['m_kg']:.2f} kg")
        if "v0_mps" in c:
            bits.append(f"v0={c['v0_mps']:.3f} m/s")
        if "Fr" in c:
            bits.append(f"Fr={c['Fr']:.2f}")
        if "kc" in c:
            bits.append(f"kc={c['kc']:.2e}")
        cfg = "，".join(f"{k}={v}" for k, v in sorted(self.cfg.items()))
        return f"{self.model}（{'，'.join(bits)}）" + (f" · {cfg}" if cfg else "")


# ──────────────────────────────────────────────────────── 单位契约
#  raw 键名 → (规范键名, 换算系数, 合理区间)
#  区间是「物理上不可能」的护栏，不是「不常见」的护栏；越界即抛错。
_SPEC: dict[str, tuple[str, float, tuple[float, float]]] = {
    # 加速度：raw 为 m/s²，规范为 g
    # 下界 0.5 g 是关键护栏：落震仿真里自由落体段就保证峰值 ≥1 g，
    # 若把「已经是 g 的数」再除一次 9.81（v3 事故的真实形态），会掉到 0.26 g 而被拦下。
    "a_res":          ("a_res_g",        1.0 / G, (0.5, 200.0)),
    "peak_a":         ("peak_a_g",       1.0 / G, (0.5, 200.0)),
    # 长度：raw 为 m，规范为 mm
    "stroke":         ("stroke_mm",      1e3,     (0.0, 2000.0)),
    "leg_stroke":     ("leg_stroke_mm",  1e3,     (0.0, 2000.0)),
    "sink":           ("sink_mm",        1e3,     (0.0, 2000.0)),
    "rebound":        ("rebound_mm",     1e3,     (0.0, 5000.0)),
    # 距离：raw 为 m，规范也为 m（量级大，mm 反而难读）
    "roll_dist_m":    ("roll_dist_m",    1.0,     (0.0, 1e4)),
    "stop_dist_m":    ("Lstop_m",        1.0,     (0.0, 1e4)),
    # 角度
    "pitch_max_deg":  ("pitch_max_deg",  1.0,     (0.0, 180.0)),
    "pitch_end_deg":  ("pitch_end_deg",  1.0,     (-180.0, 180.0)),
    # 速度
    "vx_end":         ("vx_end_mps",     1.0,     (-100.0, 100.0)),
    "wheel_spin_max": ("omega_max_radps", 1.0,    (0.0, 1e4)),
    # 质量
    "leg_mass_kg":    ("leg_mass_g",     1e3,     (0.0, 1e5)),
    "gear_mass_kg":   ("gear_mass_g",    1e3,     (0.0, 1e5)),
    "mass_frac":      ("mass_frac",      1.0,     (0.0, 10.0)),
    # 结构（已带单位）
    "D1_mm": ("D1_mm", 1.0, (0.0, 500.0)),
    "D2_mm": ("D2_mm", 1.0, (0.0, 500.0)),
    "D3_mm": ("D3_mm", 1.0, (0.0, 500.0)),
    "L1_mm": ("L1_mm", 1.0, (0.0, 5000.0)),
    "L2_mm": ("L2_mm", 1.0, (0.0, 5000.0)),
    "L3_mm": ("L3_mm", 1.0, (0.0, 5000.0)),
    # 布尔 / 计数（无量纲，原样过）
    "stopped":     ("stopped",     1.0, (0.0, 1.0)),
    "stop_extrap": ("stop_extrap", 1.0, (0.0, 1.0)),
    "struct_over": ("struct_over", 1.0, (0.0, 1.0)),
    "mass_over":   ("mass_over",   1.0, (0.0, 1.0)),
    "latch_f":     ("latch_f_s",   1.0, (-2.0, 100.0)),
    "latch_r":     ("latch_r_s",   1.0, (-2.0, 100.0)),
}

# 已经是规范键名的，直接放行（幂等：canon(canon(x)) == canon(x)）
_CANON_KEYS = {v[0] for v in _SPEC.values()}

# 明令禁止的裸键名 —— 一旦出现在规范记录里就是口径事故
_BANNED = {"a_res_g_g", "peak_g", "a_res", "peak_a", "leg_stroke", "stroke"}


class UnitError(ValueError):
    """单位 / 量纲契约被违反。"""


def canon(raw: dict, prov: Prov, strict: bool = True) -> dict:
    """把一次评价的 raw dict 转成规范记录。

    raw   : physics_vX.eval_* 的返回值（SI 原始单位）
    prov  : 出处三元组
    strict: True 时遇到未登记的键抛错（默认）；False 时原样保留并加 `_raw_` 前缀。

    返回：{规范键: 值, ..., "_prov": {...}}
    """
    if raw is None:
        return {"fail": "none", "_prov": asdict(prov)}
    out: dict[str, Any] = {}
    if raw.get("fail"):
        out["fail"] = raw["fail"]
    for k, v in raw.items():
        if k == "fail" or k.startswith("_"):        # _prov 等元数据原样跳过
            continue
        if k in _CANON_KEYS:                       # 已规范，幂等放行
            out[k] = v
            continue
        if k not in _SPEC:
            if strict:
                raise UnitError(
                    f"未登记的量 {k!r}。请先在 metrics._SPEC 里写明它的规范键名、"
                    f"换算系数与合理区间——不要绕过口径源直接取数。")
            out[f"_raw_{k}"] = v
            continue
        name, scale, (lo, hi) = _SPEC[k]
        if v is None:
            out[name] = None
            continue
        try:
            val = float(v) * scale
        except (TypeError, ValueError):
            out[name] = v
            continue
        if val == val and not (lo <= val <= hi):   # NaN 放过，越界抛错
            raise UnitError(
                f"{name} = {val:g} 越出物理可能区间 [{lo:g}, {hi:g}]。"
                f"十有八九是单位搞错了（出处：{prov.label()}）")
        out[name] = val
    out["_prov"] = asdict(prov)
    return out


def check_record(rec: dict) -> None:
    """落盘 / 画图前的最后一道闸。"""
    if "_prov" not in rec:
        raise UnitError("记录缺少 _prov 出处，拒绝使用。")
    bad = sorted(set(rec) & _BANNED)
    if bad:
        raise UnitError(f"记录里出现了禁用的裸键名 {bad} —— 这正是 v3 口径事故的形态。")
    for k, v in rec.items():
        if k.startswith("_") or v is None or not isinstance(v, (int, float)):
            continue
        if k in _CANON_KEYS:
            for nm, sc, (lo, hi) in _SPEC.values():
                if nm == k and v == v and not (lo <= float(v) <= hi):
                    raise UnitError(f"{k} = {v:g} 越界 [{lo:g}, {hi:g}]")
                if nm == k:
                    break


# ──────────────────────────────────────────────────────── 派生量
def load_factor(rec_or_g, vertical: bool = False) -> float:
    """峰值载荷系数 F/mg = 峰值加速度/g + 1。

    唯一定义，全项目只此一处。传入规范记录或直接传 g 值。
    vertical=True 用竖直分量 peak_a_g，False（默认）用合成 a_res_g。
    """
    if isinstance(rec_or_g, dict):
        key = "peak_a_g" if vertical else "a_res_g"
        if key not in rec_or_g or rec_or_g[key] is None:
            raise UnitError(f"记录里没有 {key}，无法算载荷系数")
        g_val = float(rec_or_g[key])
    else:
        g_val = float(rec_or_g)
    return g_val + 1.0


def oleo_load_factor(v0_mps: float, stroke_mm: float, eta: float = 0.85) -> float:
    """理想恒力油气支柱的载荷系数（解析，非仿真）。

    F = ½·m·v0²/(η·s) + m·g  ⇒  F/mg = v0²/(2·η·s·g) + 1
    η = 0.85 为油气典型缓冲效率（矩形恒力理想值为 1）。
    **这是一个打不过的下界，不是真实产品的数** —— 真实油气的力-行程曲线是峰状，
    油孔节流力随速度平方变化、气弹簧力随压缩非线性上升。引用时必须说明。
    """
    s = stroke_mm / 1e3
    if s <= 0:
        raise UnitError("行程必须为正")
    return v0_mps ** 2 / (2.0 * eta * s * G) + 1.0


def compare_table(recs: list[dict], keys: tuple[str, ...]) -> str:
    """把若干规范记录排成对照表，**每行强制带出处**。

    这是「三根柱子三个基准」事故的直接解药：出处不同的行会被一眼看见。
    """
    lines = []
    head = f"{'出处':<52}" + "".join(f"{k:>16}" for k in keys)
    lines.append(head)
    lines.append("-" * len(head))
    for r in recs:
        p = Prov(**{k: v for k, v in r["_prov"].items()})
        cells = []
        for k in keys:
            v = r.get(k)
            cells.append(f"{v:>16.3f}" if isinstance(v, (int, float)) else f"{str(v):>16}")
        lines.append(f"{p.label():<52}" + "".join(cells))
    models = {r["_prov"]["model"] for r in recs}
    if len(models) > 1:
        lines.append("")
        lines.append(f"⚠ 本表跨了 {len(models)} 个模型：{sorted(models)}")
        lines.append("  跨模型的数不可直接比大小，除非已做过等价性检查。")
    return "\n".join(lines)


def dump(recs: list[dict], fp: str) -> None:
    """落盘前逐条过闸。"""
    for r in recs:
        check_record(r)
    with open(fp, "w", encoding="utf8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)


# ──────────────────────────────────────────────────────── 自检
def _selftest() -> None:
    ok = 0

    # 1 单位换算正确
    p = Prov(MODEL.V4_PLANAR, dict(m_kg=4.7305, v0_mps=1.3718, Fr=0.0),
             dict(tau_max_Nm=2.0, om_th_radps=40.0))
    rec = canon(dict(a_res=24.7519, peak_a=24.7519, leg_stroke=0.0720), p)
    assert abs(rec["a_res_g"] - 2.5231) < 1e-3, rec
    assert abs(rec["leg_stroke_mm"] - 72.0) < 1e-6, rec
    ok += 1

    # 2 幂等：已规范的键再过一次不变
    assert canon(rec, p)["a_res_g"] == rec["a_res_g"]
    ok += 1

    # 3 载荷系数唯一定义
    assert abs(load_factor(rec) - 3.5231) < 1e-3
    ok += 1

    # 4 理想油气复核（与 e8_struct.strut_baseline 同式）
    assert abs(oleo_load_factor(1.3718, 89.0) - 2.267) < 2e-3
    assert abs(oleo_load_factor(1.3718, 77.24) - 2.462) < 2e-3
    ok += 1

    # 5a v3 事故的真实形态：把「已经是 g 的数」当 m/s² 再除一次
    try:
        canon(dict(a_res=2.5231), p)                # 少了一次 ×9.81
        raise AssertionError("重复除以 g 没有被拦下")
    except UnitError:
        ok += 1

    # 5b 反向错误：多乘一次 g
    try:
        canon(dict(a_res=24.75 * 9.81 * 9.81), p)
        raise AssertionError("越界值没有被拦下")
    except UnitError:
        ok += 1

    # 6 未登记的量必须被拦下
    try:
        canon(dict(some_new_thing=1.0), p)
        raise AssertionError("未登记键没有被拦下")
    except UnitError:
        ok += 1

    # 7 禁用裸键名被拦下
    try:
        check_record({"a_res": 2.5, "_prov": asdict(p)})
        raise AssertionError("裸键名没有被拦下")
    except UnitError:
        ok += 1

    # 8 未登记模型标签被拦下
    try:
        Prov("我随便写的模型")
        raise AssertionError("野模型标签没有被拦下")
    except ValueError:
        ok += 1

    # 9 跨模型对照表会自己报警
    p2 = Prov(MODEL.V25_RAIL, dict(m_kg=4.7305, v0_mps=1.3718, Fr=0.0))
    r2 = canon(dict(a_res=17.757, peak_a=14.853, leg_stroke=0.0891), p2)
    tbl = compare_table([rec, r2], ("a_res_g", "peak_a_g", "leg_stroke_mm"))
    assert "⚠ 本表跨了 2 个模型" in tbl
    ok += 1

    print(tbl)
    print(f"\n[metrics] 自检通过 {ok}/10")


if __name__ == "__main__":
    _selftest()

# -*- coding: utf-8 -*-
"""轮组质量模型 —— 由公开目录数据标定（2026-09-22）。

替换 v3 的目录级拍脑袋估计：
    旧： m_wheel = 44.4·r_w²  kg      ；m_brake+bearing = 0.020 + 0.008·τ_max + 0.015 kg
    问题：① 指数取 2，实测是 2.2–2.5；② 量级偏大 1.7–5 倍；③ 刹车项的 τ 系数
          建立在一个**超出地面摩擦上限 14–43 倍**的 τ 盒之上（见 friction_limited_tau）。

标定数据（全部为公开目录标称值，单轮/单件）：

  A. 车轮 —— DU-BRO 两条同厂同直径网格的序列，天然给出下界与上界
     · Super Lite（泡沫胎 + 尼龙轮毂）           下界
       44.45mm→4.4g  50.8→6.3  57.15→10.0  63.5→11.5  69.85→14.8  76.2→16.8
       拟合 m[g] = 3.195e-4·d[mm]^2.53   (R²=0.970)
     · Treaded Lightweight（橡胶胎面 + 轮毂）     上界
       44.45→12.8  50.8→16.0  57.15→25.5  63.5→29.4  69.85→35.0  76.2→42.0
       82.55→50.0  88.9→61.0
       拟合 m[g] = 2.752e-3·d[mm]^2.23   (R²=0.993，RMS 1.3 g)
       逐点偏差除一处外均在 ±3% 内；50.8→57.15 mm 那一跳偏 +8.8%/−11.3%，
       因为目录本身在此处换了轴径（175TL–200TL 用 1/8″，225TL 以上用 5/32″），
       是构造突变而非拟合失败。

  B. 单向轴承 —— HF 系列冲压外圈滚针离合器（INA / 通用件）
       HF0406 (4×8×6)   1.0 g      HF0612 (6×10×8)  3.0 g
       HF0812 (8×12×8)  3.5 g      HF1012 (10×14×10) 4.0 g
       HF1216 (12×18×18) 11.6 g
     旧模型给 15 g —— 高估 4–15 倍。

  C. 离心刹车 —— 无现成货架件，以 RC 甲醇机离心离合器类比（同为离心式摩擦机构）
       铝制 3–4 蹄总成（含飞轮）17–21.5 g，传递 2–3 N·m
       扣掉飞轮后蹄片 + 弹簧约 6–10 g
     我们需要的力矩只有 ~0.5 N·m（见下），但有结构下限（弹簧、销轴、紧固件）。

⚠ **关键标定：刹车力矩其实被地面摩擦卡死**
   地面能传的水平力 F ≤ μ·N。4.73 kg、μ=0.4 ⇒ F ≤ 18.6 N ⇒ 最大减速度 0.40 g。
   四轮分摊，每轮只需 τ = μ·N·r_w/4 ≈ 0.14 N·m；着地冲击 3.5 g 峰值时也只需 0.49 N·m。
   **v3 的盒 τ_max ∈ [2, 6] N·m 超出地面所能利用的 14–43 倍。**
   这正解释了相图里「τ 从 4 加到 6 几乎无收益」。
   ⇒ v4 的盒应为 τ_max ∈ [0.1, 1.0] N·m，刹车也因此可以做得轻得多。

   注意区分两件事：
     · 用来**给整机减速** —— 被 μN 卡死，τ > 0.5 N·m 纯属浪费
     · 用来**锁住足端防止外滑** —— 不被卡死，τ 越大锁得越死、着地冲击越高
   所以刹车做大在停车上毫无收益，却实打实抬高落震载荷。

自检：  python mass_model.py
"""
from __future__ import annotations

import numpy as np

G = 9.81

# ── A 车轮：m[g] = a · d[mm]^b ──────────────────────────────────
WHEEL_FOAM = (3.195e-4, 2.53)      # 泡沫胎 + 尼龙轮毂（下界）
WHEEL_TREAD = (2.752e-3, 2.23)     # 橡胶胎面 + 轮毂（上界，v4 默认）

# 轮毂加强系数：我们的轮毂要装单向轴承座 + 离心刹车鼓，比目录件重
HUB_FACTOR = {"乐观": 1.15, "基准": 1.35, "保守": 1.60}

# ── B 单向轴承：按轴径线性（HF 系列拟合）──────────────────────────
def bearing_mass_g(bore_mm: float) -> float:
    """HF 系列单向轴承质量（g）。拟合 HF0406/0612/0812/1012 四点。"""
    return float(np.clip(0.50 * bore_mm - 1.0, 0.8, 12.0))


def axle_bore_mm(d_wheel_mm: float) -> float:
    """轮径 → 轴径。DU-BRO 口径：44–51 mm 用 1/8″(3.2)，57–89 mm 用 5/32″(4.0)。
    我们要塞单向轴承，轴径取目录值上浮一档，并随轮径缓增。"""
    return float(np.clip(0.075 * d_wheel_mm + 1.0, 4.0, 10.0))


# ── C 离心刹车 ────────────────────────────────────────────────
BRAKE_BASE_G = {"乐观": 1.5, "基准": 2.5, "保守": 4.0}    # 弹簧/销轴/紧固件的结构下限
BRAKE_PER_NM_G = {"乐观": 2.5, "基准": 4.0, "保守": 7.0}  # 每 N·m 的摩擦件质量


def brake_mass_g(tau_max_Nm: float, tier: str = "基准") -> float:
    """离心刹车（蹄片 + 弹簧 + 刹车鼓附件）质量（g）。"""
    return BRAKE_BASE_G[tier] + BRAKE_PER_NM_G[tier] * float(tau_max_Nm)


# ── 组装 ──────────────────────────────────────────────────────
def wheel_assembly_g(r_w_m: float, tau_max_Nm: float, tier: str = "基准",
                     tire: str = "tread") -> dict:
    """单个「轮 + 单向轴承 + 离心刹车」总成质量（g），按 tier 给档。"""
    d_mm = 2.0 * r_w_m * 1e3
    a, b = WHEEL_TREAD if tire == "tread" else WHEEL_FOAM
    m_tire = a * d_mm ** b
    m_wheel = m_tire * HUB_FACTOR[tier]
    m_brg = bearing_mass_g(axle_bore_mm(d_mm))
    m_brk = brake_mass_g(tau_max_Nm, tier)
    return dict(d_mm=d_mm, tire_g=m_tire, wheel_g=m_wheel,
                bearing_g=m_brg, brake_g=m_brk,
                total_g=m_wheel + m_brg + m_brk)


def wheel_assembly_kg(r_w_m: float, tau_max_Nm: float, tier: str = "基准") -> float:
    return wheel_assembly_g(r_w_m, tau_max_Nm, tier)["total_g"] / 1e3


def legacy_v3_assembly_g(r_w_m: float, tau_max_Nm: float) -> float:
    """v3 的旧模型，仅供对比：44.4·r² kg + (0.020 + 0.008τ + 0.015) kg。"""
    return (44.4 * r_w_m ** 2 + 0.020 + 0.008 * tau_max_Nm + 0.015) * 1e3


# ── 地面摩擦对刹车力矩的钳制 ───────────────────────────────────
def friction_limited_tau(m_kg: float, r_w_m: float, n_wheels: int = 4,
                         mu: float = 0.40, load_factor: float = 1.0) -> float:
    """地面摩擦允许的每轮最大**有用**制动力矩（N·m）。

    超过这个值的刹车力矩对「让整机减速」毫无贡献（地面传不上去），
    但仍会通过锁死轮子抬高落震载荷。load_factor 为着地时的法向过载倍数。
    """
    return mu * load_factor * m_kg * G * r_w_m / n_wheels


# ── 自检 ──────────────────────────────────────────────────────
def _selftest():
    ok = 0
    # 1 拟合复现目录值（带胎面序列）
    cat = {44.45: 12.8, 57.15: 25.5, 76.2: 42.0, 88.9: 61.0}
    a, b = WHEEL_TREAD
    err = max(abs(a * d ** b - m) / m for d, m in cat.items())
    assert err < 0.12, err
    print(f"  [1] 带胎面拟合复现目录值，最大相对误差 {err:.1%}（集中在目录换轴径处）")
    ok += 1

    # 2 泡沫序列同样
    cat2 = {44.45: 4.4, 57.15: 10.0, 76.2: 16.8}
    a2, b2 = WHEEL_FOAM
    err2 = max(abs(a2 * d ** b2 - m) / m for d, m in cat2.items())
    assert err2 < 0.15, err2
    print(f"  [2] 泡沫拟合复现目录值，最大相对误差 {err2:.1%}")
    ok += 1

    # 3 轴承拟合复现 HF 目录
    for bore, m in ((4, 1.0), (6, 3.0), (8, 3.5), (10, 4.0)):
        assert abs(bearing_mass_g(bore) - m) < 1.1, (bore, bearing_mass_g(bore), m)
    print("  [3] HF 单向轴承拟合复现目录值（±1.1 g 内）")
    ok += 1

    # 4 摩擦钳制：v3 的盒确实超限一个量级以上
    tau_ok = friction_limited_tau(4.7305, 0.030)
    tau_peak = friction_limited_tau(4.7305, 0.030, load_factor=3.5)
    assert tau_ok < 0.2 and tau_peak < 0.6
    print(f"  [4] 摩擦钳制：静态 {tau_ok:.3f} N·m / 3.5g 峰值 {tau_peak:.3f} N·m"
          f" —— v3 盒 [2,6] 超出 {2/tau_ok:.0f}–{6/tau_ok:.0f} 倍")
    ok += 1

    # 5 新旧模型对照
    print("\n  单轮总成质量对照（g）：")
    print(f"  {'r_w':>5}{'τ':>6} | {'旧 v3':>8} | {'乐观':>7}{'基准':>7}{'保守':>7}")
    for r, t_old, t_new in ((0.020, 4.0, 0.5), (0.030, 4.0, 0.5),
                            (0.030, 2.0, 0.3), (0.045, 4.0, 1.0)):
        old = legacy_v3_assembly_g(r, t_old)
        new = [wheel_assembly_g(r, t_new, k)["total_g"] for k in ("乐观", "基准", "保守")]
        print(f"  {r*1e3:>5.0f}{t_new:>6.1f} | {old:>8.1f} | "
              f"{new[0]:>7.1f}{new[1]:>7.1f}{new[2]:>7.1f}")
    ok += 1

    d = wheel_assembly_g(0.030, 0.5)
    print(f"\n  基准 30 mm 半径 / τ=0.5 N·m 拆账："
          f"胎 {d['tire_g']:.1f} g → 含轮毂 {d['wheel_g']:.1f} g"
          f" + 轴承 {d['bearing_g']:.1f} g + 刹车 {d['brake_g']:.1f} g"
          f" = {d['total_g']:.1f} g")
    print(f"  旧模型同规格（τ=4）：{legacy_v3_assembly_g(0.030, 4.0):.1f} g"
          f"  ⇒ 高估 {legacy_v3_assembly_g(0.030,4.0)/d['total_g']:.1f} 倍")
    print(f"\n[mass_model] 自检通过 {ok}/5")


if __name__ == "__main__":
    _selftest()

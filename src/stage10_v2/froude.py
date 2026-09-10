# -*- coding: utf-8 -*-
"""水平触地速度的无量纲化：Froude 数 Fr <-> v_x。

v2.5 条件向量第 6 维放的是 Fr，不是 v_x：

    Fr = v_x / sqrt(g * L_ref(m))
    L_ref(m) = L1_med(m) * (1 + r2_med + r3_med)

L_ref 只依赖质量 m，而 m 已经在条件向量里，所以 Fr 完全由条件本身决定，
不引入设计变量、不破坏 cVAE 的条件结构。

为什么不直接用 v_x：同样 3 m/s 对 1 kg 和 30 kg 的机器不是一回事。
Fr 与质量解耦，v2.6 要扩到滑跑时只是把区间拉宽，参数化一行不用改。

用法：
    python froude.py                 # 打印换算表
    python froude.py --check         # 自检（与 bioprior 的 l1_center 对齐）
    from froude import fr_to_vx, vx_to_fr, l_ref
"""
import os, sys
import numpy as np

G = 9.81

# --- 与 bioprior 保持同源；导入不到就用字面量兜底（数值一致，见 --check）---
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from bioprior import BioPrior, R2_RANGE, R3_RANGE
    _BP = BioPrior(arm="bio", v21=True)
    _L1_MM = _BP.l1_center                                  # mm
    R2_MED = 0.5 * (R2_RANGE[0] + R2_RANGE[1])
    R3_MED = 0.5 * (R3_RANGE[0] + R3_RANGE[1])
    _SRC = "bioprior.BioPrior(arm='bio')"
except Exception as _e:                                     # pragma: no cover
    _A, _B = 0.47899383301187465, 0.39112926377807683
    _L1_MM = lambda m_kg: 10.0 ** (_A + _B * np.log10(np.asarray(m_kg, float) * 1000.0))
    R2_MED, R3_MED = 1.79, 1.06
    _SRC = "字面量兜底（bioprior 未导入：%s）" % _e

K_LEG = 1.0 + R2_MED + R3_MED          # = 3.850：L1 -> 整条腿的长度倍数


def l_ref(m_kg):
    """参考腿长（m）。仅依赖质量，取先验中位（u_L = 0, r2/r3 取盒中点）。"""
    return _L1_MM(m_kg) * 1e-3 * K_LEG


def v_scale(m_kg):
    """速度尺度 sqrt(g * L_ref)，单位 m/s。"""
    return np.sqrt(G * l_ref(m_kg))


def fr_to_vx(fr, m_kg):
    """Froude 数 -> 水平触地速度 (m/s)。"""
    return np.asarray(fr, float) * v_scale(m_kg)


def vx_to_fr(vx, m_kg):
    """水平触地速度 (m/s) -> Froude 数。"""
    return np.asarray(vx, float) / v_scale(m_kg)


# ---- 工厂采样：40% 精确取零，保住 Fr=0 的回归基线密度 ----
FR_MAX_MAIN = 2.0          # 主工厂上界（近垂直着陆）
FR_ZERO_FRAC = 0.40        # 精确取零的样本比例
FR_LADDER = (2., 4., 6., 9., 12., 15.)      # P9b 崩溃阶梯


def sample_fr(n, rng=None, fr_max=FR_MAX_MAIN, zero_frac=FR_ZERO_FRAC):
    """工厂用的 Fr 采样。

    zero_frac 的样本精确取 0（不是"接近 0"）——这样 v2.5 的结果才能和 v2.3
    逐条对比；否则加了一维之后哪儿变了都说不清。其余在 (0, fr_max] 均匀。
    """
    rng = rng or np.random.default_rng()
    fr = rng.uniform(0.0, fr_max, size=int(n))
    fr[rng.random(int(n)) < zero_frac] = 0.0
    return fr


def _table():
    print("速度尺度来源：%s" % _SRC)
    print("L_ref(m) = L1_med(m) * (1 + %.3f + %.3f) = L1_med * %.3f\n" % (R2_MED, R3_MED, K_LEG))
    cols = (0.5, 1.0, 2.0, 4.0, 6.0, 9.0, 12.0, 15.0)
    print("%6s %8s %9s %10s | %s" % ("m(kg)", "L1(mm)", "L_ref(m)", "sqrt(gL)",
                                     " ".join("Fr=%-4g" % c for c in cols)))
    print("-" * (46 + 7 * len(cols)))
    for m in (0.5, 1, 2, 5, 11.07, 15, 25, 30, 45):
        c = float(v_scale(m))
        print("%6.2f %8.1f %9.3f %10.3f | %s" % (
            m, float(_L1_MM(m)), float(l_ref(m)), c,
            " ".join("%-7.2f" % (f * c) for f in cols)))
    print("\n主工厂 Fr ∈ [0, %.1f]；P9b 阶梯 Fr ∈ %s" % (FR_MAX_MAIN, list(FR_LADDER)))


def _check():
    ok = True
    for m in (0.5, 1.0, 11.07, 30.0):
        fr = np.array([0.0, 0.37, 2.0, 15.0])
        rt = vx_to_fr(fr_to_vx(fr, m), m)
        if not np.allclose(rt, fr, atol=1e-12):
            print("往返失败 m=%g" % m); ok = False
    fr = sample_fr(200000, np.random.default_rng(0))
    z = (fr == 0.0).mean()
    print("采样自检：n=200000, Fr==0 占比 %.4f（目标 %.2f），max %.4f（上界 %.1f）"
          % (z, FR_ZERO_FRAC, fr.max(), FR_MAX_MAIN))
    if abs(z - FR_ZERO_FRAC) > 0.01 or fr.max() > FR_MAX_MAIN:
        ok = False
    print("往返换算：%s" % ("通过" if ok else "失败"))
    return 0 if ok else 1


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(_check())
    _table()

"""P2 · 质量条件生物先验 p(x|m):把 214 种水鸟的离散测量提升成连续流形。

v1 的第二处缺陷(E14 §B2):工况 LHS 与设计 LHS 相互独立,训练集里 m 与形态的
互信息为 0。修法是让**设计盒子本身随 m 平移**,平移量由异速生长律给定:

    log10(L1[mm]) = a + b·log10(m[g]) + σ·u_L        u_L ∈ [-U_MAX, +U_MAX]

网络看到的永远是无量纲的 u(与 m 无关),m 只出现在展开式里。因此:
  · cVAE 结构一行不用改,输入盒子仍是 [0,1]^7;
  · "条件生成"第一次有东西可条件化——同一个 u 在不同 m 下是不同的物理设计。

**消融就是换一个 b**(《质量耦合改造方案_v2.md》§P3),四个臂都是文献里有名有姓的
零假设,盒子形状/体积/散布完全相同,只差一个标量斜率——"你是不是挑了个坏盒子"
这类质疑结构性消失:

    bio     b = 0.391   AVONET 214 种水鸟实测(正向异速生长)
    geo     b = 1/3     几何相似(等比例放大)
    elastic b = 1/4     McMahon 弹性相似(抗屈曲优先)
    none    b = 0       无尺度知识(≈ v1 的边缘盒,但散布同为 σ)

四条线都强制通过同一个锚点(工作区间的几何平均质量处),所以它们只差**倾斜**,
不差整体大小——否则比较的就不是"标度指数"而是"谁的腿更长"。
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# AVONET 拟合(src/stage6_surrogate/allometry_data.py 产出)
FIT_JSON = os.path.join(HERE, "..", "..", "outputs", "bird_pareto",
                        "avonet_allometry.json")
FIT_FALLBACK = dict(a=0.47899383301187465, b=0.39112926377807683,
                    ci95=0.030063514185437547, r2=0.754128606756704, n=214)

U_MAX = 2.5            # 条件盒子取 ±2.5σ 的预测区间(四臂共用,消融中固定不变)
M_REF_KG = np.sqrt(1.0 * 12.0)     # 锚点 = 工作区间 [1,12] kg 的几何平均
SD_LOG_M = 0.35        # 样本 log10(体重 g) 的标准差;仅用于反推 σ,见 sigma_of_fit

# r2/r3 的边缘范围(Watanabe 2017 实测,与 v1 一致);默认无质量趋势,见文末 TODO
R2_RANGE = (1.49, 2.09)
R3_RANGE = (0.84, 1.28)
KAP_RANGE = [(1.5, 8.0), (1.5, 8.0), (6.0, 32.0)]      # 踝 / 膝 / 髋
ZETA_RANGE = (0.01, 0.10)

ARMS = {"bio": None, "geo": 1.0 / 3.0, "elastic": 0.25, "none": 0.0}
# 观测到的 L1 全距(v1 边缘盒,101 种 1–12 kg 水鸟实测)。条件盒子在质量两端会
# 略微超出它——因为大鸟样本稀疏,大残差没被采到,不是拟合有问题。默认**不裁剪**
# (裁剪会让四臂盒子形状不再相同,毁掉消融的干净性);clip=True 仅供 rebuttal 用。
L1_OBSERVED = (33.0, 121.0)


def load_fit(path=FIT_JSON, key="waterbirds_all"):
    try:
        return json.load(open(path))["fits"][key]
    except Exception:
        return dict(FIT_FALLBACK)


def sigma_of_fit(fit, sd_log_m=SD_LOG_M):
    """由 (b, ci95, n) 反推残差标准差 σ(log10 单位)。

    简单回归恒等式:SE(b) = σ_resid / (σ_x·√(n−1)),ci95 = 1.96·SE(b)。
    (已用 r² 交叉验证:SE(b) = b·√(1−r²)/(|r|·√(n−1)) 与报告的 ci95 吻合到 4 位。)
    唯一需要外部输入的是样本自变量的散布 σ_x = SD_LOG_M,由该样本的体重跨度
    (约 0.13–12 kg)估得 ≈0.35。**拿到原始数据后应直接算残差 σ 覆盖此估计。**
    """
    se_b = fit["ci95"] / 1.96
    return float(se_b * sd_log_m * np.sqrt(max(fit["n"] - 1, 1)))


class BioPrior:
    """质量条件设计先验。expand: [0,1]^7 × m → 物理设计;contract 为其逆。"""

    def __init__(self, arm="bio", fit=None, sigma=None, u_max=U_MAX,
                 m_ref_kg=M_REF_KG, clip=False):
        self.arm = arm
        self.clip = bool(clip)
        self.fit = fit or load_fit()
        self.sigma = float(sigma if sigma is not None else sigma_of_fit(self.fit))
        self.u_max = float(u_max)
        self.b = self.fit["b"] if ARMS.get(arm) is None else float(ARMS[arm])
        # 锚定:四臂在 m_ref 处给出同一条腿长,只有斜率不同
        lref = self.fit["a"] + self.fit["b"] * np.log10(m_ref_kg * 1000.0)
        self.a = float(lref - self.b * np.log10(m_ref_kg * 1000.0))
        self.L1_ref_mm = float(10 ** lref)

    # ---------------------------------------------------------------- 中心与范围
    def l1_center(self, m_kg):
        return 10.0 ** (self.a + self.b * np.log10(np.asarray(m_kg, float) * 1000.0))

    def l1_range(self, m_kg):
        c = self.l1_center(m_kg)
        f = 10.0 ** (self.sigma * self.u_max)
        return c / f, c * f

    # ---------------------------------------------------------------- 展开 / 收缩
    def expand(self, u01, m_kg):
        """u01 ∈ [0,1]^7(网络输出,与 m 无关)→ 物理设计 x7。支持批量。"""
        u = np.atleast_2d(np.asarray(u01, float))
        m = np.broadcast_to(np.asarray(m_kg, float).reshape(-1, 1), (u.shape[0], 1))
        uL = (2.0 * u[:, 0:1] - 1.0) * self.u_max
        L1 = 10.0 ** (self.a + self.b * np.log10(m * 1000.0) + self.sigma * uL)
        if self.clip:
            L1 = np.clip(L1, *L1_OBSERVED)
        cols = [L1, self._lin(u[:, 1:2], R2_RANGE), self._lin(u[:, 2:3], R3_RANGE)]
        for j, rg in enumerate(KAP_RANGE):
            cols.append(self._lin(u[:, 3 + j:4 + j], rg))
        cols.append(self._lin(u[:, 6:7], ZETA_RANGE))
        x = np.concatenate(cols, 1)
        return x[0] if np.ndim(u01) == 1 else x

    def contract(self, x7, m_kg):
        """物理设计 → u01,用于把实测/历史设计编码进网络输入空间。"""
        x = np.atleast_2d(np.asarray(x7, float))
        m = np.broadcast_to(np.asarray(m_kg, float).reshape(-1, 1), (x.shape[0], 1))
        uL = (np.log10(x[:, 0:1]) - self.a - self.b * np.log10(m * 1000.0)) / self.sigma
        cols = [0.5 * (uL / self.u_max + 1.0),
                self._inv(x[:, 1:2], R2_RANGE), self._inv(x[:, 2:3], R3_RANGE)]
        for j, rg in enumerate(KAP_RANGE):
            cols.append(self._inv(x[:, 3 + j:4 + j], rg))
        cols.append(self._inv(x[:, 6:7], ZETA_RANGE))
        u = np.concatenate(cols, 1)
        return u[0] if np.ndim(x7) == 1 else u

    @staticmethod
    def _lin(u, rg):
        return rg[0] + (rg[1] - rg[0]) * u

    @staticmethod
    def _inv(v, rg):
        return (v - rg[0]) / (rg[1] - rg[0])

    def outside_observed(self, m_lo=1.0, m_hi=12.0, n=4000, seed=0):
        """诊断:条件盒子里有多大比例落在实测全距 [33,121] mm 之外。

        这个数要在论文里报出来,而不是靠裁剪把它藏起来。"""
        rng = np.random.default_rng(seed)
        m = 10 ** rng.uniform(np.log10(m_lo), np.log10(m_hi), n)
        u = rng.random((n, 7))
        L1 = self.expand(u, m)[:, 0]
        return float(np.mean((L1 < L1_OBSERVED[0]) | (L1 > L1_OBSERVED[1])))

    def describe(self):
        return dict(arm=self.arm, a=self.a, b=self.b, sigma=self.sigma,
                    u_max=self.u_max, L1_ref_mm=self.L1_ref_mm, clip=self.clip,
                    outside_observed=self.outside_observed(),
                    fit_n=self.fit["n"], fit_r2=self.fit.get("r2"))


if __name__ == "__main__":
    fit = load_fit()
    sg = sigma_of_fit(fit)
    print(f"AVONET 拟合: log10(L1_mm) = {fit['a']:.4f} + {fit['b']:.4f}·log10(m_g)"
          f"   r²={fit['r2']:.3f}  n={fit['n']}")
    print(f"反推残差 σ = {sg:.4f}(log10),即 1σ ≈ ×{10**sg:.3f} / ÷{10**sg:.3f}")
    print(f"条件盒子取 ±{U_MAX}σ;锚点 m_ref = {M_REF_KG:.3f} kg\n")
    print(f"{'臂':<10}{'b':>8}{'L1@1kg (mm)':>22}{'L1@3.46kg':>20}{'L1@12kg (mm)':>22}")
    for arm in ARMS:
        p = BioPrior(arm)
        rows = []
        for mk in (1.0, float(M_REF_KG), 12.0):
            lo, hi = p.l1_range(mk)
            rows.append(f"{lo:5.1f}–{hi:5.1f} ({p.l1_center(mk):5.1f})")
        print(f"{arm:<10}{p.b:>8.3f}" + "".join(f"{r:>22}" for r in rows))
    print(f"\nv1 的边缘盒(对照): 33.0–121.0 mm,与 m 无关")
    print("\n落在实测全距之外的比例(诊断,不裁剪):")
    for arm in ARMS:
        print(f"  {arm:<10}{BioPrior(arm).outside_observed()*100:5.1f}%")
    p = BioPrior("bio")
    u = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
    for mk in (1.0, 5.0, 12.0):
        x = p.expand(u, mk)
        back = p.contract(x, mk)
        print(f"  往返自检 m={mk:>4g}kg: x=[{x[0]:.2f}, {x[1]:.3f}, {x[2]:.3f}, ...]"
              f"  |u-u'|max={np.max(np.abs(back - u)):.2e}")

"""6.6 真实解剖数据标度律:AVONET 跗跖骨长 × EltonTraits 体重.

数据源(均为公开发表数据集):
- AVONET (Tobias et al. 2022, Ecology Letters 25:581-597):
  9 万余条标本级测量,含 Tarsus.Length(跗跖骨,mm)。
- EltonTraits 1.0 (Wilman et al. 2014, Ecology 95:2027):
  10,009 种鸟的体重 BodyMass.Value(g)+ 科/目分类。
联接键:scientificNameStd(traitdata 包统一学名)。

做什么:
1) AVONET 按物种取跗跖骨长中位数(标本级->物种级);
2) 联 EltonTraits 体重与科名;
3) 筛水面着陆类群(雁鸭科等,见 WATER_FAMILIES);
4) log10-log10 OLS 回归 -> 实测标度指数 b(带 95% CI);
5) 与模型预测(联合模型 b≈0.45,κ 敏感性 0.37-0.49)对照,出图。

用法: python src/stage6_surrogate/allometry_data.py \
        --avonet /tmp/traitdata/data/avonet.rda \
        --elton  /tmp/traitdata/data/elton_birds.rda \
        --out    outputs/bird_pareto
"""
import argparse
import json
import os

import numpy as np

# 习惯性水面着陆的科(降落时以脚触水滑行/刹车):
# Anatidae 雁鸭科(鸭/雁/天鹅 = Duong 数据主力), Gaviidae 潜鸟科,
# Podicipedidae 䴙䴘科, Pelecanidae 鹈鹕科, Phalacrocoracidae 鸬鹚科.
WATER_FAMILIES = {
    "Anatidae": "雁鸭科",
    "Gaviidae": "潜鸟科",
    "Podicipedidae": "䴙䴘科",
    "Pelecanidae": "鹈鹕科",
    "Phalacrocoracidae": "鸬鹚科",
}

# 我们此前 4 物种近似点(总腿长 mm, 体重 kg)——报告追加5 所用
APPROX4 = {"Duck": (1.2, 45.0), "Goose": (4.0, 85.0),
           "Swan": (10.0, 123.0), "Pelican": (10.5, 115.0)}


def ols_loglog(m_g, L_mm):
    """log10 L = a + b log10 m; 返回 b, 95%CI, a, R2."""
    x = np.log10(np.asarray(m_g, float))
    y = np.log10(np.asarray(L_mm, float))
    n = len(x)
    X = np.column_stack([np.ones(n), x])
    beta, res, *_ = np.linalg.lstsq(X, y, rcond=None)
    a, b = beta
    yhat = X @ beta
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    # slope 标准误
    s2 = ss_res / (n - 2)
    se_b = float(np.sqrt(s2 / np.sum((x - x.mean()) ** 2)))
    ci = 1.96 * se_b
    return dict(b=float(b), ci95=float(ci), a=float(a), r2=float(r2), n=n)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--avonet", default="/tmp/traitdata/data/avonet.rda")
    ap.add_argument("--elton", default="/tmp/traitdata/data/elton_birds.rda")
    ap.add_argument("--out", default="outputs/bird_pareto")
    ap.add_argument("--min_specimens", type=int, default=2,
                    help="物种至少几条标本测量才纳入")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import pyreadr
    av = pyreadr.read_r(args.avonet)
    av = av[list(av.keys())[0]]
    el = pyreadr.read_r(args.elton)
    el = el[list(el.keys())[0]]

    # 1) 标本级 -> 物种级(中位数抗离群)
    av = av.dropna(subset=["Tarsus.Length", "scientificNameStd"])
    g = av.groupby("scientificNameStd")["Tarsus.Length"]
    sp = g.median().to_frame("tarsus_mm")
    sp["n_spec"] = g.size()
    sp = sp[sp["n_spec"] >= args.min_specimens]

    # 2) 联体重 + 科
    el2 = el.dropna(subset=["BodyMass.Value", "scientificNameStd"])
    el2 = el2.set_index("scientificNameStd")[["BodyMass.Value", "Family", "Order"]]
    j = sp.join(el2, how="inner").dropna(subset=["BodyMass.Value", "Family"])
    j = j[j["BodyMass.Value"] > 0]

    # 3) 水面着陆类群
    w = j[j["Family"].isin(WATER_FAMILIES)].copy()

    results = {"source": {
        "tarsus": "AVONET (Tobias et al. 2022 Ecology Letters), specimen-level Tarsus.Length, species median",
        "mass": "EltonTraits 1.0 (Wilman et al. 2014 Ecology), BodyMass.Value",
        "join_key": "scientificNameStd (traitdata R package)",
        "min_specimens": args.min_specimens,
    }, "fits": {}}

    # 全部水鸟合并
    fit_all = ols_loglog(w["BodyMass.Value"], w["tarsus_mm"])
    results["fits"]["waterbirds_all"] = {**fit_all,
                                         "families": sorted(w["Family"].unique().tolist())}
    # 分科
    for fam in WATER_FAMILIES:
        sub = w[w["Family"] == fam]
        if len(sub) >= 5:
            results["fits"][fam] = ols_loglog(sub["BodyMass.Value"], sub["tarsus_mm"])

    # 对照:全鸟类(所有科)——看水鸟是否特殊
    fit_birds = ols_loglog(j["BodyMass.Value"], j["tarsus_mm"])
    results["fits"]["all_birds"] = fit_birds

    # 4 物种近似点的 b(总腿长)——报告口径核对
    m4 = [v[0] * 1000 for v in APPROX4.values()]  # kg->g
    L4 = [v[1] for v in APPROX4.values()]
    results["fits"]["approx4_total_leg"] = ols_loglog(m4, L4)

    # 打印摘要
    print(f"joined species total: {len(j)}, waterbirds: {len(w)}")
    for k, v in results["fits"].items():
        fams = "" if "families" not in v else f"  {v['families']}"
        print(f"{k:28s} b={v['b']:.3f} ±{v['ci95']:.3f}  R2={v['r2']:.3f}  n={v['n']}{fams}")

    # 物种明细存档
    w_out = w.reset_index()[["scientificNameStd", "Family", "tarsus_mm",
                             "BodyMass.Value", "n_spec"]]
    w_out.to_csv(os.path.join(args.out, "avonet_waterbirds.csv"), index=False)
    with open(os.path.join(args.out, "avonet_allometry.json"), "w") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "WenQuanYi Zen Hei",
                                       "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    colors = {"Anatidae": "#1f77b4", "Gaviidae": "#2ca02c",
              "Podicipedidae": "#9467bd", "Pelecanidae": "#d62728",
              "Phalacrocoracidae": "#8c564b"}
    for fam, cn in WATER_FAMILIES.items():
        sub = w[w["Family"] == fam]
        ax.scatter(sub["BodyMass.Value"] / 1000, sub["tarsus_mm"], s=18,
                   alpha=0.65, c=colors[fam], label=f"{cn} n={len(sub)}")
    mm = np.logspace(np.log10(w["BodyMass.Value"].min()),
                     np.log10(w["BodyMass.Value"].max()), 50)
    fit = results["fits"]["waterbirds_all"]
    ax.plot(mm / 1000, 10 ** fit["a"] * mm ** fit["b"], "k-", lw=2,
            label=f"实测拟合 b={fit['b']:.2f}±{fit['ci95']:.2f}")
    # 模型预测带:联合模型 κ∈[3,6] -> b∈[0.37,0.49],锚定拟合线中点
    m_mid = np.sqrt(mm[0] * mm[-1])
    L_mid = 10 ** fit["a"] * m_mid ** fit["b"]
    for b_model, ls in [(0.37, ":"), (0.45, "--"), (0.49, ":")]:
        ax.plot(mm / 1000, L_mid * (mm / m_mid) ** b_model, ls, c="crimson",
                lw=1.6, label=f"模型 b={b_model}" if b_model == 0.45 else None)
    ax.plot([], [], ":", c="crimson", lw=1.6, label="模型 κ 敏感带 0.37–0.49")
    # 几何相似对照
    ax.plot(mm / 1000, L_mid * (mm / m_mid) ** (1 / 3), "-.", c="gray", lw=1.4,
            label="几何相似 b=1/3")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("体重 m (kg)")
    ax.set_ylabel("跗跖骨长 (mm)")
    ax.set_title(f"AVONET 水面着陆鸟类 {len(w)} 种:跗跖骨长-体重标度律")
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fp = os.path.join(args.out, "avonet_allometry.png")
    fig.savefig(fp, dpi=150)
    print("saved:", fp)


if __name__ == "__main__":
    main()

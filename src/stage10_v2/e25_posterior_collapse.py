# -*- coding: utf-8 -*-
"""E25 · 后验塌缩量化（A 组，零仿真，纯推理）。

动机：待办 E1 —— 同条件下采 216 次，7 维中位 CV 只有 0.21%，即**解码器忽略了 z**。
这条不是要"修"，是要**定口径**：赵老师 12:16 已经替我们说过一次
「it is one solution, not necessarily the best one」，PPT 和论文得跟这句对齐。

三个互相独立的诊断（都不需要仿真）：

  ① 每维隐变量的 KL（编码器侧，标准做法）
     KL_j = E_x[ 0.5·(mu_j² + e^{lv_j} − lv_j − 1) ]，单位 nat。
     KL_j < 0.01 → 该维是死的；全死 = 完全塌缩。
     同时报 Burda 的 active-units：Var_x(mu_j) > 0.01 记为活。

  ② 解码器对 z 的敏感度（解码器侧）
     固定条件采 nz 个 z，看 9 维输出的散布 σ_z；
     与"跨条件的散布" σ_c 比。σ_z ≪ σ_c → 输出完全由条件决定，z 是摆设。

  ③ 与数据自身的多样性比（这一条最关键）
     训练集里**同一批相近条件**下有多少种 u？σ_data。
     若 σ_z ≪ σ_data，说明模型丢掉的是数据里本来就有的多样性，不是数据本来就窄。

  ④ 直接复现待办 E1 的口径（物理 x 空间的 CV），看那个 0.21% 还在不在。

用法：
    python src/stage10_v2/e25_posterior_collapse.py --fig
    python src/stage10_v2/e25_posterior_collapse.py --ckpt outputs/v23_e5_bio/cvae_r12.pt
"""
import argparse, json, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "stage7_generative"))
from train_cvae import CVAE, norm                                      # noqa: E402
from bioprior import BioPrior                                          # noqa: E402
from agroup_io import setup_font, CRIM, STEEL, GREEN                   # noqa: E402

DIMS9 = ["L1", "r2", "r3", "κ踝", "κ膝", "κ髋", "τ", "θ_A", "θ_K"]
KL_DEAD, AU_TH = 0.01, 0.01


def load_ckpt(fp):
    ck = torch.load(fp, map_location="cpu", weights_only=False)
    meta = ck["meta"]
    m = CVAE(xd=ck["xd"], cd=len(meta["c_lo"]), z=ck["zdim"])
    m.load_state_dict(ck["state"]); m.eval()
    return m, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=os.path.join(ROOT, "outputs/v23_e5_bio/cvae_r12.pt"))
    ap.add_argument("--data", default=os.path.join(ROOT, "outputs/v23_data_bio/dataset.npz"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--nz", type=int, default=216, help="每个条件采多少个 z")
    ap.add_argument("--ncond", type=int, default=64, help="抽多少组条件")
    ap.add_argument("--knn", type=int, default=40, help="③ 里取多少个最近邻当作「同条件」")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--e1-cv", type=float, default=0.21,
                    help="待办 E1 报告的中位 CV（%%），用于对照")
    ap.add_argument("--fig", action="store_true")
    args = ap.parse_args()
    out_dir = args.out or os.path.dirname(args.ckpt)

    model, meta = load_ckpt(args.ckpt)
    d = np.load(args.data)
    C, U = d["C_tr"], d["U_tr"]
    c_lo, c_hi = np.array(meta["c_lo"]), np.array(meta["c_hi"])
    Cn = norm(C, c_lo, c_hi)
    nd = U.shape[1]
    names = DIMS9[:nd] if nd <= len(DIMS9) else ["u%d" % i for i in range(nd)]
    print("[e25] %s  隐维 %d  设计维 %d  训练对 %d"
          % (os.path.basename(args.ckpt), model.zdim, nd, len(U)))

    # ---------------- ① 编码器侧：每维 KL ----------------
    with torch.no_grad():
        e = model.enc(torch.cat([torch.tensor(np.clip(U, 0, 1), dtype=torch.float32),
                                 torch.tensor(Cn, dtype=torch.float32)], -1))
        mu, lv = model.mu(e).numpy(), model.lv(e).clamp(-8, 8).numpy()
    kl = 0.5 * (mu ** 2 + np.exp(lv) - lv - 1.0)
    kl_j = kl.mean(0)
    au_j = mu.var(0)
    n_dead = int((kl_j < KL_DEAD).sum()); n_au = int((au_j > AU_TH).sum())
    print("-" * 74)
    print("① 编码器：每维 KL（nat）与 active-units")
    for j in range(model.zdim):
        print("   z%-2d  KL = %8.5f   Var(mu) = %8.5f   %s"
              % (j, kl_j[j], au_j[j],
                 "死" if kl_j[j] < KL_DEAD else ("活" if au_j[j] > AU_TH else "半死")))
    print("   总 KL = %.4f nat，死维 %d/%d，active units %d/%d"
          % (kl_j.sum(), n_dead, model.zdim, n_au, model.zdim))

    # ---------------- ② 解码器侧：z 敏感度 vs 跨条件散布 ----------------
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    pick = rng.choice(len(Cn), size=min(args.ncond, len(Cn)), replace=False)
    sig_z, means = [], []
    with torch.no_grad():
        for i in pick:
            u = model.sample(torch.tensor(Cn[i], dtype=torch.float32), args.nz).numpy()
            sig_z.append(u.std(0)); means.append(u.mean(0))
    sig_z = np.array(sig_z).mean(0)          # 每维：同条件下 z 带来的散布
    sig_c = np.array(means).std(0)           # 每维：跨条件的散布
    # ---------------- ③ 数据自身的多样性 ----------------
    sig_data = []
    for i in pick:
        dist = np.abs(Cn - Cn[i]).sum(1)
        nb = np.argsort(dist)[:args.knn]
        sig_data.append(U[nb].std(0))
    sig_data = np.array(sig_data).mean(0)

    print("-" * 74)
    print("②③ 解码器散布（u 空间，盒宽 = 1.0）")
    print("%-6s %10s %10s %10s %10s" % ("维", "σ_z(同条件)", "σ_c(跨条件)", "σ_data(数据)", "σ_z/σ_data"))
    ratios = []
    for j, nm in enumerate(names):
        r = sig_z[j] / max(sig_data[j], 1e-9); ratios.append(r)
        print("%-6s %10.4f %10.4f %10.4f %10.2f" % (nm, sig_z[j], sig_c[j], sig_data[j], r))
    med_r = float(np.median(ratios))
    print("-" * 74)
    print("同条件散布 σ_z 中位 = %.4f（盒宽 1.0 的 %.2f%%）" % (np.median(sig_z), 100 * np.median(sig_z)))
    print("σ_z / σ_data 中位 = %.2f  —— 模型保留了数据自身多样性的 %.0f%%" % (med_r, 100 * med_r))

    # ---------------- ④ 复现待办 E1：物理 x 空间的 CV ----------------
    pr = meta.get("prior", {})
    prior = BioPrior(arm=pr.get("arm", "bio"), sigma=pr.get("sigma"),
                     u_max=pr.get("u_max", 2.5), v21=bool(pr.get("v21", True)))
    cvs = []
    with torch.no_grad():
        for i in pick:
            m_kg = 10 ** float(C[i, 0])
            u = model.sample(torch.tensor(Cn[i], dtype=torch.float32), args.nz).numpy()
            x = prior.expand(np.clip(u, 0, 1), m_kg)
            cvs.append(np.abs(x.std(0) / np.where(np.abs(x.mean(0)) > 1e-12,
                                                  x.mean(0), np.nan)))
    cvs = np.array(cvs)
    cv_dim = np.nanmedian(cvs, 0) * 100.0           # 每维中位 CV(%)
    cv_med = float(np.nanmedian(cv_dim))
    print("-" * 74)
    print("④ 复现待办 E1 的口径：同条件采 %d 次，物理 x 空间的 CV（%%）" % args.nz)
    print("   " + "  ".join("%s=%.2f" % (n, v) for n, v in zip(names, cv_dim)))
    print("   9 维中位 CV = %.2f%%   （待办 E1 报的是 %.2f%%）" % (cv_med, args.e1_cv))
    if cv_med > 10 * args.e1_cv:
        print("   → **E1 的塌缩结论在 v2.3 r12 上不复现**，差了 %.0f 倍。" % (cv_med / max(args.e1_cv, 1e-9)))
        print("      E1 测的是 v2 的 7 维模型；此后换过先验、加过姿态两维、重训过多轮。")
        print("      这条待办可以关掉，但要把这个数写进报告，别让旧结论留在文档里。")
    else:
        print("   → 与 E1 一致，塌缩仍在。")

    # ---------------- 结论与口径 ----------------
    print("-" * 74)
    collapsed = ((n_dead == model.zdim) or (np.median(sig_z) < 0.01)
                 or (med_r < 0.15) or (cv_med < 1.0))
    if collapsed:
        print("判定：**后验已塌缩**。z 基本不影响输出，模型实际是一个确定性的 条件→设计 映射。")
        print()
        print("这不是 bug，也不打算在 v2.5 修（修它要动 β / 架构，属于生成模型本身的课题，")
        print("不是本文的贡献点）。它决定的是**说法**：")
        print()
        print("  ✗ 不能说：模型给出设计的**分布**，可以采样出多样的候选。")
        print("  ✓ 应该说：给定工况，模型给出**一个**仿生设计；")
        print("            它满足验收条件，但不宣称是最优解。")
        print("            —— 与赵老师原话一致：it is one solution, not necessarily the best one。")
        print()
        print("需要跟着改的地方：PPT 第 9 页「Designs from the generator」的措辞；")
        print("论文里凡是出现 p(u|c)「分布」「采样」「多样性」的句子；")
        print("以及 E20 走廊图的图注 —— 走廊的宽度来自**条件在变**，不是来自同条件下的采样。")
    else:
        print("判定：**未塌缩**。%d/%d 维隐变量活跃（总 KL %.2f nat），"
              % (n_au, model.zdim, kl_j.sum()))
        print("      同条件下换 z，输出确实在动：σ_z = 盒宽的 %.1f%%，物理 CV 中位 %.1f%%。"
              % (100 * np.median(sig_z), cv_med))
        print()
        print("      但**多样性只有数据自身的 %.0f%%**（σ_z/σ_data 中位 %.2f）——" % (100 * med_r, med_r))
        print("      模型在往条件的条件均值收，这是 cVAE 的常态，不是缺陷。")
        print()
        print("      所以口径仍然建议保守，理由不再是「塌缩」，而是「覆盖不足」：")
        print("      ✓ 给定工况，模型给出满足验收条件的仿生设计；采样能给出若干变体，")
        print("        但**不宣称覆盖了全部可行解，也不宣称其中有最优解**。")
        print("        —— 与赵老师原话一致：it is one solution, not necessarily the best one。")

    res = dict(ckpt=os.path.basename(args.ckpt), zdim=int(model.zdim), u_dim=int(nd),
               kl_per_dim=[float(v) for v in kl_j], kl_total=float(kl_j.sum()),
               var_mu=[float(v) for v in au_j], n_dead=n_dead, n_active=n_au,
               sigma_z=[float(v) for v in sig_z], sigma_c=[float(v) for v in sig_c],
               sigma_data=[float(v) for v in sig_data],
               ratio_z_data=[float(v) for v in ratios], ratio_median=med_r,
               cv_phys_pct=[float(v) for v in cv_dim], cv_phys_median=cv_med,
               e1_reported_cv=args.e1_cv,
               collapsed=bool(collapsed), names=names)
    fp = os.path.join(out_dir, "e25_posterior_collapse.json")
    json.dump(res, open(fp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print("→ %s" % fp)

    if args.fig:
        _fig(kl_j, au_j, sig_z, sig_c, sig_data, names, model.zdim, out_dir,
             collapsed, med_r, cv_med)


def _fig(kl_j, au_j, sig_z, sig_c, sig_data, names, zdim, out_dir,
         collapsed, med_r, cv_med):
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    setup_font()
    fig, ax = plt.subplots(1, 2, figsize=(12.4, 4.3),
                           gridspec_kw=dict(width_ratios=[0.62, 1.0], wspace=.26))
    a = ax[0]
    a.bar(range(zdim), kl_j, color=[CRIM if k < KL_DEAD else STEEL for k in kl_j])
    a.axhline(KL_DEAD, color=CRIM, ls="--", lw=1.4)
    a.text(zdim - .5, KL_DEAD * 1.4, "死维阈值 0.01 nat", ha="right", fontsize=9, color=CRIM)
    a.set_yscale("log"); a.set_xticks(range(zdim))
    a.set_xticklabels(["z%d" % j for j in range(zdim)])
    a.set_ylabel("每维 KL / nat")
    a.set_title("(a) 隐变量用了多少信息\n总 KL = %.3f nat，活跃 %d/%d 维"
                % (kl_j.sum(), int((au_j > AU_TH).sum()), zdim), fontsize=11)

    b = ax[1]
    x = np.arange(len(names)); w = .27
    b.bar(x - w, sig_data, w, color=GREEN, label="σ_data 数据自身")
    b.bar(x, sig_c, w, color=STEEL, label="σ_c 跨条件")
    b.bar(x + w, sig_z, w, color=CRIM, label="σ_z 同条件采 z")
    b.set_xticks(x); b.set_xticklabels(names, fontsize=10)
    b.set_ylabel("u 空间标准差（盒宽 = 1）")
    b.set_title(("(b) 同条件下换 z，输出几乎不动\nσ_z 中位仅为盒宽的 %.1f%%（已塌缩）"
                 % (100 * np.median(sig_z))) if collapsed else
                ("(b) 同条件下换 z 确实在动，但覆盖不足\nσ_z = 盒宽的 %.1f%%，仅为数据自身多样性的 %.0f%%"
                 % (100 * np.median(sig_z), 100 * med_r)), fontsize=11)
    b.legend(fontsize=9, frameon=False); b.grid(alpha=.22, axis="y")
    fig.tight_layout()
    fp = os.path.join(out_dir, "e25_posterior_collapse.png")
    fig.savefig(fp, dpi=170); print("→ %s" % fp)


if __name__ == "__main__":
    main()

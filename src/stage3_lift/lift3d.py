"""Stage 3 · 诚实版:远端测量 + 近端解剖先验。

分工(对应缺口 #2/#3 的严谨处理):
  - 远端(ankle, mtp, toe)= DLC 测量,可观测 → 真实数据
  - 近端(knee, hip)      = 体外不可观测(股骨/膝藏体内,金标准需 X 光 XROMM)
                            → 用文献骨长比 + 姿态先验几何构造,明确标为 "prior"

输出 provenance 分层:phi = measured;theta_K/theta_A = prior-based;link_ratio = literature。
纯 numpy,CPU。

文献骨长比(天鹅,Hiroshige & Riko;蓝本同源):
  femur : tibiotarsus : tarsometatarsus = 0.89 : 1.72 : 1  (相对 L1=tarsometatarsus)
姿态先验:hip-knee-ankle (theta_K) 取文献均值 ~90°(蓝本 Table 4)。

用法:
  python src/stage3_lift/lift3d.py --kp outputs/swan01/kp.json --clip_id swan01 --fps 25
产出:outputs/<clip>/motionchain.json + stage3.png
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.schema import MotionChain, dump  # noqa: E402

# 文献骨长比(相对跖骨 L1)
R_TIBIO = 1.72     # tibiotarsus / tarsometatarsus
R_FEMUR = 0.89     # femur / tarsometatarsus
THETA_K_PRIOR = 90.0   # hip-knee-ankle 文献均值(度)


def load_distal(path):
    d = json.load(open(path, encoding="utf-8"))
    fr = d["frames"]; T = len(fr)
    def col(b): return np.array([[f["kps"][b][0], f["kps"][b][1]] for f in fr])
    return col("ankle"), col("mtp"), col("toe"), T


def unit(v):
    return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-9)


def rot(v, deg):
    r = np.deg2rad(deg); c, s = np.cos(r), np.sin(r)
    return np.stack([c * v[..., 0] - s * v[..., 1], s * v[..., 0] + c * v[..., 1]], -1)


def proximal_prior(A, M):
    """从测到的踝A、跖趾M,几何构造膝K、髋H(近端先验)。"""
    L1 = np.linalg.norm(A - M, axis=-1, keepdims=True)      # 跖骨长(测量)
    u = unit(A - M)                                          # 沿可见腿向上
    K = A + R_TIBIO * L1 * u                                 # 膝:文献比例、沿腿方向
    # 股骨折向身体:u 旋转 ±90°,选更靠上(y 更小)的那支
    Hc1 = K + R_FEMUR * L1 * rot(u, 90)
    Hc2 = K + R_FEMUR * L1 * rot(u, -90)
    H = np.where((Hc1[:, 1:2] <= Hc2[:, 1:2]), Hc1, Hc2)
    return K, H, L1[:, 0]


def ang(a, b, c):
    v1, v2 = a - b, c - b
    cs = (v1 * v2).sum(-1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cs, -1, 1)))


def plot(phi, thK, thA, A, M, T_pts, K, H, fps, path):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    T = len(phi); fr = np.arange(T); snaps = [T // 20, T // 2, T - 5]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    ax[0].plot(fr, phi, color="C2", lw=2, label="phi (mtp-toe)  [MEASURED]")
    ax[0].plot(fr, thK, color="C0", ls="--", label="thetaK  [PRIOR ~90]")
    ax[0].plot(fr, thA, color="C1", ls="--", label="thetaA  [prior-based]")
    ax[0].set_xlabel(f"frame  (fps={fps:.0f}, 1 frame={1/fps:.3f}s)"); ax[0].set_ylabel("angle (deg)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("Angles: solid=measured, dashed=prior")
    td = int(np.argmin(np.diff(phi))) + 1
    ax[0].axvline(td, color="gray", ls=":", lw=1); ax[0].annotate(f"touchdown ~f{td}", (td, phi[td]),
        xytext=(td + 6, 60), fontsize=9, color="gray", arrowprops=dict(arrowstyle="->", color="gray"))
    for fi, cc in zip(snaps, ["C0", "C1", "C2"]):
        # 测量段(实线) + 先验段(虚线)
        ax[1].plot([A[fi, 0], M[fi, 0], T_pts[fi, 0]], [-A[fi, 1], -M[fi, 1], -T_pts[fi, 1]], "-o", color=cc, label=f"f{fi}")
        ax[1].plot([H[fi, 0], K[fi, 0], A[fi, 0]], [-H[fi, 1], -K[fi, 1], -A[fi, 1]], ":", color=cc)
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_title("Skeleton: solid=measured distal, dotted=prior proximal")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", required=True); ap.add_argument("--clip_id", default="swan01")
    ap.add_argument("--fps", type=float, default=25.0); ap.add_argument("--species", default="swan")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    A, M, T_pts, T = load_distal(args.kp)
    K, H, L1 = proximal_prior(A, M)
    phi = np.degrees(np.arctan2(np.abs((T_pts - M)[:, 1]), np.abs((T_pts - M)[:, 0])))  # 测量
    thK = ang(H, K, A)     # 先验(~90)
    thA = ang(K, A, M)     # 先验几何 → 半先验
    L1m = float(np.median(L1))

    print(f"[stage3-honest] {T} frames")
    print(f"  MEASURED: phi {phi.min():.0f}-{phi.max():.0f} deg | tarsometatarsus L1(median)={L1m:.0f}px")
    print(f"  PRIOR   : thetaK {thK.min():.0f}-{thK.max():.0f}(~90) | link_ratio L2/L1={R_TIBIO} L3/L1={R_FEMUR}")

    mc = MotionChain(
        clip_id=args.clip_id, species=args.species,
        link_ratio=(R_TIBIO, R_FEMUR),          # 文献先验
        init_angles={"theta_K": float(thK[-1]), "theta_A": float(thA[-1]), "phi": float(phi[-1])},
        angle_series=[{"t": t, "phi": float(phi[t]), "theta_K": float(thK[t]), "theta_A": float(thA[t])}
                      for t in range(T)],
        n_frames=T,
    )
    # provenance 标注(诚实分层)
    mc_d = mc.__dict__ if hasattr(mc, "__dict__") else None
    outdir = os.path.join(args.out, args.clip_id); os.makedirs(outdir, exist_ok=True)
    dump(mc, os.path.join(outdir, "motionchain.json"))
    # 附加 provenance 说明文件
    json.dump({"phi": "MEASURED (ankle/mtp/toe visible)",
               "theta_K": "PRIOR (~90deg literature; femur/knee not observable externally)",
               "theta_A": "PRIOR-BASED (depends on prior knee)",
               "link_ratio": "LITERATURE (swan femur:tibiotarsus:tarsometatarsus = 0.89:1.72:1)",
               "note": "distal measured, proximal reconstructed by anatomical prior; see 隐藏关节处理分析与文献"},
              open(os.path.join(outdir, "provenance.json"), "w"), ensure_ascii=False, indent=2)
    plot(phi, thK, thA, A, M, T_pts, K, H, args.fps, os.path.join(outdir, "stage3.png"))
    print(f"[stage3-honest] wrote motionchain.json + provenance.json + stage3.png → {outdir}")


if __name__ == "__main__":
    main()

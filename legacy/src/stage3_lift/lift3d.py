"""Stage 3 · 诚实版重建 + 可选骨长恒定 3D 矫正。

分工(对应缺口 #2/#3 的严谨处理):
  - 远端(ankle, mtp, toe)= DLC 测量,可观测 → 真实数据(φ 足角、ψ 跖骨角、踝轨迹)
  - 近端(knee, hip)      = 体外不可观测(股骨/膝藏体内,金标准需 X 光 XROMM)
                            → 不逐帧重建;关节角取文献常数,骨长比取文献

设计依据:任何对不可观测近端关节的逐帧定位都是无根据推断(见《隐藏关节处理分析与文献》)。
故本版仅报告可观测远端的测量量,近端以文献常数进入设计契约,不伪装为测量。

两种模式(--mode):
  honest : 直接用 2D 测量角(默认,最保守)。
  3d     : 骨长恒定 3D 提升(治缺口 #7)。跖骨/足段的 2D 投影长度忽长忽短,唯一
           物理解释是骨段在朝相机方向倾斜(前缩短);用「骨长跨帧恒定」约束反解
           每帧深度 z,把 2D 骨架撑成真三维,再算矫正后的 φ/ψ。仅作用于可观测远端链,
           近端仍是文献常数——不对不可观测关节硬凑深度。

文献常数(天鹅,蓝本 Table 4 / Hiroshige & Riko):
  theta_K (hip-knee-ankle)   = 90°
  theta_A (knee-ankle-mtp)   = 120°
  link_ratio  L2/L1=1.72, L3/L1=0.89

纯 numpy,CPU。用法:
  python src/stage3_lift/lift3d.py --kp outputs/swan01/kp.json --clip_id swan01 --fps 25 --mode 3d
产出:outputs/<clip>/motionchain.json + provenance.json + stage3.png
      (--mode 3d 额外产出 joints3d.json + stage3_3d.png)
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.schema import MotionChain, dump  # noqa: E402

# 文献常数
R_TIBIO, R_FEMUR = 1.72, 0.89      # L2/L1, L3/L1
THETA_K, THETA_A = 90.0, 120.0     # 近端关节角(度),文献


def load_distal(path):
    d = json.load(open(path, encoding="utf-8"))
    fr = d["frames"]; T = len(fr)
    def col(b): return np.array([[f["kps"][b][0], f["kps"][b][1]] for f in fr])
    def conf(b): return np.array([f["kps"][b][2] for f in fr])
    return col("ankle"), col("mtp"), col("toe"), conf("toe"), conf("mtp"), T


def orient(vec):
    """2D 向量相对水平的夹角(度),0=水平,90=竖直。"""
    return np.degrees(np.arctan2(np.abs(vec[:, 1]), np.abs(vec[:, 0])))


def orient3d(vec3):
    """3D 向量相对水平的夹角(度):把恢复出的深度 z 一并计入,消除前缩短。"""
    return np.degrees(np.arctan2(np.abs(vec3[:, 1]), np.sqrt(vec3[:, 0] ** 2 + vec3[:, 2] ** 2)))


def lift_depth(A, M, Tp, iters=6000, lr=1.0, w_smooth=0.15, w_prior=0.02, seed=0):
    """骨长恒定 3D 提升:反解每帧每关节深度 z,使 3D 骨段长度跨帧恒定。

    未知量 = 每帧 mtp / toe 的深度 z(锚定 ankle z=0 为参考)。
    目标骨长取「最长 2D 投影」——该帧骨段落在像平面内,前缩短最小,是真长的下界。
    损失 = 骨长偏差² + 时序平滑(z 不跳变) + 极小深度先验(不无端拉深)。
    numpy 手写解析梯度 + Adam。深度符号歧义由平滑项 + 随机初始化破对称解决。

    返回:J (T,3joints,3coord) x,y,z;res 每根骨的长度变异系数(残差)。
    """
    rng = np.random.default_rng(seed); T = len(A)
    P2 = np.stack([A, M, Tp], 1)                 # (T,3,2)  ankle,mtp,toe
    bones = [(0, 1), (1, 2)]                       # ankle-mtp, mtp-toe
    L2d = [np.linalg.norm(P2[:, j] - P2[:, i], axis=1) for i, j in bones]
    Ltgt = [l.max() for l in L2d]                 # 真长下界
    z = np.zeros((T, 3)); z[:, 1] = rng.normal(0, 3, T); z[:, 2] = rng.normal(0, 3, T)
    mz = np.zeros_like(z); vz = np.zeros_like(z); b1, b2, eps = 0.9, 0.999, 1e-8

    def lap(col):  # 一维时序拉普拉斯(平滑项梯度)
        return np.concatenate([[col[0] - col[1]], 2 * col[1:-1] - col[:-2] - col[2:], [col[-1] - col[-2]]])

    for it in range(1, iters + 1):
        g = np.zeros_like(z)
        for k, (i, j) in enumerate(bones):
            dz = z[:, j] - z[:, i]
            d = np.sqrt(L2d[k] ** 2 + dz ** 2)
            gd = ((d - Ltgt[k]) / d) * dz
            g[:, j] += gd; g[:, i] -= gd
        g[:, 1] += w_smooth * lap(z[:, 1])
        g[:, 2] += w_smooth * lap(z[:, 2])
        g += w_prior * z; g[:, 0] = 0             # 锚定 ankle 深度
        mz = b1 * mz + (1 - b1) * g; vz = b2 * vz + (1 - b2) * g * g
        z -= lr * (mz / (1 - b1 ** it)) / (np.sqrt(vz / (1 - b2 ** it)) + eps)

    J = np.concatenate([P2, z[..., None]], axis=2)   # (T,3,3)
    res = []
    for i, j in bones:
        L = np.linalg.norm(J[:, j] - J[:, i], axis=1)
        res.append(float(np.std(L) / np.mean(L)))
    return J, res


def plot(phi, psi, thK, thA, A, M, T_pts, fps, path, phi2=None, psi2=None):
    import matplotlib.pyplot as plt
    T = len(phi); fr = np.arange(T); snaps = [T // 20, T // 2, T - 5]
    fig, ax = plt.subplots(1, 2, figsize=(13, 4))
    if phi2 is not None:  # 3d 模式:画 2D 原始 vs 3D 矫正
        ax[0].plot(fr, phi2, color="C2", lw=1, ls=":", alpha=.6, label="phi 2D (raw)")
        ax[0].plot(fr, psi2, color="C4", lw=1, ls=":", alpha=.6, label="psi 2D (raw)")
    ax[0].plot(fr, phi, color="C2", lw=2, label="phi (foot vs horiz)  [MEASURED]")
    ax[0].plot(fr, psi, color="C4", lw=2, label="psi (tarsometatarsus vs horiz)  [MEASURED]")
    ax[0].plot(fr, thK, color="C0", ls="--", label="thetaK  [PRIOR 90, literature]")
    ax[0].plot(fr, thA, color="C1", ls="--", label="thetaA  [PRIOR 120, literature]")
    ax[0].set_xlabel(f"frame  (fps={fps:.0f}, 1 frame={1/fps:.3f}s)"); ax[0].set_ylabel("angle (deg)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3); ax[0].set_title("Angles: solid=measured, dashed=prior(literature)")
    td = int(np.argmin(np.diff(phi))) + 1
    ax[0].axvline(td, color="gray", ls=":", lw=1)
    ax[0].annotate(f"touchdown ~f{td}", (td, phi[td]), xytext=(td + 6, 60),
                   fontsize=9, color="gray", arrowprops=dict(arrowstyle="->", color="gray"))
    # 骨架:仅测量远端(ankle-mtp-toe)
    for fi, cc in zip(snaps, ["C0", "C1", "C2"]):
        ax[1].plot([A[fi, 0], M[fi, 0], T_pts[fi, 0]], [-A[fi, 1], -M[fi, 1], -T_pts[fi, 1]],
                   "-o", color=cc, label=f"f{fi}")
    ax[1].set_aspect("equal"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3)
    ax[1].set_xlabel("x (px)"); ax[1].set_ylabel("-y (px)")
    ax[1].set_title("Measured distal skeleton (ankle-mtp-toe)")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def show_3d(J):
    """交互式:弹窗显示可拖动旋转的三维远端骨架(ankle-mtp-toe),按住鼠标拖动旋转。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    T = len(J)
    fig = plt.figure(figsize=(8, 7)); ax = fig.add_subplot(111, projection="3d")
    for fi in range(0, T, max(1, T // 30)):
        c = plt.cm.viridis(fi / T)
        ax.plot(J[fi, :, 0], J[fi, :, 2], -J[fi, :, 1], "-o", color=c, ms=3, lw=1.2, alpha=.8)
    ax.plot(J[:, 2, 0], J[:, 2, 2], -J[:, 2, 1], color="gray", lw=.6, alpha=.5)  # toe 轨迹
    ax.set_xlabel("x"); ax.set_ylabel("z (depth)"); ax.set_zlabel("-y (up)")
    ax.set_title("distal 3D skeleton — 按住鼠标左键拖动旋转 (colored by time)")
    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass
    plt.tight_layout(); plt.show()


def plot_3d(J, path):
    """3D 矫正结果:侧视 / 俯视(暴露深度) / 三维斜视。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    T = len(J); snaps = list(range(0, T, max(1, T // 12)))
    fig = plt.figure(figsize=(14, 4.5))
    axa = fig.add_subplot(1, 3, 1); axb = fig.add_subplot(1, 3, 2); axc = fig.add_subplot(1, 3, 3, projection="3d")
    for fi in snaps:
        c = plt.cm.viridis(fi / T)
        axa.plot(J[fi, :, 0], -J[fi, :, 1], "-o", color=c, ms=3, lw=1)     # side (相机视角)
        axb.plot(J[fi, :, 0], J[fi, :, 2], "-o", color=c, ms=3, lw=1)      # top-down (x vs depth)
        axc.plot(J[fi, :, 0], J[fi, :, 2], -J[fi, :, 1], "-o", color=c, ms=2, lw=1)
    axa.set_aspect("equal"); axa.set_title("side view (camera)"); axa.set_xlabel("x"); axa.set_ylabel("-y"); axa.grid(alpha=.3)
    axb.set_aspect("equal"); axb.set_title("top-down: recovered depth z"); axb.set_xlabel("x"); axb.set_ylabel("z (depth)"); axb.grid(alpha=.3)
    axc.set_title("3D distal skeleton"); axc.set_xlabel("x"); axc.set_ylabel("z"); axc.set_zlabel("-y")
    plt.tight_layout(); plt.savefig(path, dpi=110); plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kp", required=True); ap.add_argument("--clip_id", default="swan01")
    ap.add_argument("--fps", type=float, default=25.0); ap.add_argument("--species", default="swan")
    ap.add_argument("--mode", choices=["honest", "3d"], default="honest",
                    help="honest=2D 测量角(默认);3d=骨长恒定深度矫正(治前缩短,缺口#7)")
    ap.add_argument("--show", action="store_true", help="3d 模式下弹窗显示可拖动旋转的三维骨架")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    # 有 --show 时用交互后端弹窗;否则强制 Agg 只存图(无显示器也能跑)
    import matplotlib
    if not (args.show and args.mode == "3d"):
        matplotlib.use("Agg")

    A, M, T_pts, ctoe, cmtp, T = load_distal(args.kp)
    phi2 = orient(T_pts - M)     # 2D 测量:足-水平
    psi2 = orient(A - M)         # 2D 测量:跖骨-水平
    thK = np.full(T, THETA_K)    # 文献常数
    thA = np.full(T, THETA_A)
    L1m = float(np.median(np.linalg.norm(A - M, axis=-1)))

    outdir = os.path.join(args.out, args.clip_id); os.makedirs(outdir, exist_ok=True)
    prov = {"theta_K": "PRIOR CONSTANT (90 deg, literature; femur/knee not observable externally)",
            "theta_A": "PRIOR CONSTANT (120 deg, literature)",
            "link_ratio": "LITERATURE (swan femur:tibiotarsus:tarsometatarsus = 0.89:1.72:1)",
            "note": "distal measured; proximal joint angles are literature constants, not per-frame estimates. See 隐藏关节处理分析与文献."}

    if args.mode == "3d":
        J, res = lift_depth(A, M, T_pts)
        phi = orient3d(J[:, 2] - J[:, 1])       # 矫正后:足段真三维角
        psi = orient3d(J[:, 0] - J[:, 1])       # 矫正后:跖骨段真三维角
        cv2d = float(np.std(np.linalg.norm(A - M, axis=1)) / np.mean(np.linalg.norm(A - M, axis=1)))
        print(f"[stage3-3d] {T} frames · 骨长恒定深度矫正")
        print(f"  bone-length residual: 2D CV {cv2d*100:.0f}% → 3D ankle-mtp {res[0]*100:.1f}% mtp-toe {res[1]*100:.1f}%")
        print(f"  phi 2D→3D diff: mean {np.abs(phi-phi2).mean():.1f} max {np.abs(phi-phi2).max():.1f} deg")
        print(f"  psi 2D→3D diff: mean {np.abs(psi-psi2).mean():.1f} max {np.abs(psi-psi2).max():.1f} deg")
        print(f"  recovered depth z: mtp [{J[:,1,2].min():.0f},{J[:,1,2].max():.0f}] toe [{J[:,2,2].min():.0f},{J[:,2,2].max():.0f}] px")
        # 导出三维关节序列
        joints = [{"t": t, "ankle": J[t, 0].tolist(), "mtp": J[t, 1].tolist(), "toe": J[t, 2].tolist()} for t in range(T)]
        json.dump({"clip_id": args.clip_id, "method": "constant-bone-length-lift (distal only)",
                   "residual_cv": {"ankle_mtp": res[0], "mtp_toe": res[1]}, "frames": joints},
                  open(os.path.join(outdir, "joints3d.json"), "w"), ensure_ascii=False, indent=2)
        plot_3d(J, os.path.join(outdir, "stage3_3d.png"))
        if args.show:
            show_3d(J)
        prov["phi"] = "MEASURED + 3D-corrected (constant-bone-length depth lift, foreshortening removed)"
        prov["psi"] = "MEASURED + 3D-corrected (constant-bone-length depth lift, foreshortening removed)"
        prov["depth_z"] = "RECOVERED (distal chain only; ankle/mtp/toe). Proximal NOT lifted."
        prov["residual_cv"] = f"2D {cv2d*100:.0f}% -> 3D {res[0]*100:.1f}%/{res[1]*100:.1f}%"
        plot(phi, psi, thK, thA, A, M, T_pts, args.fps, os.path.join(outdir, "stage3.png"), phi2=phi2, psi2=psi2)
    else:
        phi, psi = phi2, psi2
        print(f"[stage3-honest] {T} frames")
        print(f"  MEASURED: phi {phi.min():.0f}-{phi.max():.0f} | psi {psi.min():.0f}-{psi.max():.0f} deg | L1(median)={L1m:.0f}px")
        print(f"            toe conf mean={ctoe.mean():.2f}(<0.6:{100*(ctoe<0.6).mean():.0f}%) | mtp conf mean={cmtp.mean():.2f}")
        print(f"  PRIOR   : thetaK={THETA_K} thetaA={THETA_A} (literature) | link_ratio L2/L1={R_TIBIO} L3/L1={R_FEMUR}")
        prov["phi"] = "MEASURED (mtp-toe, foot vs horizontal)"
        prov["psi"] = "MEASURED (ankle-mtp, tarsometatarsus vs horizontal)"
        plot(phi, psi, thK, thA, A, M, T_pts, args.fps, os.path.join(outdir, "stage3.png"))

    mc = MotionChain(
        clip_id=args.clip_id, species=args.species,
        link_ratio=(R_TIBIO, R_FEMUR),
        init_angles={"theta_K": THETA_K, "theta_A": THETA_A, "phi": float(phi[-1])},
        angle_series=[{"t": t, "phi": float(phi[t]), "psi": float(psi[t]),
                       "theta_K": THETA_K, "theta_A": THETA_A} for t in range(T)],
        n_frames=T,
    )
    dump(mc, os.path.join(outdir, "motionchain.json"))
    json.dump(prov, open(os.path.join(outdir, "provenance.json"), "w"), ensure_ascii=False, indent=2)
    extra = " + joints3d.json + stage3_3d.png" if args.mode == "3d" else ""
    print(f"[stage3-{args.mode}] wrote motionchain.json + provenance.json + stage3.png{extra} → {outdir}")


if __name__ == "__main__":
    main()

"""水平视角(侧视)腿部连杆运动视频。

把 Stage 2 的 2D 关键点(kp.json)渲染成一段侧视连杆动画,直观展示着陆过程中
整条腿的运动。五点连杆按「测量 vs 先验」分层显示:

  远端 ankle-mtp-toe   = DLC 测量           → 绿色实线(粗)
  近端 knee-hip        = 文献先验重建        → 红色虚线(细,半透明)

近端重建(体外不可观测,不逐帧测量,见《方法学初稿》§6.2):
  - 骨长比 L2/L1=1.72(胫跗骨)、L3/L1=0.89(股骨),近端骨长用 L1 中位数锁为恒定;
  - 膝 K 沿小腿方向延伸;股骨在膝处折向「体侧」;
  - 体侧方向由对侧踝 ankle_contra 相对本侧踝锚定(解剖依据,回答"身体在哪一侧"),
    再加时间连续约束,消除小腿近竖直帧两候选点积相近导致的方向抖动。

用法:
  python tools/render_video.py --kp outputs/swan01/kp.json --out outputs/swan01/leg_linkage.mp4 --fps 25
无 ffmpeg 时自动退回输出 .gif(仅依赖 numpy + matplotlib + imageio)。
"""
from __future__ import annotations
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import imageio.v2 as imageio

R2, R3 = 1.72, 0.89   # tibiotarsus/tarsometatarsus, femur/tarsometatarsus (swan, 蓝本 Table 4)


def load(path):
    fr = json.load(open(path, encoding="utf-8"))["frames"]
    col = lambda b: np.array([[f["kps"][b][0], f["kps"][b][1]] for f in fr])
    return col("ankle"), col("mtp"), col("toe"), col("ankle_contra")


def build_chain(A, M, Tp, AC):
    """挂近端先验:骨长恒定(L1 中位数),膝沿小腿延伸,股骨折向体侧(由 ankle_contra 锚定)。"""
    L1 = np.median(np.linalg.norm(A - M, axis=1))
    u = (A - M); u = u / (np.linalg.norm(u, axis=1, keepdims=True) + 1e-8)   # mtp->ankle,近端方向
    K = A + (R2 * L1) * u
    rot_p = np.stack([-u[:, 1],  u[:, 0]], 1)   # û 转 +90°
    rot_m = np.stack([ u[:, 1], -u[:, 0]], 1)   # û 转 -90°
    bd = AC - A                                  # 身体侧方向(对侧踝 - 本侧踝),解剖锚点
    pick_p = (rot_p * bd).sum(1) > (rot_m * bd).sum(1)
    v = np.where(pick_p[:, None], rot_p, rot_m)
    for t in range(1, len(v)):                   # 时间连续兜底:近共线边界帧不许翻侧
        if np.dot(v[t], v[t - 1]) < 0:
            v[t] = -v[t]
    H = K + (R3 * L1) * v
    return H, K   # hip, knee


def render(kp, out, fps=25.0):
    A, M, Tp, AC = load(kp); T = len(A)
    H, K = build_chain(A, M, Tp, AC)
    allx = np.concatenate([H[:, 0], K[:, 0], A[:, 0], M[:, 0], Tp[:, 0]])
    ally = np.concatenate([H[:, 1], K[:, 1], A[:, 1], M[:, 1], Tp[:, 1]])
    x0, x1 = allx.min() - 20, allx.max() + 20
    y0, y1 = ally.min() - 20, ally.max() + 20
    phi = np.degrees(np.arctan2(np.abs(Tp[:, 1] - M[:, 1]), np.abs(Tp[:, 0] - M[:, 0])))
    td = int(np.argmin(np.diff(phi))) + 1        # 触地帧:足角骤降
    ground = np.median(np.concatenate([Tp[td:, 1], M[td:, 1]])) if td < T - 2 else Tp[:, 1].max()

    frames = []
    for t in range(T):
        fig, ax = plt.subplots(figsize=(6.4, 5.2), dpi=100)
        if t >= td:                              # 触地后画水面线
            ax.axhline(-ground, color="#4a90d9", lw=1.2, alpha=.5)
            ax.fill_between([x0, x1], -y1, -ground, color="#4a90d9", alpha=.06)
        ax.plot(Tp[:t + 1, 0], -Tp[:t + 1, 1], color="gray", lw=.8, alpha=.35)   # 脚趾轨迹
        ax.plot([H[t, 0], K[t, 0], A[t, 0]], [-H[t, 1], -K[t, 1], -A[t, 1]],
                "--o", color="#c0392b", lw=1.6, ms=5, alpha=.55, label="proximal (prior)")
        ax.plot([A[t, 0], M[t, 0], Tp[t, 0]], [-A[t, 1], -M[t, 1], -Tp[t, 1]],
                "-o", color="#1a7a3a", lw=3, ms=6, label="distal (measured)")
        for p, n in [(H[t], "hip"), (K[t], "knee"), (A[t], "ankle"), (M[t], "mtp"), (Tp[t], "toe")]:
            ax.annotate(n, (p[0], -p[1]), textcoords="offset points", xytext=(5, 4), fontsize=7, color="#333")
        ax.set_xlim(x0, x1); ax.set_ylim(-y1, -y0); ax.set_aspect("equal")
        ax.set_xlabel("x (px)"); ax.set_ylabel("-y (px, up)")
        tag = "  <- TOUCHDOWN" if abs(t - td) <= 1 else ""
        ax.set_title(f"lateral-view leg linkage   frame {t:3d}/{T-1}   t={t/fps:.2f}s{tag}", fontsize=10)
        ax.legend(loc="upper right", fontsize=8); ax.grid(alpha=.25)
        fig.tight_layout(); fig.canvas.draw()
        buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(fig.canvas.get_width_height()[::-1] + (4,))[:, :, :3]
        frames.append(buf); plt.close(fig)

    eff = min(fps, 20)
    try:
        imageio.mimsave(out, frames, fps=eff, codec="libx264", quality=8, macro_block_size=None)
        final = out
    except Exception as e:                        # 无 ffmpeg → 退回 gif
        final = os.path.splitext(out)[0] + ".gif"
        imageio.mimsave(final, frames, fps=eff, loop=0)
        print(f"[warn] mp4 编码失败({type(e).__name__}),已退回 gif")
    print(f"wrote {final}  {T} frames @ {eff:.0f}fps  touchdown~f{td}")
    return final


def main():
    ap = argparse.ArgumentParser(description="侧视腿部连杆运动视频(测量远端 + 先验近端)")
    ap.add_argument("--kp", required=True, help="Stage 2 关键点 kp.json")
    ap.add_argument("--out", default=None, help="输出视频路径(默认 kp 同目录 leg_linkage.mp4)")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()
    out = args.out or os.path.join(os.path.dirname(args.kp), "leg_linkage.mp4")
    render(args.kp, out, args.fps)


if __name__ == "__main__":
    main()

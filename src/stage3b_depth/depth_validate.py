"""Stage 3b · 深度模型验证工具:VGGT-Omega 深度 → 真实骨长 → 骨长恒定性打分。

思路(方法学上是"镜像验证"):
  - Stage 3 的 3D 矫正:拿「骨长恒定」当**约束**,反解深度;
  - 本工具:拿 VGGT-Omega **独立**估出的深度,算 3D 骨长,再用「骨长是否逐帧恒定」
    当**免费真值**给深度打分。两条路互相印证 → 双重验证。

为什么选 VGGT-Omega:多帧联合推理、共享场景几何 → 深度跨帧同尺度(单帧单目深度
的逐帧仿射歧义在架构层面被消掉),相机跟拍运动被位姿估计吸收。

流程:
  1. 视频帧 → VGGT-Omega → 每帧 depth map + confidence + 相机内参;
  2. 关键点像素处取深度:限制在 SAM2 掩码内、取邻域中位数(防细腿渗色到背景水面);
  3. 用内参反投影成相机系 3D 点 → 逐帧 3D 骨长(ankle-mtp、mtp-toe);
  4. 指标:每根骨 3D 长度的 CV(变异系数),对比三条基线:
       纯 2D(~15-21%) / 骨长恒定反解(~5-7%) / VGGT-Omega 直算(本实验回答)。
     若 <10%:深度工具成立;显著更差:说明该场景仍需约束优化(负结果也有价值)。

两种模式:
  extract : 跑模型出深度(需 GPU + vggt-omega 安装,A100 上执行),存 depth.npz
  analyze : 读 depth.npz + kp.json + masks,算骨长表与 CV(CPU 即可)

A100 用法:
  # 安装(一次):
  #   git clone https://github.com/facebookresearch/vggt-omega.git && cd vggt-omega
  #   pip install -r requirements.txt && pip install -e .
  python src/stage3b_depth/depth_validate.py extract \
      --frames data/swan01_frames --out outputs/swan01/depth.npz [--chunk 48 --overlap 8]
  python src/stage3b_depth/depth_validate.py analyze \
      --depth outputs/swan01/depth.npz --kp outputs/swan01/kp.json \
      --masks outputs/swan01/masks --out outputs/swan01
产出:bone3d_table.csv + depth_validation.json + stage3b_depth.png
"""
from __future__ import annotations
import argparse, glob, json, os, sys
import numpy as np

BONES = [("ankle", "mtp"), ("mtp", "toe")]          # 只验证可观测远端链
PATCH = 4                                            # 关键点邻域半径(px,深度分辨率下)


# ---------------------------------------------------------------- extract
def run_extract(args):
    """VGGT-Omega 推理:帧目录 → depth.npz(depth/conf/K/尺寸映射)。GPU."""
    import torch
    from PIL import Image
    from vggt_omega.models.vggt_omega import VGGTOmega          # noqa
    from vggt_omega.utils.load_fn import load_and_preprocess_images  # noqa

    frame_paths = sorted(glob.glob(os.path.join(args.frames, "*.png"))) or \
                  sorted(glob.glob(os.path.join(args.frames, "*.jpg")))
    assert frame_paths, f"no frames in {args.frames}"
    W0, H0 = Image.open(frame_paths[0]).size
    print(f"[extract] {len(frame_paths)} frames  src {W0}x{H0}")

    device = "cuda"
    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    model = VGGTOmega.from_pretrained(args.model).to(device).eval()

    # 分块推理(整段 VRAM 不够时);块间靠重叠帧的深度中位数比值对齐尺度
    n = len(frame_paths); chunk, ov = args.chunk, args.overlap
    depths, confs, Ks = [], [], []
    prev_tail = None; scale = 1.0
    starts = list(range(0, n, chunk - ov)) if chunk > 0 else [0]
    for si, s in enumerate(starts):
        e = min(n, s + chunk) if chunk > 0 else n
        batch = load_and_preprocess_images(frame_paths[s:e]).to(device)
        with torch.inference_mode(), torch.autocast("cuda", dtype=dtype):
            pred = model(batch)
        d = pred["depth"].float().squeeze(-1).cpu().numpy()          # (T,h,w)
        c = pred.get("depth_conf")
        c = c.float().cpu().numpy() if c is not None else np.ones_like(d)
        K = pred["intrinsic"].float().cpu().numpy() if "intrinsic" in pred else None
        if prev_tail is not None and ov > 0:                          # 块间尺度对齐
            cur_head = d[:ov]
            r = np.nanmedian(prev_tail[prev_tail > 0]) / max(np.nanmedian(cur_head[cur_head > 0]), 1e-9)
            scale *= r
        d = d * scale
        keep = slice(ov if si > 0 else 0, None)
        depths.append(d[keep]); confs.append(c[keep])
        if K is not None: Ks.append(K[keep] if K.ndim == 3 else np.repeat(K[None], d[keep].shape[0], 0))
        prev_tail = d[-ov:] if ov > 0 else None
        print(f"  chunk {s}-{e} scale={scale:.3f}")
        if e >= n: break
    depth = np.concatenate(depths); conf = np.concatenate(confs)
    K = np.concatenate(Ks) if Ks else None
    np.savez_compressed(args.out, depth=depth, conf=conf, K=K,
                        src_size=np.array([W0, H0]), n_frames=len(frame_paths))
    print(f"[extract] wrote {args.out}  depth {depth.shape}")


# ---------------------------------------------------------------- analyze
def _sample_depth(dmap, cmap, mask, u, v, patch=PATCH):
    """掩码约束的邻域中位数深度:防细腿像素渗色到背景。"""
    h, w = dmap.shape
    u0, u1 = max(0, u - patch), min(w, u + patch + 1)
    v0, v1 = max(0, v - patch), min(h, v + patch + 1)
    win_d = dmap[v0:v1, u0:u1]; win_c = cmap[v0:v1, u0:u1]
    win_m = mask[v0:v1, u0:u1] if mask is not None else np.ones_like(win_d, bool)
    sel = win_m & (win_d > 0)
    if sel.sum() < 3:                                   # 掩码内点太少 → 放宽到整窗
        sel = win_d > 0
    if sel.sum() == 0:
        return np.nan, 0.0
    return float(np.median(win_d[sel])), float(np.median(win_c[sel]))


def run_analyze(args):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

    z = np.load(args.depth, allow_pickle=True)
    depth, conf, K = z["depth"], z["conf"], z["K"]
    W0, H0 = z["src_size"]; T, h, w = depth.shape
    sx, sy = w / W0, h / H0                              # 原图 → 深度图坐标缩放
    kp = json.load(open(args.kp, encoding="utf-8"))["frames"]
    T = min(T, len(kp))
    joints = sorted({j for b in BONES for j in b})

    # 掩码(可选):SAM2 png,resize 到深度分辨率
    masks = None
    if args.masks and os.path.isdir(args.masks):
        from PIL import Image
        mfiles = sorted(glob.glob(os.path.join(args.masks, "*.png")))
        if len(mfiles) >= T:
            masks = [np.array(Image.open(m).resize((w, h), Image.NEAREST)) > 0 for m in mfiles[:T]]

    # 3D 关节
    P3, C = {j: np.full((T, 3), np.nan) for j in joints}, {j: np.zeros(T) for j in joints}
    for t in range(T):
        Kt = K[t] if K is not None and K.ndim == 3 else K
        fx, fy, cx, cy = (Kt[0, 0], Kt[1, 1], Kt[0, 2], Kt[1, 2]) if Kt is not None \
                         else (max(h, w), max(h, w), w / 2, h / 2)   # 无内参时近似
        m = masks[t] if masks is not None else None
        for j in joints:
            x, y = kp[t]["kps"][j][0] * sx, kp[t]["kps"][j][1] * sy
            d, c = _sample_depth(depth[t], conf[t], m, int(round(x)), int(round(y)))
            P3[j][t] = [(x - cx) / fx * d, (y - cy) / fy * d, d]
            C[j][t] = c

    # 骨长:3D vs 2D(2D 用原图像素,单独归一)
    res = {}
    for a, b in BONES:
        L3 = np.linalg.norm(P3[a] - P3[b], axis=1)
        x2 = np.array([[kp[t]["kps"][j][0], kp[t]["kps"][j][1]] for t in range(T) for j in (a, b)])
        L2 = np.linalg.norm(x2[0::2] - x2[1::2], axis=1)
        ok = np.isfinite(L3) & (L3 > 0)
        cv3 = float(np.nanstd(L3[ok]) / np.nanmean(L3[ok])) if ok.sum() > 5 else np.nan
        cv2 = float(np.std(L2) / np.mean(L2))
        res[f"{a}-{b}"] = dict(cv3d=cv3, cv2d=cv2, n_valid=int(ok.sum()),
                               L3=L3.tolist(), L2=L2.tolist())
        print(f"[analyze] {a}-{b}: CV 2D={cv2*100:.1f}%  VGGT-Omega 3D={cv3*100:.1f}%  (valid {ok.sum()}/{T})")

    os.makedirs(args.out, exist_ok=True)
    # 表
    import csv
    with open(os.path.join(args.out, "bone3d_table.csv"), "w", newline="") as f:
        wcsv = csv.writer(f); wcsv.writerow(["frame"] + [f"L3_{a}-{b}" for a, b in BONES] +
                                            [f"L2_{a}-{b}" for a, b in BONES])
        for t in range(T):
            wcsv.writerow([t] + [round(res[f'{a}-{b}']["L3"][t], 2) for a, b in BONES] +
                          [round(res[f'{a}-{b}']["L2"][t], 2) for a, b in BONES])
    json.dump({"summary": {k: {kk: v[kk] for kk in ("cv3d", "cv2d", "n_valid")} for k, v in res.items()},
               "baselines": {"2d_raw": "见 cv2d", "const_bone_lift": "~5-7% (lift3d --mode 3d)",
                             "pass_criterion": "cv3d < 10% → 深度工具成立"},
               "model": "VGGT-Omega", "note": "深度为多帧联合推理,跨帧同尺度;掩码约束邻域中位数采样。"},
              open(os.path.join(args.out, "depth_validation.json"), "w"), ensure_ascii=False, indent=2)
    # 图
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
    for i, (a, b) in enumerate(BONES):
        L3 = np.array(res[f"{a}-{b}"]["L3"]); L2 = np.array(res[f"{a}-{b}"]["L2"])
        ax[0].plot(L2 / np.nanmean(L2), ls="--", alpha=.6, color=f"C{i}", label=f"{a}-{b} 2D")
        ax[0].plot(L3 / np.nanmean(L3), color=f"C{i}", label=f"{a}-{b} 3D(VGGT-Omega)")
        ax[1].bar([i * 3, i * 3 + 1],
                  [res[f"{a}-{b}"]["cv2d"] * 100, res[f"{a}-{b}"]["cv3d"] * 100],
                  color=["#999", f"C{i}"])
    ax[0].axhline(1, color="k", lw=.5); ax[0].set_xlabel("frame")
    ax[0].set_ylabel("bone length / mean"); ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    ax[0].set_title("Bone length constancy: flat=good")
    ax[1].set_xticks([0.5, 3.5]); ax[1].set_xticklabels([f"{a}-{b}" for a, b in BONES])
    ax[1].axhline(10, color="crimson", ls=":", label="10% pass line")
    ax[1].set_ylabel("CV (%)"); ax[1].legend(fontsize=8); ax[1].grid(alpha=.3, axis="y")
    ax[1].set_title("gray=2D raw, color=VGGT-Omega 3D")
    plt.tight_layout(); plt.savefig(os.path.join(args.out, "stage3b_depth.png"), dpi=110); plt.close()
    print(f"[analyze] wrote bone3d_table.csv + depth_validation.json + stage3b_depth.png → {args.out}")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract"); e.add_argument("--frames", required=True)
    e.add_argument("--out", required=True); e.add_argument("--model", default="facebook/VGGT-Omega-1B-512")
    e.add_argument("--chunk", type=int, default=48); e.add_argument("--overlap", type=int, default=8)
    a = sub.add_parser("analyze"); a.add_argument("--depth", required=True)
    a.add_argument("--kp", required=True); a.add_argument("--masks", default=None)
    a.add_argument("--out", default="outputs/swan01")
    args = ap.parse_args()
    (run_extract if args.cmd == "extract" else run_analyze)(args)


if __name__ == "__main__":
    main()

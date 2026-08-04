"""把 Stage 1 的 mask 叠回帧上,导出一段视频,人眼确认腿被稳定跟住。

用法:
  python tools/overlay.py --clip_id swan01 --out outputs --fps 25
产出:outputs/<clip_id>/overlay.mp4  (红色 = 分割到的腿部)
"""
from __future__ import annotations
import argparse, glob, os
import cv2
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--fps", type=float, default=25.0)
    args = ap.parse_args()

    base = os.path.join(args.out, args.clip_id)
    frames = sorted(glob.glob(os.path.join(base, "frames", "*.jpg")))
    if not frames:
        raise SystemExit("没找到 frames,先跑 stage1")
    h, w = cv2.imread(frames[0]).shape[:2]
    vw = cv2.VideoWriter(os.path.join(base, "overlay.mp4"),
                         cv2.VideoWriter_fourcc(*"mp4v"), args.fps, (w, h))
    for fp in frames:
        img = cv2.imread(fp)
        mp = os.path.join(base, "masks", os.path.basename(fp).replace(".jpg", ".png"))
        if os.path.exists(mp):
            m = cv2.imread(mp, 0) > 0
            red = np.zeros_like(img); red[..., 2] = 255
            img[m] = (0.5 * img[m] + 0.5 * red[m]).astype(np.uint8)
        vw.write(img)
    vw.release()
    print(f"[overlay] wrote {base}/overlay.mp4  ({len(frames)} frames)")


if __name__ == "__main__":
    main()

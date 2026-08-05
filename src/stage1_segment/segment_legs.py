"""Stage 1 · 腿部分割 & 跟踪 (SAM2).

治缺口 #8:记忆机制穿瞬时自遮挡跟踪;为 Stage3 提供 silhouette 约束。

输入:一段裁剪好的着陆视频 (mp4) + 首帧对腿部的 prompt(点或框)。
     可用 --start_sec/--end_sec 只处理有用的着陆窗口(其余帧不抽)。
输出:每帧腿部 mask(png)+ bbox,跨帧同一 track_id → SegOut(JSON)。

需要 GPU。sandbox 无法跑,这份代码在 A100 上执行。用法见 README。

依赖:pip install --no-build-isolation "git+https://github.com/facebookresearch/sam2.git"
      pip install opencv-python numpy
checkpoint:sam2.1_hiera_large.pt(见 sam2 官方 README 下载)
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.schema import SegFrame, SegOut, dump  # noqa: E402


def parse_prompt(s: str):
    """'point:cx,cy' 或 'box:x1,y1,x2,y2' → (kind, np.array)。首帧提示腿部位置。"""
    kind, vals = s.split(":")
    nums = [float(x) for x in vals.split(",")]
    if kind == "point":
        return "point", np.array([[nums[0], nums[1]]], dtype=np.float32)
    if kind == "box":
        return "box", np.array(nums, dtype=np.float32)
    raise ValueError(f"bad prompt: {s}")


def extract_frames(video_path: str, out_dir: str,
                   start_sec: float = 0.0, end_sec: float = 1e9) -> tuple[list[str], float]:
    """抽取 [start_sec, end_sec] 窗口内的帧,重新从 0 编号(SAM2 需要连续序列)。"""
    import cv2
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise SystemExit(f"打不开视频: {video_path}(不存在或损坏。注意 data/*.mp4 不进 git,需 scp 到本机)")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if (fps and fps > 0) else 25.0   # 兜底,避免 -1/0 导致空窗口
    f0 = int(start_sec * fps)
    f1 = int(end_sec * fps)
    src_i, dst_i, paths = 0, 0, []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if f0 <= src_i <= f1:
            p = os.path.join(out_dir, f"{dst_i:05d}.jpg")
            cv2.imwrite(p, frame)
            paths.append(p)
            dst_i += 1
        src_i += 1
        if src_i > f1:
            break
    cap.release()
    return paths, fps


def bbox_from_mask(mask: np.ndarray):
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return (0, 0, 0, 0)
    x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
    return (x1, y1, x2 - x1, y2 - y1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="着陆视频 mp4")
    ap.add_argument("--clip_id", required=True)
    ap.add_argument("--ckpt", required=True, help="sam2.1_hiera_large.pt 路径")
    ap.add_argument("--cfg", default="configs/sam2.1/sam2.1_hiera_l.yaml")
    ap.add_argument("--prompt", required=True, help="'point:cx,cy' 或 'box:x1,y1,x2,y2'(窗口首帧腿部)")
    ap.add_argument("--start_sec", type=float, default=0.0, help="只处理该时刻之后")
    ap.add_argument("--end_sec", type=float, default=1e9, help="只处理该时刻之前")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    import cv2
    import torch
    from sam2.build_sam import build_sam2_video_predictor

    frames_dir = os.path.join(args.out, args.clip_id, "frames")
    masks_dir = os.path.join(args.out, args.clip_id, "masks")
    os.makedirs(masks_dir, exist_ok=True)
    frame_paths, fps = extract_frames(args.video, frames_dir, args.start_sec, args.end_sec)
    print(f"[stage1] window [{args.start_sec},{args.end_sec}]s @ {fps:.1f}fps → {len(frame_paths)} frames")
    if not frame_paths:
        raise SystemExit("窗口内没有帧,检查 start_sec/end_sec")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    predictor = build_sam2_video_predictor(args.cfg, args.ckpt, device=device)
    kind, prompt = parse_prompt(args.prompt)

    seg = SegOut(clip_id=args.clip_id)
    with torch.inference_mode(), torch.autocast(device, dtype=torch.bfloat16):
        state = predictor.init_state(video_path=frames_dir)
        if kind == "point":
            predictor.add_new_points_or_box(
                state, frame_idx=0, obj_id=1, points=prompt,
                labels=np.array([1], dtype=np.int32))
        else:
            predictor.add_new_points_or_box(state, frame_idx=0, obj_id=1, box=prompt)

        for fidx, obj_ids, mask_logits in predictor.propagate_in_video(state):
            m = (mask_logits[0] > 0).cpu().numpy().squeeze().astype(np.uint8) * 255
            mp = os.path.join(masks_dir, f"{fidx:05d}.png")
            cv2.imwrite(mp, m)
            seg.frames.append(SegFrame(
                frame_id=fidx, bbox=bbox_from_mask(m), mask_path=mp, track_id=1))

    out_json = os.path.join(args.out, args.clip_id, "seg.json")
    dump(seg, out_json)
    print(f"[stage1] wrote {len(seg.frames)} masks + {out_json}")
    print("[验证] 用 tools/overlay.py 叠 mask 回放,人眼确认腿被稳定跟住")


if __name__ == "__main__":
    main()

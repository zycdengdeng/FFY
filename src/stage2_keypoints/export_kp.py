"""Stage 2 · DLC 推理输出 (.h5) → KpOut json(对齐 schema)。

用法:
  python src/stage2_keypoints/export_kp.py --h5 <analyze输出.h5> --clip_id swan01
产出:outputs/<clip_id>/kp.json  (每帧 6 关键点 [x,y,conf],喂 Stage 3)
"""
from __future__ import annotations
import argparse, os, sys
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.schema import KpFrame, KpOut, dump, KEYPOINTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--h5", required=True, help="DLC analyze_videos 输出的 .h5")
    ap.add_argument("--clip_id", default="swan01")
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    df = pd.read_hdf(args.h5)
    scorer = df.columns.levels[0][0]
    out = KpOut(clip_id=args.clip_id)
    for i in range(len(df)):
        kps = {}
        for bp in KEYPOINTS:
            if (scorer, bp, "x") in df.columns:
                x = float(df[(scorer, bp, "x")].iloc[i])
                y = float(df[(scorer, bp, "y")].iloc[i])
                c = float(df[(scorer, bp, "likelihood")].iloc[i])
                kps[bp] = (x, y, c)
        out.frames.append(KpFrame(frame_id=i, kps=kps))

    os.makedirs(os.path.join(args.out, args.clip_id), exist_ok=True)
    p = os.path.join(args.out, args.clip_id, "kp.json")
    dump(out, p)
    print(f"[export] wrote {p}  ({len(out.frames)} frames)")


if __name__ == "__main__":
    main()

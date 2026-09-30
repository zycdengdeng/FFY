"""Stage 2 · 建 DeepLabCut 项目 + 抽帧待标注。

在**有显示器的机器(Windows)**上跑,因为后面标注要 GUI。
关键点 = schema.KEYPOINTS 的 6 个鸟腿点。

用法:
  python src/stage2_keypoints/dlc_setup.py data/swan01_win.mp4
产出:stage2_dlc/ffy-leg-<experimenter>-<date>/  (含 config.yaml + 抽好的待标帧)
"""
from __future__ import annotations
import os, sys

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))
from src.common.schema import KEYPOINTS  # noqa: E402

BODYPARTS = list(KEYPOINTS)   # hip, knee, ankle(=intertarsal), mtp, toe, ankle_contra
SKELETON = [["hip", "knee"], ["knee", "ankle"], ["ankle", "mtp"], ["mtp", "toe"]]


def main():
    import deeplabcut
    from deeplabcut.utils import auxiliaryfunctions

    video = sys.argv[1] if len(sys.argv) > 1 else "data/swan01_win.mp4"
    if not os.path.exists(video):
        raise SystemExit(f"找不到视频 {video} —— 先 trim(见 README_stage2)")

    cfg = deeplabcut.create_new_project(
        "ffy-leg", "zihanw", [os.path.abspath(video)],
        working_directory="stage2_dlc", copy_videos=True, multianimal=False)
    print("config.yaml:", cfg)

    auxiliaryfunctions.edit_config(cfg, {
        "bodyparts": BODYPARTS,
        "numframes2pick": 20,     # 抽 20 帧标注,单 clip 概念验证足够
        "skeleton": SKELETON,
        "dotsize": 6,
    })
    print("bodyparts:", BODYPARTS)

    deeplabcut.extract_frames(cfg, mode="automatic", algo="kmeans",
                              userfeedback=False, crop=False)
    print("\n抽帧完成。下一步在 Windows 标注:")
    print(f'  python -c "import deeplabcut; deeplabcut.label_frames(r\'{cfg}\')"')


if __name__ == "__main__":
    main()

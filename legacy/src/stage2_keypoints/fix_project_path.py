"""在 A100 上修正从 Windows 搬过来的 DLC 项目绝对路径。

DLC 的 config.yaml 里 project_path / video_sets 存的是 Windows 绝对路径,
搬到 Linux 后失效。此脚本把它们改成当前实际位置。

用法(在 A100,dlc 环境):
  python src/stage2_keypoints/fix_project_path.py \
    /mnt/zihanw/FFY/stage2_dlc/ffy-leg-zihanw-2026-08-04/config.yaml
"""
import os
import sys

from deeplabcut.utils import auxiliaryfunctions


def main():
    cfg = os.path.abspath(sys.argv[1])
    proj = os.path.dirname(cfg)
    c = auxiliaryfunctions.read_config(cfg)
    c["project_path"] = proj
    vs = c.get("video_sets") or {}
    newvs = {}
    for k, v in vs.items():
        name = os.path.basename(k.replace("\\", "/"))
        newvs[os.path.join(proj, "videos", name)] = v
    if newvs:
        c["video_sets"] = newvs
    auxiliaryfunctions.write_config(cfg, c)
    print("project_path ->", proj)
    print("video_sets   ->", list(newvs.keys()))


if __name__ == "__main__":
    main()

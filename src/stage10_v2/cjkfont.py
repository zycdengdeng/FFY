# -*- coding: utf-8 -*-
"""中文字体解析:让画图脚本在各种机器上都别变方框。

踩过的坑,按出现频率排:
  1. 系统装了字体,但 matplotlib 的字体缓存是装之前建的 → ttflist 里没有
  2. 字体装在 conda 环境里(share/fonts),不在系统字体目录
  3. 真的没装
前两种都不该让用户去 rm -rf ~/.cache/matplotlib,脚本自己解决。
"""
from __future__ import annotations

import glob
import os
import subprocess

CANDIDATES = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
              "Source Han Sans SC", "Source Han Sans CN", "Source Han Sans",
              "WenQuanYi Zen Hei", "WenQuanYi Micro Hei", "Droid Sans Fallback",
              "Microsoft YaHei", "SimHei", "PingFang SC", "Heiti SC",
              "AR PL UMing CN", "Noto Serif CJK SC")
# conda 环境、用户目录、系统目录都扫一遍
SEARCH = ("~/.fonts", "~/.local/share/fonts",
          os.path.join(os.environ.get("CONDA_PREFIX", "/nonexistent"), "share", "fonts"),
          os.path.join(os.environ.get("CONDA_PREFIX", "/nonexistent"), "fonts"),
          "/usr/share/fonts", "/usr/local/share/fonts")
PAT = ("*CJK*", "*NotoSansCJK*", "*SourceHanSans*", "*wqy*", "*WenQuanYi*",
       "*msyh*", "*simhei*", "*SimHei*", "*Droid*Fallback*")


def _names(fm):
    return {f.name for f in fm.fontManager.ttflist}


def _addfont(fm, path):
    try:
        fm.fontManager.addfont(path)
        return True
    except Exception:
        return False


def resolve(verbose=True):
    """返回一个可用的中文字体名,找不到返回 None。会主动修缓存和注册字体文件。"""
    from matplotlib import font_manager as fm

    have = _names(fm)
    for c in CANDIDATES:
        if c in have:
            return c

    # ① 缓存可能过期:强制重建一次再看
    try:
        fm._load_fontmanager(try_read_cache=False)
        have = _names(fm)
        for c in CANDIDATES:
            if c in have:
                if verbose:
                    print(f"[font] 字体缓存过期,已重建 → {c}")
                return c
    except Exception:
        pass

    # ② 问 fontconfig 要中文字体文件,直接注册进 matplotlib
    try:
        out = subprocess.run(["fc-list", ":lang=zh", "file"],
                             capture_output=True, text=True, timeout=15).stdout
        for line in out.splitlines():
            fp = line.split(":")[0].strip()
            if fp.lower().endswith((".ttf", ".otf", ".ttc")) and _addfont(fm, fp):
                for c in CANDIDATES:
                    if c in _names(fm):
                        if verbose:
                            print(f"[font] 由 fc-list 注册 {os.path.basename(fp)} → {c}")
                        return c
    except Exception:
        pass

    # ③ 扫常见目录(含 conda 环境),按文件名猜
    for root in SEARCH:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for pat in PAT:
            for fp in glob.glob(os.path.join(root, "**", pat), recursive=True):
                if fp.lower().endswith((".ttf", ".otf", ".ttc")) and _addfont(fm, fp):
                    for c in CANDIDATES:
                        if c in _names(fm):
                            if verbose:
                                print(f"[font] 由 {root} 注册 {os.path.basename(fp)} → {c}")
                            return c
    return None


def setup(verbose=True):
    """装到 rcParams 上;返回字体名或 None。图里的中文靠它。"""
    import matplotlib.pyplot as plt
    name = resolve(verbose=verbose)
    if name:
        plt.rcParams["font.sans-serif"] = [name] + list(
            plt.rcParams["font.sans-serif"])
    elif verbose:
        print("[font][warn] 没找到中文字体,图上中文会变方框。\n"
              "  有 sudo: sudo apt-get install -y fonts-noto-cjk\n"
              "  无 sudo: conda install -y -c conda-forge fonts-anaconda\n"
              "  装完若仍不行: rm -rf ~/.cache/matplotlib")
    plt.rcParams["axes.unicode_minus"] = False
    return name


if __name__ == "__main__":
    n = setup()
    print("中文字体:", n or "无")
    raise SystemExit(0 if n else 1)

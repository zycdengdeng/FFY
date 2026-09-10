# -*- coding: utf-8 -*-
"""A 组共用的读盘与判据。

刻意**不 import physics_v2 / factory_v2**：那条链会拉进 exudyn，而 A 组是纯后处理，
应该在任何一台只有 numpy 的机器上都能跑（包括没装 exudyn 的笔记本）。
所有常量都从 `factory_meta.json` 或下面的兜底常量取，并在 `selfcheck()` 里
与真模块比对，防止两边悄悄漂开。
"""
import json, os
import numpy as np

G = 9.81

# 与 factory_v2.KEYS_V2 一致（顺序即 Y 的列序，下游一切索引都按此）
KEYS_V2 = ["peak_a", "stroke", "leg_stroke", "sink", "eta", "cfe", "peak_jerk",
           "E_abs", "F_peak", "rebound", "n_bounce", "t_settle",
           "leg_mass_kg", "mass_frac", "struct_over", "mass_over"]
iP, iS, iL = (KEYS_V2.index("peak_a"), KEYS_V2.index("stroke"),
              KEYS_V2.index("leg_stroke"))
iSO, iMO = KEYS_V2.index("struct_over"), KEYS_V2.index("mass_over")
iREB, iNB = KEYS_V2.index("rebound"), KEYS_V2.index("n_bounce")
iLM = KEYS_V2.index("leg_mass_kg")

# 与 dataset_v2 一致
GCAP_RANGE = (4.0, 25.0)          # g
SMAX_RANGE = (0.008, 0.040)       # m
NREQ_DEFAULT, KTOP_DEFAULT = 4, 8
SEED_DATASET = 90_000             # dataset_v2.main 的每块种子基数


def selfcheck(verbose=True):
    """能 import 到真模块时，比对常量；不能就跳过。返回 True 表示一致或无从比对。"""
    try:
        import sys
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from factory_v2 import KEYS_V2 as K2
        from dataset_v2 import GCAP_RANGE as GR, SMAX_RANGE as SR
    except Exception as e:
        if verbose:
            print("[agroup_io] 未能 import 真模块（%s），使用兜底常量" % type(e).__name__)
        return True
    ok = (list(K2) == KEYS_V2 and tuple(GR) == GCAP_RANGE and tuple(SR) == SMAX_RANGE)
    if verbose:
        print("[agroup_io] 常量比对：%s" % ("一致" if ok else "**不一致，请同步**"))
    return ok


def load_blocks(fp):
    rows = []
    with open(fp, encoding="utf-8") as f:
        for line in f:
            try:
                rows.append(json.loads(line))
            except Exception:
                pass
    return rows


def to_Y(blk):
    return np.array([[np.nan if v is None else v for v in y] for y in blk["Y"]], float)


def feasible_mask(Y, gcap, smax):
    """与 physics_v2.feasible_v2 同口径：gcap 作用在峰值加速度，smax 作用在**腿行程**。

    注意这里**没有回弹判据** —— 这正是 E24 要补的那条。
    """
    ok = np.isfinite(Y[:, iP])
    return (ok & (Y[:, iP] <= gcap) & (Y[:, iL] <= smax)
            & (np.nan_to_num(Y[:, iSO], nan=1.0) < 0.5)
            & (np.nan_to_num(Y[:, iMO], nan=1.0) < 0.5))


def pareto2(a, b):
    """二目标非支配前沿（都取小），返回下标。"""
    idx = np.argsort(a, kind="stable")
    out, best = [], np.inf
    for i in idx:
        if b[i] < best - 1e-15:
            out.append(int(i)); best = b[i]
    return out


def iter_requirements(blk, nreq=NREQ_DEFAULT, seed_base=SEED_DATASET):
    """复现 dataset_v2 给每个块抽的那 nreq 组 (gcap, smax)。种子一致 = 同一批要求。"""
    crng = np.random.default_rng(seed_base + blk["cid"])
    for _ in range(nreq):
        yield (float(crng.uniform(*GCAP_RANGE) * G),
               float(crng.uniform(*SMAX_RANGE)))


def meta_of(factory_fp):
    fp = os.path.join(os.path.dirname(factory_fp), "factory_meta.json")
    return json.load(open(fp, encoding="utf-8")) if os.path.exists(fp) else {}


def default_factory(root):
    return os.path.join(root, "outputs", "v23_data_bio", "factory.jsonl")


def setup_font():
    """中文字体 + **拉丁/数字回退**。

    A100 上踩过的坑：cjkfont.setup() 挑中的可能是 Droid Sans Fallback 一类
    「有汉字、没数字和括号」的字体，于是图上中文正常、数字全是豆腐块。
    所以无论 cjkfont 成不成功，这里都强制把 DejaVu Sans 追加到回退链末尾，
    并显式设定 font.family —— 少了 family 这一行，回退链根本不生效。
    """
    import matplotlib.pyplot as plt, matplotlib.font_manager as fm
    try:
        import cjkfont
        cjkfont.setup(verbose=False)
    except Exception:
        pass
    have = {f.name for f in fm.fontManager.ttflist}
    for pth in ("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.otf",
                "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc"):
        if os.path.exists(pth):
            try:
                fm.fontManager.addfont(pth)
            except Exception:
                pass
    have = {f.name for f in fm.fontManager.ttflist}
    # 优先挑「汉字数字都全」的；Droid Sans Fallback 放最后（它缺拉丁与数字）
    order = ("Noto Sans CJK SC", "Noto Sans CJK JP", "Noto Sans CJK TC",
             "Source Han Sans SC", "WenQuanYi Zen Hei", "WenQuanYi Micro Hei",
             "Microsoft YaHei", "SimHei", "Droid Sans Fallback")
    cjk = [f for f in order if f in have]
    plt.rcParams["font.sans-serif"] = cjk + ["DejaVu Sans"]   # DejaVu 兜底拉丁与数字
    plt.rcParams["font.family"] = "sans-serif"                 # 少这行回退链不生效
    plt.rcParams["axes.unicode_minus"] = False
    if not cjk:
        print("[agroup_io] 没找到中文字体，图里的中文会是方块。"
              "装一个：sudo apt-get install -y fonts-noto-cjk")


CRIM, STEEL, GREEN, ORANGE = "#7F2D32", "#5c7f9e", "#2E7D5B", "#d98032"

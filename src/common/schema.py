"""Pipeline I/O contracts (对应技术方案 v0.1 §0/§2).

每个 stage 的输入/输出都用这里的 dataclass 表达,序列化为 JSON。
坐标系:Intertarsal(踝)为原点,对侧踝定水平轴,矢状面内分析(沿用蓝本 AESCTE)。
尺度:全局单一尺度,最终只用骨长比,不求绝对毫米。
范围:只重建空中下降段 + 触地瞬间,不涉及水下/水动力学。
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional
import json

# 关键点集:蓝本 4 点(hip/knee/ankle=intertarsal/mtp)+ 加密 toe + 对侧踝(矫正参考)
KEYPOINTS = ["hip", "knee", "ankle", "mtp", "toe", "ankle_contra"]

# 三段连杆(近端→远端),对应鸟腿骨骼
SEGMENTS = ["femur(L3)", "tibiotarsus(L2)", "tarsometatarsus(L1)"]


# ---------- Stage 0: 数据筛选 ----------
@dataclass
class Clip:
    clip_id: str
    path: str
    species: str          # swan / goose / duck / pelican
    fps: float
    frame_range: tuple[int, int]
    view: str = "lateral"     # 侧视为主
    quality: str = "ok"        # ok / blur / occluded
    notes: str = ""


# ---------- Stage 1: 腿部分割 & 跟踪 (SAM2) ----------
@dataclass
class SegFrame:
    frame_id: int
    bbox: tuple[int, int, int, int]     # x, y, w, h
    mask_path: str                       # 存成 png,避免 JSON 里塞 RLE
    track_id: int = 0
    score: float = 1.0


@dataclass
class SegOut:
    clip_id: str
    frames: list[SegFrame] = field(default_factory=list)


# ---------- Stage 2: 2D 关键点 (DeepLabCut) ----------
@dataclass
class KpFrame:
    frame_id: int
    # 每个关键点: [x, y, confidence]; 缺失/低conf 由下游用先验补
    kps: dict[str, tuple[float, float, float]] = field(default_factory=dict)


@dataclass
class KpOut:
    clip_id: str
    keypoints: list[str] = field(default_factory=lambda: list(KEYPOINTS))
    frames: list[KpFrame] = field(default_factory=list)


# ---------- Stage 3: 逐帧 3D 关节化重建 ----------
@dataclass
class Pose3DFrame:
    frame_id: int
    joints3d: dict[str, tuple[float, float, float]] = field(default_factory=dict)
    links: tuple[float, float, float] = (0.0, 0.0, 0.0)   # L1,L2,L3 (逐帧,未约束前会漂)


@dataclass
class Pose3DOut:
    clip_id: str
    method: str = "avian-mesh"    # or "lassie"
    frames: list[Pose3DFrame] = field(default_factory=list)


# ---------- Stage 4/5: 骨长恒定运动链 → DESIGN INPUT ----------
@dataclass
class MotionChain:
    """最终交付给蓝本第二步生成设计的契约。字段名对齐 AESCTE eq.12/13 + Table 4。"""
    clip_id: str
    species: str
    link_ratio: tuple[float, float]        # (r2=L2/L1, r3=L3/L1)  ← 尺度无关
    init_angles: dict[str, float]          # {theta_K, theta_A, phi} at t=0 触地帧
    angle_series: list[dict]               # [{t, theta_K, theta_A, phi}, ...] 全程运动学(蓝本没有)
    segments: list[str] = field(default_factory=lambda: list(SEGMENTS))
    coord_frame: str = "intertarsal-origin, contra-ankle horizontal"
    n_frames: int = 0


def dump(obj, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(obj), f, ensure_ascii=False, indent=2)


def load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

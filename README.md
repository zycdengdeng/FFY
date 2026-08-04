# 水鸟腿部 视频→三段连杆运动链 pipeline

从公开水鸟着陆视频重建腿部**简单三维关节化运动链**(比例为关键,绝对尺度不重要),
输出对齐蓝本(AESCTE)第二步生成设计的输入契约。设计详见《技术方案 v0.1》。

范围:只重建**空中下降段 + 触地瞬间**腿部姿态,不涉及水下运动/水动力学。

## 目录

```
src/common/schema.py        # 各 stage 的 I/O 契约(dataclass)
src/stage1_segment/         # SAM2 腿部分割+跟踪   (治缺口 #8)
src/stage2_keypoints/       # DeepLabCut 2D 关键点 (治缺口 #2)  [待写]
src/stage3_lift/            # avian-mesh ∥ LASSIE 逐帧3D        [待写]
src/stage4_temporal/        # 骨长恒定时序约束 ★方法贡献        [待写]
configs/  data/  outputs/
```

## 运行环境(A100 集群,非本地沙箱)

```bash
# /mnt/zihanw 下 clone 本 repo,建 conda 环境
conda create -n feifei python=3.11 -y && conda activate feifei
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install "git+https://github.com/facebookresearch/sam2.git" opencv-python numpy
# 下载 SAM2 checkpoint(见 sam2 官方 README): sam2.1_hiera_large.pt
```

## Stage 1 · 跑腿部分割

```bash
python src/stage1_segment/segment_legs.py \
  --video data/swan_landing_01.mp4 \
  --clip_id swan01 \
  --ckpt /path/to/sam2.1_hiera_large.pt \
  --prompt "point:640,720" \
  --out outputs
# 首帧腿部大致位置给个点 point:cx,cy(或框 box:x1,y1,x2,y2)
```

产出:`outputs/swan01/masks/*.png` + `outputs/swan01/seg.json`。
验证:叠 mask 回放,人眼确认腿被稳定跟住(尤其自遮挡帧)。

## 工作流(sandbox 写代码 → 集群跑)

Claude 在云沙箱写/改代码 → commit 进本地文件夹 → 你 `git push` →
A100 `git pull` 到 /mnt/zihanw → 跑 → 回传日志 → Claude 改。
(沙箱无 GPU、连不到集群,故 GPU 任务在 A100 执行。)

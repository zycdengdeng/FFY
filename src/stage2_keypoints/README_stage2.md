# Stage 2 · DeepLabCut 腿部 2D 关键点

目标:对着陆窗口每帧提取 6 个鸟腿关键点 → `outputs/<clip>/kp.json`,喂 Stage 3。
关键点:`hip, knee, ankle(=intertarsal), mtp, toe, ankle_contra`(蓝本 4 点 + toe + 对侧踝矫正)。

分工:**Windows 建项目 + 标注(要 GUI)→ A100 训练(GPU)**。

---

## 0. 先 trim 一个窗口 clip(两阶段共用,保证帧对齐)

```bash
# 16–20s,重编码保证逐帧精确;SAM2 和 DLC 都用这个
ffmpeg -ss 16 -to 20 -i data/swan01.mp4 -c:v libx264 -an data/swan01_win.mp4
```
> 之后 SAM2 也建议直接跑 `swan01_win.mp4`(`--start_sec 0 --end_sec 100`),这样 Stage1/2 帧号一一对应。

## 1. 装 DLC(单独 conda 环境,别污染 ffy)

```bash
# Windows(标注) 和 A100(训练) 各建一个
conda create -n dlc python=3.10 -y && conda activate dlc
pip install "deeplabcut[gui]"      # A100 无需 gui,可去掉 [gui]
```

## 2. 建项目 + 抽帧(Windows)

```bash
conda activate dlc
python src/stage2_keypoints/dlc_setup.py data/swan01_win.mp4
# 产出 stage2_dlc/ffy-leg-zihanw-<date>/ ,已抽 20 帧待标
```

## 3. 标注(Windows,GUI)

```bash
python -c "import deeplabcut; deeplabcut.label_frames(r'stage2_dlc/ffy-leg-.../config.yaml')"
```
在弹出的 GUI 里,对每帧按顺序点 6 个点:hip→knee→ankle→mtp→toe→ankle_contra。
被遮住看不见的点可跳过(DLC 允许缺标)。标完保存。

## 4. 训练(A100,GPU)

把整个 `stage2_dlc/` 项目同步到 A100(git 或 scp),然后:
```bash
conda activate dlc
python - <<'PY'
import deeplabcut
cfg = "stage2_dlc/ffy-leg-.../config.yaml"   # 改成实际路径
deeplabcut.create_training_dataset(cfg)
deeplabcut.train_network(cfg, maxiters=20000, saveiters=5000, displayiters=200)
deeplabcut.evaluate_network(cfg, plotting=True)
PY
```

## 5. 推理 + 导出

```bash
python - <<'PY'
import deeplabcut
cfg = "stage2_dlc/ffy-leg-.../config.yaml"
deeplabcut.analyze_videos(cfg, ["data/swan01_win.mp4"], save_as_csv=True)
deeplabcut.create_labeled_video(cfg, ["data/swan01_win.mp4"])   # 出带点的视频,肉眼验证
PY
# DLC 会在 data/ 下生成 *.h5;转成我们的 schema:
python src/stage2_keypoints/export_kp.py --h5 data/swan01_win*.h5 --clip_id swan01
```

产出 `outputs/swan01/kp.json`。验证:看 `create_labeled_video` 出的带点视频,关节点跟得准不准。

---

## 备注
- 单 clip、20 帧标注只能在这段视频上用;要泛化到别的 clip 需标更多帧/多视频。先把这段跑通验证流程。
- 帧对齐:Stage1(SAM2)和 Stage2(DLC)都用 `swan01_win.mp4`,帧号一致,Stage3 才能对上。

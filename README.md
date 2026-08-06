# 水鸟腿部 视频→三段连杆运动链 pipeline

从公开水鸟着陆视频重建腿部**简单三维关节化运动链**(比例为关键,绝对尺度不重要),
输出对齐蓝本(AESCTE)第二步生成设计的输入契约。设计详见《技术方案 v0.1》。

范围:只重建**空中下降段 + 触地瞬间**腿部姿态,不涉及水下运动/水动力学。

## 目录

```
src/common/schema.py        # 各 stage 的 I/O 契约(dataclass)
src/stage1_segment/         # SAM2 腿部分割+跟踪   (治缺口 #8)  ✅
src/stage2_keypoints/       # DeepLabCut 2D 关键点 (治缺口 #2)  ✅
src/stage3_lift/            # 诚实版重建 + 骨长恒定3D矫正(#7)  ✅
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

## Stage 3 · 重建运动链(纯 numpy / CPU,本地即可跑,无需 GPU)

只依赖 `numpy + matplotlib`,在本机 conda 环境直接跑。输入是 Stage 2 的 `kp.json`。

```bash
# 诚实版:直接用 2D 测量角(最保守,默认)
python src/stage3_lift/lift3d.py --kp outputs/swan01/kp.json --clip_id swan01 --fps 25

# 骨长恒定 3D 矫正:反解深度、消前缩短,产出三维关节序列
python src/stage3_lift/lift3d.py --kp outputs/swan01/kp.json --clip_id swan01 --fps 25 --mode 3d

# 想直接弹窗、按住鼠标拖动旋转看三维模型,加 --show
python src/stage3_lift/lift3d.py --kp outputs/swan01/kp.json --clip_id swan01 --fps 25 --mode 3d --show
```

产出(`outputs/swan01/`):
- `motionchain.json` — 交付蓝本第二步的契约(连杆比 + 初始角 + 全程角时序)
- `provenance.json` — 逐量标注「测量 vs 文献先验」
- `stage3.png` — 角度曲线(实线=测量,虚线=文献先验;3d 模式叠 2D vs 3D 对比)
- `--mode 3d` 额外:`joints3d.json`(三维关节序列)+ `stage3_3d.png`(侧视/俯视露深度/三维斜视)

看三维模型的两种方式:① `--show` 弹交互窗口拖动旋转;② 直接打开 `outputs/swan01/stage3_3d.png` 看三面板静图。
只矫正可观测远端链(ankle-mtp-toe),近端 knee/hip 仍文献常数——不对不可观测关节硬凑深度。

**侧视连杆运动视频**(整条腿逐帧动画,测量远端实线 + 先验近端虚线):

```bash
python tools/render_video.py --kp outputs/swan01/kp.json --out outputs/swan01/leg_linkage.mp4 --fps 25
```

近端体侧方向由对侧踝 `ankle_contra` 锚定(全程同侧,不翻转)。无 ffmpeg 时自动退回 `.gif`。

## Stage 4 · 对接蓝本第二步(MotionChain → 设计 → 冲击 → Pareto,纯 numpy/CPU)

把 Stage 3 的 `motionchain.json` 喂进「生成式设计」流程,端到端跑通,验证第 1 步输出能驱动设计。

```bash
python src/stage4_design/design_opt.py --mc outputs/swan01/motionchain.json --clip_id swan01 --n 60
```

- 设计空间对齐蓝本 eq.12/13:`L1∈[250,490]mm`、`r2=L2/L1∈[1.3,2.5]`、`r3=L3/L1∈[0.9,2.0]`,拉丁超立方采样。
- motionchain 的初始关节角定触地姿态(力臂),骨长比作参考点标注。
- 产出:`design_results.csv`、`design_opt.json`、`stage4_pareto.png`(目标空间+设计空间)、`stage4_response.png`(冲击时程)。

⚠ **冲击评估是简化 1-DOF 解析代理,不是 ANSYS FE**,只用于打通 pipeline 与设计排序(数值量级示意)。
已知:被动落震 EA 近守恒(降为诊断,改用 stroke 作第三目标);线性代理里 peak_jerk 与 peak_a 近共线。
一个发现:真实天鹅 `r3=0.89` 落在蓝本设计下界 `0.9` **之外**——生物比例与工程设计空间存在错位。

## Stage 5 · 生物动态耦合(★独立贡献,纯 numpy/CPU)

把全程角时序提炼成尺度无关的「生物着陆律」,揭示被动柔顺范式无法复现的动态签名。

```bash
python src/stage5_biocouple/biocouple.py --kp outputs/swan01/kp.json --clip_id swan01
```

- 产出:`bio_landing_law.csv`、`biocouple.json`、`stage5_biocouple.png`(左:生物律 vs 被动设计;右:延迟起峰签名)。
- 核心结论:生物律「慢-快-慢」,压缩速率峰值延迟到 τ≈0.24;被动弹簧-阻尼速率峰值恒在 τ=0 → 物理上够不到 → 需主动/分段柔顺。
- 详见《独立贡献_生物动态耦合方法草案.md》。

## 工作流(sandbox 写代码 → 集群跑)

Claude 在云沙箱写/改代码 → commit 进本地文件夹 → 你 `git push` →
A100 `git pull` 到 /mnt/zihanw → 跑 → 回传日志 → Claude 改。
(沙箱无 GPU、连不到集群,故 GPU 任务在 A100 执行。)

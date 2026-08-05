# PROJECT STATUS — 水鸟腿部「视频 → 三段连杆运动链」pipeline

> 更新日期:2026-08-05 · 仓库:https://github.com/zycdengdeng/FFY

## 0. 一句话

从公开水鸟着陆视频,重建腿部**简单三维关节化运动链**(比例为关键,绝对尺度不重要),
输出对齐蓝本(AESCTE)第二步生成式设计的输入,支撑「仿生 + AI 起落架」课题。

**范围界定**:只重建**空中下降段 + 触地瞬间**的腿部姿态,不涉及水下运动与水动力学(与不碰 CFD/流固耦合的红线一致)。

---

## 1. Pipeline 总览与进度

```
网络视频 → [S0 筛选] → [S1 分割跟踪] → [S2 2D关键点] → [S3 2D→3D] → [S4 骨长恒定约束] → [S5 契约导出]
```

| 阶段 | 内容 | 工具 | 状态 |
|---|---|---|---|
| S0 | 数据筛选 + 裁窗口 | ffmpeg | ✅ 完成 |
| S1 | 整鸟分割 + 跟踪 | SAM2 | ✅ 跑通 |
| S2 | 腿部 6 关节 2D 检测 | DeepLabCut | ✅ 跑通 |
| S3+S4 | 2D→3D 重建 + 骨长恒定时序约束(**方法贡献 #7**) | 自研优化(numpy) | 🟡 原型跑通(数值待标注改进) |
| S5 | 导出 MotionChain 契约 | 已并入 lift3d | ✅ 跑通 |

---

## 2. 已完成阶段详情

### S1 · SAM2 整鸟分割 + 跟踪 ✅
- **做什么**:首帧给一个点提示,SAM2 记忆机制向后传播,逐帧分割整只天鹅并跨帧跟踪。
- **为什么是整鸟不是腿**:S3 的 avian-mesh/LASSIE 本来就吃整只鸟的 silhouette;腿部定位交给 S2 关键点。
- **输入 → 输出**:`swan01_win.mp4` → 每帧 mask(png)+ bbox → `seg.json`。
- **治缺口**:#8(穿自遮挡/运动模糊跟踪)。
- **结果**:150 帧全程稳定跟踪,含触水滑行段。

### S2 · DeepLabCut 腿部 2D 关键点 ✅
- **做什么**:用**迁移学习**把 ImageNet 预训练的 ResNet-50 微调成"天鹅腿关节检测器"。不存在现成的鸟腿关节模型,是自己训的。
- **关键点(6)**:`hip, knee, ankle(跗间), mtp, toe, ankle_contra`。主腿 5 点链 + 对侧踝 1 点(坐标系矫正参考,同蓝本 §2.2.1)。
- **标注量**:20 帧(DLC kmeans 从 150 帧自动挑代表性帧)。种子标注由 AI 生成、人工微调。~19 训练 / 1 测试。
- **精度**:train rmse 3.48 px,test rmse_pcutoff 3.20 px(误差约 3 像素,概念验证足够)。
- **推理**:训练好后自动预测**全部 150 帧** → `kp.json`(每帧 6 关节 [x,y,conf],平均置信度 0.79–0.87)。
- **核心逻辑**:标少(20)推多(150)。

### S3+S4 · 3D 提升 + 骨长恒定优化 🟡 原型
- **做什么**:2D 关键点固定,优化每关节深度 z + 每骨长度,使**同一根骨跨帧长度恒定**(硬约束 = 贡献 #7),加时序平滑 + 最小深度先验化解单目歧义。纯 numpy/CPU。
- **输入 → 输出**:`kp.json` → 3D 关节 + 恒定骨长 + 关节角时序 → `motionchain.json`(字段对齐蓝本)。
- **机制结果**:主 3 根骨恒定残差 5.5–6%;φ(足-水平角)在 t≈19s 触水时急降到 0,正确捕捉触水事件。
- **已知局限**:绝对骨长比/关节角与蓝本生物学有偏差,根因是 hip/knee 隐藏关节的 2D 标注质量(仅 20 帧种子标注)——暴露缺口 #2/#3,是下一步优化重点,非架构问题。近侧视 → 面外运动小,z 偏小属正常。

---

## 3. 代码结构

```
FFY/
├── README.md                 # 项目说明 + A100 运行命令
├── PROJECT_STATUS.md         # 本文件
├── environment.yml           # conda 环境(sam2/torch cu121)
├── .gitignore                # data/outputs/*.mp4/*.pt/stage2_dlc 不进 git
├── sync.ps1                  # 一键 commit+push(5 次重试,治 GitHub 443 抽风)
├── configs/pipeline.yaml     # 全局配置:坐标系 / 关键点 / 骨长比先验
├── data/                     # 视频(不进 git,走 scp)
├── outputs/                  # 结果(不进 git)
│   └── swan01/               # seg.json, masks/, kp.json
├── tools/overlay.py          # mask/关键点叠加可视化验证
└── src/
    ├── common/schema.py               # 所有 stage 的 I/O 契约(dataclass)
    ├── stage1_segment/segment_legs.py # SAM2 分割+跟踪
    └── stage2_keypoints/
        ├── dlc_setup.py               # 建 DLC 项目 + 抽帧
        ├── export_kp.py               # DLC h5 → kp.json
        ├── fix_project_path.py        # Windows→Linux 跨平台路径修复
        └── README_stage2.md           # Stage 2 步骤说明
```

---

## 4. 关键设计决策

- **倒推设计**:先定死交付给蓝本第二步的 `MotionChain`(连杆比例 + 初始关节角 + 全程角时序,字段对齐 AESCTE eq.12/13 + Table 4),再逐级往上定每个 stage。
- **只输出比例 + 角度,不求绝对尺度** → 绕过单目尺度歧义(蓝本本就只用比例)。
- **坐标系**:踝(Intertarsal)为原点,对侧踝定水平轴,矢状面分析(沿用蓝本)。
- **骨长比先验**:天鹅 1:2:1、苍鹭 1:2.31:1.85(来自蓝本),防物种域外污染(#9)。
- **单腿链 + 对侧踝**:远腿常被遮、两腿对称冗余,故只标一条腿 + 对侧踝作矫正参考。

---

## 5. 环境与工作流

- **仓库**:`github.com/zycdengdeng/FFY`
- **Windows(本机)**:`C:\Users\ZihanWANG\Desktop\FFY\FFY` — 写代码、DLC 标注(GUI)、`sync.ps1` 推送。
- **A100 集群**:`/mnt/zihanw/FFY` — GPU 训练/推理。conda 环境:`ffy`(SAM2)、`dlc`(DeepLabCut)。
- **协作流**:Claude 写代码 → Windows `sync.ps1` push → A100 `git pull`。DLC 项目(含帧/模型)走 `scp`,不进 git。
- **样例数据**:`swan01_win.mp4` — istock 天鹅着陆视频裁 16–22s,150 帧 @ 25fps,768×432。

---

## 6. 待办(下一步)

- [ ] **S3 · 2D→3D 重建**:跑 avian-mesh 或 LASSIE,把 `kp.json` + mask 拟合成每帧 3D 关节;裁成三段连杆运动链。
- [ ] **S4 · 方法贡献**:加「骨长恒定 + 生物比例先验」的整段序列联合优化(治缺口 #7,最强增量)。
- [ ] **S5 · 契约导出**:算 `link_ratio` + `init_angles` + `angle_series`,导出 `MotionChain`,对接蓝本第二步。
- [ ] 扩数据:更多视频/物种/视角(当前单段单物种,仅概念验证)。
- [ ] 无 3D 真值下的验证协议(多视角机会样本 / 合成数据 / 生物力学一致性自检)。

---

## 7. 已知问题 / 备忘

- DLC 3.0 评估阶段有 `reshape failed / Expected 2 individuals` 警告 → 只是可视化把项目当 multi-animal,不影响训练/推理,可忽略。
- 路径不能带中文(napari/DLC 在非 ASCII 路径会静默失败)—— 已把工作目录从 `EIT飞飞鱼` 改为 `FFY`。
- 蓝本论文未发布公开数据集,水鸟视频均自采公开视频(与蓝本一致)。
- S1 与 S2 必须用同一窗口(16–22s / 150 帧),帧号一一对应,S3 才能融合。

---

## 8. 关联文档

- 《文献调研索引》《研究缺口清单》《技术方案 v0.1》《素材源清单》—— 见 `文献调研_视频到3D重建/`
- 蓝本论文:`AESCTE-D-26-02311`(Zhao 组在审),`Manuscript v2`
- 交接文档:`博士面试_项目交接文档.md`

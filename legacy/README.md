# legacy/ 退役代码（2026-09-30 整合）

活代码闭包分析（从 run_overnight/run_v25_main/run_e18c/run_er3_full/run_p4_guild/run_pgls/run_g10/run_g1_arms 及分析入口出发顺 import 链）之外的文件，整目录或逐文件移入此处。git 历史完整保留，确认不再需要后可整目录删除。

- `src/stage1..5, 3b, 9, common`：视频→3D→早期设计链（Duong 数据处理的原始管线；姿态范围的出处代码在此）。
- `src/stage6_surrogate`：旧生物分析与画图脚本（pgls_p1*、allometry_*、mk_*、fig_*）。现行生物分析只剩 b_sensitivity / pgls_p4_prep / pgls_p2·p3·p4 R 脚本。
- `src/stage7_generative`：v1 代生成线（e5_loop v1、data_factory 等）。现行只剩 train_cvae.py。
- `src/stage10_v2`：v2.0–v2.4 时代的实验/画图/PPT/动画脚本（e15–e27 旧实验、fig_*、mk_ppt_*、anim_*、p1/p3/p4/p7/p8 探针）。p6_buckling.py 因缺陷登记"屈曲待查"仍留在活区。
- `run_scripts/`：v2.0–v2.3 时代的 run 脚本。
- `root_probes/`：根目录一次性探针（p9w*、v3_grid_probe、w0_smoke、freeze_v1）。

留在活区的 33 个文件见仓库 src/；物理配置唯一出处 evalcfg.py。

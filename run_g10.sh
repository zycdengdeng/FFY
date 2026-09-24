#!/usr/bin/env bash
# G10 · b 口径修正敏感性(2026-09-24 判定的收尾实验)
#   问题:先验历史口径 b=0.391 的拟合总体混入 10 个不会飞物种;修正口径 b=0.407。
#   要证:下游设计分布对这 4% 的斜率差不敏感(预期 L_ref 变 ~2%,判据=臂差<种子噪声地板)。
#   用法(A100): cd /mnt/zihanw/FFY && nohup bash run_g10.sh > /dev/null 2>&1 &
#   机时:≈ 单种子全量的一份(工厂 + 1 seed 闭环);跑完看 logs/g10_*.log 末尾结论。
set -uo pipefail
cd "$(dirname "$0")"
LOG="logs/g10_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

# 复用 v25 主跑批(bird 构型、cfnylon、同 codever),只换臂、单种子、独立输出目录
ARM=bio407 SEEDS="0" ROUNDS="${ROUNDS:-40}" WORKERS="${WORKERS:-128}" \
  OUT_F=outputs/g10_bio407_data OUT_E=outputs/g10_bio407_e5 \
  ROOT_D=outputs/g10_bio407_root P9=outputs/g10_bio407_p9 \
  bash run_v25_main.sh

echo; echo "=========== G10 判定:臂差 vs 种子噪声地板 ==========="
echo "--- 噪声地板(bio 两种子) ---"
python src/stage10_v2/seed_compare.py --dirs outputs/v25_bird_e5 outputs/v25_bird_e5_s1
echo "--- 臂差(bio seed0 vs bio407 seed0) ---"
python src/stage10_v2/seed_compare.py --dirs outputs/v25_bird_e5 outputs/g10_bio407_e5
echo "判据:第二段的差 ≤ 第一段的散布 ⇒ 口径修正不动结论,论文口径切 0.407 无代价。"

#!/usr/bin/env bash
# G1 决断实验:v2.5 终版物理下的四臂消融补全(geo/elastic/none,各单种子)。
#   背景:v2.5 bio 臂 b_eff=0.33±0.04(九工况,两种子),不是 v2.2 的 0.238;
#   判据:四臂 b_eff 收敛到共同 b* ⇒ 物理选斜率;各臂继承各自先验 ⇒ R3 改写。
#   用法(A100): cd /mnt/zihanw/FFY && nohup bash run_g1_arms.sh > /dev/null 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
LOG="logs/g1_arms_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
for A in geo elastic none; do
  echo "=============== 臂 $A ==============="
  ARM="$A" SEEDS="0" ROUNDS="${ROUNDS:-40}" WORKERS="${WORKERS:-128}" \
    OUT_F="outputs/g1_${A}_data" OUT_E="outputs/g1_${A}_e5" \
    ROOT_D="outputs/g1_${A}_root" P9="outputs/g1_${A}_p9" \
    bash run_v25_main.sh
done
echo "四臂齐了:bio=outputs/v25_bird_e5  bio407=outputs/g10_bio407_e5  geo/elastic/none=outputs/g1_*_e5"

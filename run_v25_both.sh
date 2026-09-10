#!/usr/bin/env bash
# 一次跑完两种构型。仿鸟在前（优先级高），对置在后。
#   cd /mnt/zihanw/FFY && nohup bash run_v25_both.sh > /dev/null 2>&1 &
#
# 两个各约 18 小时，串行 ≈ 36 小时。前一个挂了后一个照跑（各自有自己的关卡）。
# 中途想只看第一个的结果：reports/v25_bird/ 在第一个跑完时就已经齐了。
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
MASTER="logs/v25_both_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$MASTER") 2>&1
t0=$(date +%s)

for CFG in bird skid; do
  echo
  echo "############################################################"
  echo "###   构型 $CFG   开始于 $(date '+%F %T')"
  echo "############################################################"
  CONFIG="$CFG" bash run_v25_main.sh
  echo "###   构型 $CFG 结束，累计 $(( ($(date +%s)-t0)/60 )) 分钟"
done

echo
echo "########## 两种构型都跑完了，共 $(( ($(date +%s)-t0)/60 )) 分钟 ##########"
echo
echo "结果在 reports/v25_bird/ 与 reports/v25_skid/"
echo "带回本地：git add -A reports && git commit -m 'v2.5 两构型' && git push"
echo
echo "最先该看的三个数（两个构型各一份）："
echo "  · p9a_report.json 的 fr_recommended  —— 这个构型能扛多大的水平速度"
echo "  · p9_mu.json      的 mu_crit         —— 需要多大的摩擦才站得住"
echo "  · trajectory.json 最后一轮的 feas_rate 与 median_gap"

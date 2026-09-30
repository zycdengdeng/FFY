#!/usr/bin/env bash
# ER3 修复后复评(2026-09-30):物理 = 修复后 physics_v2(D-6..D-9)。
#   seed 0:4 工况 × 7 质量 × {common, bio} = 56 次 DE,预算 6000,约 6 h。
#   与 outputs/er3_full(修复前)同网格同种子 → 天然配对。可中断续跑(已有 json 跳过)。
#   用法(A100,tmux):cd /mnt/zihanw/FFY && git pull && bash run_er3_recheck.sh
#   完毕后下载:outputs/er3_recheck/ 与本日志。
set -u
cd "$(dirname "$0")"
mkdir -p logs outputs/er3_recheck
LOG="logs/er3_recheck_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-96}"; B="${BUDGET:-6000}"
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟] $(date '+%m-%d %H:%M')"; }
echo "开始 $(date)  workers=$W  physics_v2 修复版(确认下行 stamp)"
python src/stage10_v2/evalcfg.py || exit 1
for C in turf1.2 wetsand1.2 concrete1.2 wetsand2.0; do
  for M in 4 5.77 8.32 12 17.3 24.96 36; do
    for SP in common bio; do
      echo "--- cond=$C m=$M space=$SP de budget=$B seed=0 ---"
      python src/stage10_v2/er3_search.py --cond "$C" --space "$SP" --mass "$M" \
        --method de --budget "$B" --seed 0 --workers "$W" --out outputs/er3_recheck
    done
  done
  el
done
echo; echo "=========== 汇总 $(date) ==========="
echo "完成 $(ls outputs/er3_recheck/er3_*.json 2>/dev/null | wc -l) / 56"
el
echo "结束。下载:outputs/er3_recheck/  $LOG"

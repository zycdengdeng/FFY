#!/usr/bin/env bash
# 2026-10-01 出门长跑:① ENG-prior 对照(≈40 min)→ ② ER3 复评 seed 1(≈6.3 h)→ ③ ER3 复评 seed 2(≈6.3 h)。
#   全部可中断续跑(已有 json 自动跳过);中途回来可以直接 Ctrl-C,下次 bash run_trip.sh 接着跑。
#   用法(A100,tmux):cd /mnt/zihanw/FFY && git pull && bash run_trip.sh
#   下载:outputs/e18c_dev*/ outputs/er5_engprior*/ outputs/er3_recheck/(seed 1、2 新文件) logs/trip_*.log
set -u
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/trip_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-96}"; B="${BUDGET:-6000}"
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟] $(date '+%m-%d %H:%M')"; }
echo "开始 $(date)  workers=$W"
python src/stage10_v2/evalcfg.py || { echo "!!!! evalcfg 导入失败,停"; exit 1; }

echo; echo "################ ① ENG-prior ################"
WORKERS=$W bash run_engprior.sh
echo "ENG-prior 退出码 $?"; el

for S in 1 2; do
  echo; echo "################ ②/③ ER3 复评 seed=$S ################"
  for C in turf1.2 wetsand1.2 concrete1.2 wetsand2.0; do
    for M in 4 5.77 8.32 12 17.3 24.96 36; do
      for SP in common bio; do
        [ -f "outputs/er3_recheck/er3_${C}_${SP}_m${M}_de_s${S}.json" ] && { echo "跳过 $C $SP m=$M s=$S"; continue; }
        echo "--- cond=$C m=$M space=$SP de budget=$B seed=$S ---"
        python src/stage10_v2/er3_search.py --cond "$C" --space "$SP" --mass "$M" --method de --budget "$B" --seed "$S" --workers "$W" --out outputs/er3_recheck
      done
    done
    el
  done
  echo "seed $S 完成 $(ls outputs/er3_recheck/er3_*_s${S}.json 2>/dev/null | wc -l) / 56"
done
echo; echo "=========== 全部结束 $(date) ==========="; el
echo "下载:outputs/e18c_dev/ outputs/e18c_dev_s40/ outputs/er5_engprior/ outputs/er5_engprior_s40/ outputs/er3_recheck/ $LOG"

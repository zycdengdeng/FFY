#!/usr/bin/env bash
# E-R3 正式跑(物理=evalcfg=v2.5 工厂同款;旧 outputs/er3 试跑不合并)
#   7 个对数等距质量 × 4 个有解工况 × {common, bio} × DE × 3 种子,预算 6000/次。
#   种子放最外层:seed 0 全部跑完(56 次,约 6 h)就已可出初版带;中途可 Ctrl-C,重启自动跳过已完成。
#   用法(A100,tmux):cd /mnt/zihanw/FFY && git pull && bash run_er3_full.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs/er3_full
LOG="logs/er3_full_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-96}"; B="${BUDGET:-6000}"
SEEDS="${SEEDS:-0 1 2}"
CONDS="${CONDS:-turf1.2 wetsand1.2 concrete1.2 wetsand2.0}"
MASSES="${MASSES:-4 5.77 8.32 12 17.3 24.96 36}"
python src/stage10_v2/evalcfg.py
for S in $SEEDS; do
  for C in $CONDS; do
    for M in $MASSES; do
      for SP in common bio; do
        echo "=== seed=$S cond=$C m=$M space=$SP de budget=$B ==="
        python src/stage10_v2/er3_search.py --cond "$C" --space "$SP" --mass "$M" \
          --method de --budget "$B" --seed "$S" --workers "$W" --out outputs/er3_full
      done
    done
  done
done
echo "ER3 正式跑完毕:outputs/er3_full/er3_*.json"

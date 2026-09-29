#!/usr/bin/env bash
# E-R3 试跑(小预算,定正式预算用):3 质量点 × {common, bio} × {random, de} × 1 种子。
#   共同空间 L1∈[20,350] mm 常数区间 —— 指数无法经边界进入;近最优带 ε 敏感性内建。
#   用法(A100,可与四臂并行,默认只占 32 workers):
#     cd /mnt/zihanw/FFY && bash run_er3_trial.sh
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs/er3
LOG="logs/er3_trial_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-32}"; B="${BUDGET:-3000}"
for M in 5 12 30; do
  for SP in common bio; do
    for MT in random de; do
      echo "=== m=$M space=$SP method=$MT ==="
      python src/stage10_v2/er3_search.py --space "$SP" --mass "$M" \
        --method "$MT" --budget "$B" --seed 0 --workers "$W" --out outputs/er3
    done
  done
done
echo "试跑完毕:outputs/er3/er3_*.json;看各 band_eps5 的 L1 带与 edge_frac(>0.10 报警)"

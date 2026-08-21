#!/usr/bin/env bash
# E18 可行体重走廊 + E19 并集标尺互考:一键顺序跑(A100,约 35 分钟)
#   bash run_v2_e18_e19.sh
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-128}"
mkdir -p logs
LOG="logs/v2_e18e19_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }

echo "=========== E18 可行体重走廊(4臂 × 9u × 14级 × 32探针) ==========="
python src/stage10_v2/e18_mass_limit.py --workers "$WORKERS" --out outputs/v2_e18
el

echo "=========== E19 并集标尺互考(72题 × 4盒参考 + 4臂生成) ==========="
python src/stage10_v2/e19_union_exam.py --workers "$WORKERS" --out outputs/v2_e19
el
echo "跑完后下载: outputs/v2_e18/e18_corridor.json  outputs/v2_e19/e19_results.json  outputs/v2_e19/union_refs.json"

#!/usr/bin/env bash
# v2 专家迭代:按臂顺序跑自提升循环(接生物先验、按束防泄漏、结构判据在线)
#
#   PILOT=1 bash run_v2_e5.sh                    # G2 小试:主臂 8 轮,约 25 分钟
#   bash run_v2_e5.sh                            # 主臂 bio 85 轮,约 4–5 小时
#   ARMS="bio geo elastic none" ROUNDS=20 bash run_v2_e5.sh   # 四臂消融,约 4 小时
#
# 单轮成本(A100 128 并行,约 0.7 s/仿真):
#   评测 76 题 × 40 设计 × 2 遍 ≈ 6,100 次 ≈ 35 秒
#   提议 ~350 块 × 24 候选 × 2 遍 ≈ 16,800 次 ≈ 95 秒
#   训练 CPU 约 30 秒     ⇒ 约 2.5–3 分钟/轮
# 考卷只建一次:76 题 × 300 参考设计 × 2 遍 ≈ 45,600 次 ≈ 4 分钟(之后走缓存)
set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${WORKERS:-128}"
ARMS="${ARMS:-bio}"
if [[ "${PILOT:-0}" == "1" ]]; then
  ROUNDS="${ROUNDS:-8}"; SUF="_pilot"; NREF="${NREF:-120}"; NEXAM="${NEXAM:-40}"
else
  ROUNDS="${ROUNDS:-85}"; SUF=""; NREF="${NREF:-300}"; NEXAM="${NEXAM:-76}"
fi
KGEN="${KGEN:-24}"; NGEN="${NGEN:-40}"; KSCEN="${KSCEN:-6}"; SEED="${SEED:-0}"

mkdir -p logs
LOG="logs/v2_e5_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1

t0=$(date +%s)
step() { echo -e "\n=========== [$(date +%H:%M:%S)] $* ==========="; }
el()   { echo "[累计 $(( ($(date +%s) - t0) / 60 )) 分钟]"; }

echo "v2 专家迭代 | 臂=$ARMS  轮=$ROUNDS  考卷=${NEXAM}题×${NREF}参考  workers=$WORKERS"
echo "日志: $LOG"

for ARM in $ARMS; do
  FAC="outputs/v2_data_${ARM}${SUF}/factory.jsonl"
  OUT="outputs/v2_e5_${ARM}${SUF}"
  if [[ ! -f "$FAC" ]]; then
    echo "!! 缺数据池 $FAC —— 先跑 run_v2_factory.sh(同样的 PILOT/ARMS 设置)"; exit 1
  fi
  step "臂 ${ARM} · 自提升 ${ROUNDS} 轮 → $OUT"
  python src/stage10_v2/e5_loop_v2.py --factory "$FAC" --out "$OUT" \
    --rounds "$ROUNDS" --kgen "$KGEN" --kscen "$KSCEN" --ngen-eval "$NGEN" \
    --nref "$NREF" --nexam "$NEXAM" --workers "$WORKERS" --seed "$SEED"
  el
done

step "汇总"
python - "$SUF" $ARMS <<'PY'
import json, os, sys
import numpy as np
suf, arms = sys.argv[1], sys.argv[2:]
print(f"{'臂':<10}{'轮':>5}{'r0 gap':>10}{'最好':>9}{'末5轮中位':>12}"
      f"{'末轮可行':>10}{'末轮崩盘':>10}{'池':>10}")
for a in arms:
    fp = f"outputs/v2_e5_{a}{suf}/trajectory.json"
    if not os.path.exists(fp):
        print(f"{a:<10}  (缺 {fp})"); continue
    t = json.load(open(fp)); g = [x["median_gap"] * 100 for x in t]
    print(f"{a:<10}{len(t):>5}{g[0]:>9.1f}%{min(g):>8.1f}%"
          f"{np.median(g[-5:]):>11.1f}%{t[-1]['feas_rate']*100:>9.0f}%"
          f"{t[-1]['fail']:>10}{t[-1]['pool']:>10}")
print("\nG2 检查清单(小试跑完必须逐条看):")
print("  1. 考卷题数是否接近 --nexam?剔除太多说明工况范围里无解题过多")
print("  2. r0 的可行率 ≥ 20%?过低则 gap 会被'找不到可行解'主导,曲线没意义")
print("  3. gap 是否单调下降?末 5 轮是否已平?没平就加轮数")
print("  4. 崩盘数是否随轮次下降到 0 附近?")
PY
el

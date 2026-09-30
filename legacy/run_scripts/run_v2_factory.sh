#!/usr/bin/env bash
# v2 数据工厂:按臂顺序建池 + 构训练对(防泄漏按束切分)
#
#   bash run_v2_factory.sh                 # 只建主臂 bio(全量)
#   PILOT=1 bash run_v2_factory.sh         # 小试规模,几分钟,用于 G2
#   ARMS="bio geo elastic none" bash run_v2_factory.sh   # 四臂消融全建
#
# 规模与耗时(A100 128 并行,约 0.7 s/仿真的经验值):
#   全量/臂: 500 块 × 120 设计 × 2 遍 = 120,000 次 ≈ 15–25 分钟
#   小试/臂: 170 块 ×  60 设计 × 2 遍 =  20,400 次 ≈ 3–5 分钟
set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${WORKERS:-128}"
ARMS="${ARMS:-bio}"
if [[ "${PILOT:-0}" == "1" ]]; then
  NGLOBAL="${NGLOBAL:-120}"; NPATH="${NPATH:-10}"; ND="${ND:-60}"; SUF="_pilot"
else
  NGLOBAL="${NGLOBAL:-375}"; NPATH="${NPATH:-25}"; ND="${ND:-120}"; SUF=""
fi
K="${K:-5}"; NREQ="${NREQ:-4}"; KTOP="${KTOP:-8}"

mkdir -p logs
LOG="logs/v2_factory_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1

t0=$(date +%s)
step() { echo -e "\n=========== [$(date +%H:%M:%S)] $* ==========="; }
el()   { echo "[累计 $(( ($(date +%s) - t0) / 60 )) 分 $(( ($(date +%s) - t0) % 60 )) 秒]"; }

echo "v2 工厂 | 臂=$ARMS  workers=$WORKERS  ${NGLOBAL}独立块 + ${NPATH}束×${K}步 × ${ND}设计"
echo "日志: $LOG"

for ARM in $ARMS; do
  OUT="outputs/v2_data_${ARM}${SUF}"
  step "臂 ${ARM} · 建池 → $OUT"
  python src/stage10_v2/factory_v2.py --arm "$ARM" \
    --nglobal "$NGLOBAL" --npath "$NPATH" --K "$K" --nd "$ND" \
    --workers "$WORKERS" --out "$OUT"
  el

  step "臂 ${ARM} · 构训练对(按 bid 切分)"
  python src/stage10_v2/dataset_v2.py --factory "$OUT/factory.jsonl" \
    --out "$OUT" --nreq "$NREQ" --ktop "$KTOP"
  el
done

step "汇总"
python - "$SUF" $ARMS <<'PY'
import json, os, sys
import numpy as np
suf, arms = sys.argv[1], sys.argv[2:]
print(f"{'臂':<10}{'块':>6}{'可行率':>9}{'训练对':>9}{'验证':>8}{'测试':>8}"
      f"{'路径束':>8}{'失败率':>9}")
for a in arms:
    d = f"outputs/v2_data_{a}{suf}"
    try:
        dm = json.load(open(os.path.join(d, "dataset_meta.json")))
        z = np.load(os.path.join(d, "dataset.npz"))
        nb = sum(1 for _ in open(os.path.join(d, "factory.jsonl")))
        fails = []
        for line in open(os.path.join(d, "factory.jsonl")):
            r = json.loads(line)
            fails += [f != "ok" for f in r["fail"]]
        p = json.load(open(os.path.join(d, "paths.json")))
        print(f"{a:<10}{nb:>6}{dm['feas_rate']*100:>8.1f}%{len(z['C_tr']):>9}"
              f"{len(z['C_va']):>8}{len(z['C_te']):>8}{len(p):>8}"
              f"{np.mean(fails)*100:>8.1f}%")
    except Exception as e:
        print(f"{a:<10}  (缺: {e})")
PY
el

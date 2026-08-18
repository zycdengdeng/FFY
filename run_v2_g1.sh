#!/usr/bin/env bash
# v2 · G1 验收门一键跑(A100)。总耗时预计 2–5 分钟。
#
#   bash run_v2_g1.sh
#   NDES=400 WORKERS=140 bash run_v2_g1.sh      # 覆盖默认
#
# 三步:① 生物先验自检(0 仿真) ② G1 反验 ③ 通道归因
# 判据(《质量耦合改造方案_v2.md》§三):
#   G1-a  同一设计沿 m 的 peak_a 相对极差 ≥ 10%
#   G1-b  可行率随 m 单调下降
# 不过就别进 G2 小试。
set -euo pipefail
cd "$(dirname "$0")"

WORKERS="${WORKERS:-128}"
NDES="${NDES:-200}"          # G1 设计数
NDES_CH="${NDES_CH:-120}"    # 通道归因设计数
OUT="${OUT:-outputs/gen_v2_g1}"
TERRAINS="${TERRAINS:-concrete,asphalt,turf,wetsand}"

mkdir -p logs "$OUT"
LOG="logs/v2_g1_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1

t0=$(date +%s)
step() { echo -e "\n=========== [$(date +%H:%M:%S)] $* ==========="; }
el()   { echo "[累计 $(( $(date +%s) - t0 )) 秒]"; }

echo "v2 G1 | workers=$WORKERS  G1 设计=$NDES  通道设计=$NDES_CH"
echo "地形=$TERRAINS   输出=$OUT   日志=$LOG"

step "① 生物先验自检(0 仿真:四臂盒子 + 往返精度 + 出实测全距比例)"
python src/stage10_v2/bioprior.py | tee "$OUT/bioprior_selfcheck.txt"
el

step "② G1 反验(设计 $NDES × 地形 × 质量 5)"
python src/stage10_v2/e15_mass_check.py \
  --ndes "$NDES" --workers "$WORKERS" --terrains "$TERRAINS" --out "$OUT"
el

step "③ 通道归因(5 组合 × 设计 $NDES_CH × 质量 5,地形 turf)"
python src/stage10_v2/e15_mass_check.py --channels \
  --ndes "$NDES_CH" --workers "$WORKERS" --terrain turf --out "$OUT"
el

step "汇总"
python - "$OUT" <<'PY'
import json, os, sys
d = sys.argv[1]
g = json.load(open(os.path.join(d, "e15_g1.json")))
c = json.load(open(os.path.join(d, "e15_channels.json")))
print("G1-a  各地形 peak_a 沿 m 的相对极差(判据 ≥10%):")
best = 0.0
for t, v in g["spreads"].items():
    p = v["peak"] * 100
    best = max(best, p)
    print(f"  {t:<10}{p:7.1f}%   (有效设计 {v['n']})")
print("\nG1-b  可行率随 m:")
mono = False
for t, row in g["feas"].items():
    ok = all(row[i] >= row[i+1] - 1e-9 for i in range(len(row)-1)) and row[0] > row[-1]
    mono = mono or ok
    print(f"  {t:<10}" + "".join(f"{v:6.0f}" for v in row) + ("   单调↓" if ok else ""))
print("\n通道归因(turf):")
for k, v in c["channels"].items():
    print(f"  {k:<24}{v['peak']*100:8.2f}%   (n={v['n']})")
print(f"\n[G1] {'✅ 两条都过,可进 G2 小试' if (best>=10 and mono) else '❌ 未过,回头调物理'}")
PY
el
echo "结果目录: $OUT"

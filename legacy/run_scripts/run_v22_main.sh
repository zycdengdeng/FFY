#!/usr/bin/env bash
# v2.2 主线:质量区间重新定标到产品范围(小型固定翼无人机 5–30 kg)
#   bash run_v22_main.sh
# 唯一改动:数据工厂的质量采样 1–12 kg → 4–36 kg(产品区间 5–30,两端留对数余量)。
# 物理与设计空间沿用 v2.1(9 维含姿态、髋阻尼统一式、放宽的 κ/τ 盒),不再变动。
# 起因:1–2 kg 没有对应的固定翼机型,而 12 kg 以上全是外推 ——
#       旧口径里 65% 的训练数据花在产品区间以下,顶端却够不到。
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-128}"; ROUNDS="${ROUNDS:-40}"
# 纪律:训练类指标进汇报前必须 ≥2 种子。写进脚本,不靠人记得。
SEEDS="${SEEDS:-0 1}"
OUT_F="${OUT_F:-outputs/v22_data_bio}"; OUT_E="${OUT_E:-outputs/v22_e5_bio}"
mkdir -p logs; LOG="logs/v22_main_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }

echo "=========== 0. 自检:5–30 kg 的结构可行性 ==========="
python - <<'PY'
import sys, numpy as np; sys.path.insert(0,'src/stage10_v2')
from bioprior import BioPrior
import physics_v2 as P
from factory_v2 import zeta_of_kc, M_RANGE
pr=BioPrior("bio",v21=True); rng=np.random.default_rng(0)
base={**P.SCEN_BIRD_X,"hip_damp_unified":True}
print(f"工厂质量范围 {M_RANGE}")
for m in (5,10,20,30):
    ok=0
    for _ in range(4):
        x=pr.expand(rng.random(9),float(m))
        r=P.eval_v2(tuple(x),float(m),1.2,kc=1e5,zeta_c=zeta_of_kc(1e5),npass=1,base=base)
        if r and not r.get("fail") and P.feasible_v2(r,98.1,0.024)[0]: ok+=1
    print(f"  {m:>2} kg  L1中心 {pr.l1_center(m):.0f}mm  随机设计可行 {ok}/4")
PY
el

echo "=========== 1. 数据工厂 v2.2(4–36 kg,12 万次落震) ==========="
python src/stage10_v2/factory_v2.py --v21 --m-range 4,36 --arm bio \
  --nglobal 375 --npath 25 --K 5 --nd 120 --npass 2 --workers "$WORKERS" --out "$OUT_F"
el
echo "=========== 2. 训练集 ==========="
python src/stage10_v2/dataset_v2.py --factory "$OUT_F/factory.jsonl" --out "$OUT_F"
el
for SD in $SEEDS; do
  if [ "$SD" = "0" ]; then O="$OUT_E"; else O="${OUT_E}_s${SD}"; fi
  echo "=========== 3. 自提升闭环 ×$ROUNDS  (seed $SD → $O) ==========="
  python src/stage10_v2/e5_loop_v2.py --factory "$OUT_F/factory.jsonl" --out "$O" \
    --rounds "$ROUNDS" --workers "$WORKERS" --seed "$SD"
  el
done
echo; echo "=========== 完成 ==========="
echo "下载: 每个种子目录下的 trajectory.json / model_meta.json / cvae_r39.pt"
echo "      种子目录: $OUT_E 以及 ${OUT_E}_s1 ..."
echo "口径提醒:v2.2 与 v2.1 的条件分布不同,gap/覆盖不能跨版本直接比;"
echo "         能比的是机制(通道使用、b_eff、走廊形状)。"
echo "日志: $LOG"

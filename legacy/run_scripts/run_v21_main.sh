#!/usr/bin/env bash
# v2.1 主线一键跑(A100):9 维设计(含触地姿态) + 髋阻尼统一式 + 放宽的 κ/τ 盒
#   bash run_v21_main.sh
# 三处物理改动(相对 v2),全部有实验依据:
#   1) 姿态 (thetaA, thetaK) 进设计向量,范围 = Duong 视频实测全距 113–160°/118–157°
#      —— P3:姿态是最大杠杆(峰值 +34%~184%),冻结在任何常数都不对
#   2) 髋阻尼统一为 c = τ·k —— P1:特例式把髋通道压死(响应 3.9%→13.1%),且抬高峰值
#   3) κ踝下界 1.5→0.75、κ髋 6→3、τ 0.01→0.005 —— P1/E20 的边界饱和旗标
# 老 7 维代码路径完全不变;新旧由 factory 的 --v21 与 meta 里的 u_dim/v21 区分。
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-128}"
ROUNDS="${ROUNDS:-40}"
OUT_F="${OUT_F:-outputs/v21_data_bio}"
OUT_E="${OUT_E:-outputs/v21_e5_bio}"
mkdir -p logs
LOG="logs/v21_main_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }

echo "=========== 0. 冒烟自检(9 维往返 + 单次实摔,约 10 秒) ==========="
python - <<'PY'
import sys, numpy as np; sys.path.insert(0,'src/stage10_v2')
from bioprior import BioPrior
import physics_v2 as P
from factory_v2 import zeta_of_kc
pr=BioPrior("bio",v21=True); u=np.random.default_rng(0).random((2,9))
x=pr.expand(u,12.0); assert x.shape==(2,9) and np.allclose(u,pr.contract(x,12.0))
r=P.eval_v2(tuple(x[0]),12.0,1.2,kc=1e5,zeta_c=zeta_of_kc(1e5),npass=1,
            base={**P.SCEN_BIRD_X,"hip_damp_unified":True})
assert not r.get("fail"), r
print(f"自检 ok: ndim={pr.ndim} κ盒={pr.kap_range} peak={r['peak_a']/9.81:.2f}g")
PY
el

echo "=========== 1. 数据工厂 v2.1(500 块 × 120 设计 × 2 遍 = 12 万次落震) ==========="
python src/stage10_v2/factory_v2.py --v21 --arm bio --nglobal 375 --npath 25 --K 5 --nd 120 --npass 2 --workers "$WORKERS" --out "$OUT_F"
el

echo "=========== 2. 训练集构建 ==========="
python src/stage10_v2/dataset_v2.py --factory "$OUT_F/factory.jsonl" --out "$OUT_F"
el

echo "=========== 3. 自提升闭环 ×$ROUNDS 轮 ==========="
python src/stage10_v2/e5_loop_v2.py --factory "$OUT_F/factory.jsonl" --out "$OUT_E" --rounds "$ROUNDS" --workers "$WORKERS"
el

echo
echo "=========== 完成 ==========="
echo "下载: $OUT_E/trajectory.json  $OUT_E/model_meta.json  $OUT_E/cvae_r*.pt(最后一个)"
echo "对照口径提醒:v2.1 与 v2 的物理不同,gap 等指标不能跨版本直接比;"
echo "能比的是机制(可行率走廊形状、b_eff、姿态通道是否被使用)。"
echo "日志: $LOG"

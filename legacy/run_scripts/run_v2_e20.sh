#!/usr/bin/env bash
# E20 生成侧体重走廊 + 出图:一键顺序跑(A100,约 50 分钟)
#   bash run_v2_e20.sh
# 可调(环境变量):
#   WORKERS=128   并行进程数
#   NZ=216        每格采多少隐变量。216≈50分钟;432 与 E18b 样本数完全对齐,≈1.6 小时
#   ARMS=bio      要跑的臂,逗号分隔(需已有 outputs/v2_e5_<arm>/cvae_r*.pt)
#   CONDS=...     只跑部分工况,逗号分隔
#   SKIP_SIM=1    跳过仿真,只重新出图(改了画图脚本时用)
set -euo pipefail
cd "$(dirname "$0")"
WORKERS="${WORKERS:-128}"
NZ="${NZ:-216}"
ARMS="${ARMS:-bio}"
OUT="${OUT:-outputs/v2_e20}"
RAND="${RAND:-outputs/v2_e18b}"
mkdir -p logs
LOG="logs/v2_e20_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }

echo "=========== 0. 前置检查 ==========="
# E18b 的随机基线是图 A 的对照,没有它图 A 只有一条线
if ! ls "$RAND"/e18b_*.json >/dev/null 2>&1; then
  echo "!! 找不到 $RAND/e18b_*.json —— 请先跑 e18b_corridor_multi.py,否则图 A 没有随机基线对照"
  exit 1
fi
# 中文字体:缺了图上全是方框,跑完才发现很亏,所以先查(cjkfont 会自己修缓存、扫 conda 目录)
if python src/stage10_v2/cjkfont.py; then :; else
  echo "!! 上面的办法装完字体再跑。确认不在意方框就用:  IGNORE_FONT=1 bash run_v2_e20.sh"
  echo "!! (注意 IGNORE_FONT=1 必须和 bash 写在同一行,单独一行设的变量子进程看不到)"
  [ "${IGNORE_FONT:-0}" = "1" ] || exit 1
fi
echo "臂=$ARMS  每格样本=$NZ  并行=$WORKERS  输出=$OUT"
el

if [ "${SKIP_SIM:-0}" = "1" ]; then
  echo "=========== 1. 跳过仿真(SKIP_SIM=1) ==========="
else
  echo "=========== 1. 实验 A:cVAE 条件生成 → 真摔(6工况 × 16+4级 × $NZ) ==========="
  python src/stage10_v2/e20_gen_corridor.py --workers "$WORKERS" --nz "$NZ" --arms "$ARMS" ${CONDS:+--conds "$CONDS"} --out "$OUT"
  el
fi

echo "=========== 2. 实验 B:出图(盒子vs模型 · 四指标成绩单 · 机型锚点) ==========="
for a in ${ARMS//,/ }; do
  echo "--- $a ---"
  python src/stage10_v2/e20_figs.py --gen "$OUT" --rand "$RAND" --arm "$a"
done
el

echo
echo "=========== 完成 ==========="
echo "下载这个目录即可: $OUT"
ls -la "$OUT"
echo "日志: $LOG"

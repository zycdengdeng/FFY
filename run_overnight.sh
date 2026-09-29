#!/usr/bin/env bash
# 过夜脚本(2026-09-29):顺序执行 E18c 扫描 → ER3 正式跑(seed 0 → 1 → 2)。
#   全程物理 = evalcfg(v2.5 工厂同款);每步独立日志;任一步失败不中断后面;
#   全部可中断续跑(已完成的 json 自动跳过)。
#   用法(A100,tmux 里):cd /mnt/zihanw/FFY && git pull && bash run_overnight.sh
#   早上看:logs/overnight_*.log 末尾的汇总;下载 outputs/e18c_scan/ 与 outputs/er3_full/。
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs/e18c_scan outputs/er3_full
LOG="logs/overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-96}"
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟] $(date '+%m-%d %H:%M')"; }
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; local rc=$?; echo "退出码 $rc"; el; return $rc; }

echo "开始 $(date)  workers=$W"
python src/stage10_v2/evalcfg.py || { echo "!!!! evalcfg 导入失败,停"; exit 1; }

# ---- 1 · E18c 绝对腿长扫描(6 工况 × 9 质量 × 19 长度 × 96 探针) ----
run "1/4 · E18c 绝对腿长扫描" \
  python src/stage10_v2/e18c_abs_scan.py --workers "$W" --nprobe 96 --out outputs/e18c_scan
RC1=$?
# 扫完立刻在机上出一版分解表(纯 numpy 后处理,几秒钟),早上直接看
run "1b · E-R4b 分解(机上预览)" \
  python src/stage10_v2/er4b_decompose.py --scan outputs/e18c_scan --out outputs/er4b --boot 200

# ---- 2 · ER3 正式跑:seed 0 全部 → seed 1 → seed 2 ----
B="${BUDGET:-6000}"
CONDS="${CONDS:-turf1.2 wetsand1.2 concrete1.2 wetsand2.0}"
MASSES="${MASSES:-4 5.77 8.32 12 17.3 24.96 36}"
for S in 0 1 2; do
  echo; echo "=========== $((S+2))/4 · ER3 正式跑 seed=$S ==========="
  for C in $CONDS; do
    for M in $MASSES; do
      for SP in common bio; do
        echo "--- seed=$S cond=$C m=$M space=$SP de budget=$B ---"
        python src/stage10_v2/er3_search.py --cond "$C" --space "$SP" --mass "$M" \
          --method de --budget "$B" --seed "$S" --workers "$W" --out outputs/er3_full
      done
    done
  done
  el
done

# ---- 汇总 ----
echo; echo "=========== 汇总 $(date) ==========="
echo "E18c 退出码 $RC1;扫描文件:"; ls -1 outputs/e18c_scan/ 2>/dev/null
echo "ER3 完成数:$(ls outputs/er3_full/er3_*.json 2>/dev/null | wc -l) / 168"
[ -f outputs/er4b/er4b_tables.md ] && { echo; echo "---- E-R4b 分解表(机上预览)----"; cat outputs/er4b/er4b_tables.md; }
el
echo "全部结束。下载:outputs/e18c_scan/  outputs/er4b/  outputs/er3_full/  $LOG"

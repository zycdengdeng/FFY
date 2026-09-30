#!/usr/bin/env bash
# 修复后重扫(2026-09-30):顺序执行两遍 E18c——①标准闸(s_max=24 mm) ②行程闸敏感性(40 mm)。
#   物理 = 修复后的 physics_v2(D-6 力矩直读 / D-7 配对 / D-8 相对行程 / D-9 滑移窗口最大)。
#   任一步失败不中断后面;已完成的工况 json 自动跳过,可中断续跑。
#   用法(A100,tmux):cd /mnt/zihanw/FFY && git pull && bash run_rescan.sh
#   完毕后下载:outputs/e18c_scan_fix/  outputs/e18c_scan_fix_s40/  outputs/er4b_fix/  本日志
set -u
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/rescan_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-96}"
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟] $(date '+%m-%d %H:%M')"; }

echo "开始 $(date)  workers=$W  physics_v2 修复版"
python src/stage10_v2/evalcfg.py || { echo "!!!! evalcfg 导入失败,停"; exit 1; }

echo; echo "=========== 1/2 · E18c 重扫(标准闸 s_max=24 mm) ==========="
OUT=outputs/e18c_scan_fix bash run_e18c.sh
echo "退出码 $?"; el

echo; echo "=========== 1b · ER4b 分解(机上预览) ==========="
python src/stage10_v2/er4b_decompose.py --scan outputs/e18c_scan_fix --out outputs/er4b_fix --boot 200
echo "退出码 $?"; el

echo; echo "=========== 2/2 · E18c 行程闸敏感性(s_max=40 mm) ==========="
OUT=outputs/e18c_scan_fix_s40 FFY_SMAX=0.040 bash run_e18c.sh
echo "退出码 $?"; el

echo; echo "=========== 汇总 $(date) ==========="
for d in outputs/e18c_scan_fix outputs/e18c_scan_fix_s40; do echo "$d: $(ls "$d" 2>/dev/null | wc -l) 个文件"; done
[ -f outputs/er4b_fix/er4b_tables.md ] && { echo; echo "---- 修复后分解表(24 mm 闸)----"; cat outputs/er4b_fix/er4b_tables.md; }
el
echo "全部结束。下载:outputs/e18c_scan_fix/  outputs/e18c_scan_fix_s40/  outputs/er4b_fix/  $LOG"

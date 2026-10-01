#!/usr/bin/env bash
# ENG-prior 工程先验对照(2026-10-01,协议 R-H):顺序执行
#   1) E18c 开发集扫描(s_max=24 mm;质量 4.5–30 kg 5 档;工况 soil/loam × 1.3/1.7;48 探针;种子 1)= 18,240 次评价
#   2) E-R5 分析:对照测试集 outputs/e18c_scan_fix(机上预览)
#   3) 敏感性:开发集 40 mm 闸重扫 + 对照 outputs/e18c_scan_fix_s40
#   已完成的工况 json 自动跳过,可中断续跑。
#   用法(A100,tmux):cd /mnt/zihanw/FFY && git pull && bash run_engprior.sh
#   完毕后下载:outputs/e18c_dev/  outputs/e18c_dev_s40/  outputs/er5_engprior/  outputs/er5_engprior_s40/  本日志
set -u
cd "$(dirname "$0")"
mkdir -p logs
LOG="logs/engprior_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-96}"
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟] $(date '+%m-%d %H:%M')"; }
DEV="--conds soil1.3,soil1.7,loam1.3,loam1.7 --nm 5 --mlo 4.5 --mhi 30 --nprobe 48 --seed 1 --workers $W"

echo "开始 $(date)  workers=$W"
python src/stage10_v2/evalcfg.py || { echo "!!!! evalcfg 导入失败,停"; exit 1; }
for d in outputs/e18c_scan_fix outputs/e18c_scan_fix_s40; do [ -f "$d/stamp.json" ] || echo "!! 缺测试集 $d(分析步会失败,扫描照跑)"; done

echo; echo "=========== 1/3 · E18c 开发集扫描(s_max=24 mm) ==========="
python src/stage10_v2/e18c_abs_scan.py $DEV --out outputs/e18c_dev
echo "退出码 $?"; el

echo; echo "=========== 2/3 · E-R5 分析(对照 e18c_scan_fix) ==========="
python src/stage10_v2/er5_engprior.py --dev outputs/e18c_dev --test outputs/e18c_scan_fix --out outputs/er5_engprior --reps 20 --boot 200
echo "退出码 $?"; el

echo; echo "=========== 3/3 · 敏感性:开发集 s_max=40 mm + 对照 e18c_scan_fix_s40 ==========="
FFY_SMAX=0.040 python src/stage10_v2/e18c_abs_scan.py $DEV --out outputs/e18c_dev_s40
echo "退出码 $?"; el
python src/stage10_v2/er5_engprior.py --dev outputs/e18c_dev_s40 --test outputs/e18c_scan_fix_s40 --out outputs/er5_engprior_s40 --reps 20 --boot 200
echo "退出码 $?"; el

echo; echo "=========== 汇总 $(date) ==========="
[ -f outputs/er5_engprior/er5_tables.md ] && { echo "---- 主结果(24 mm)----"; cat outputs/er5_engprior/er5_tables.md; }
el
echo "全部结束。下载:outputs/e18c_dev/  outputs/e18c_dev_s40/  outputs/er5_engprior/  outputs/er5_engprior_s40/  $LOG"

#!/usr/bin/env bash
# PGLS 主跑:P2 闸门 → 过了才投 P3。R 缺包时先装(conda 环境内)。
set -uo pipefail
cd "$(dirname "$0")"
command -v Rscript >/dev/null || { echo "缺 R:先执行  conda install -y -c conda-forge r-base r-ape r-phylolm r-nlme"; exit 1; }
Rscript -e 'suppressMessages({library(ape);library(phylolm);library(nlme)})' 2>/dev/null || { echo "缺 R 包:conda install -y -c conda-forge r-ape r-phylolm r-nlme"; exit 1; }
mkdir -p logs
LOG="logs/pgls_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=========== P2 单树闸门 ==========="
Rscript src/stage6_surrogate/pgls_p2_single.R . || { echo "P2 未过闸门,按预案处理,P3 不跑。"; exit 1; }
echo; echo "=========== P3 100 树 × 2 骨架 ==========="
Rscript src/stage6_surrogate/pgls_p3_full.R .
echo "日志: $LOG"

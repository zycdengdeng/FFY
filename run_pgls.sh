#!/usr/bin/env bash
# PGLS 主跑:P2 闸门 → 过了才投 P3。R 缺包时先装(conda 环境内)。
set -uo pipefail
cd "$(dirname "$0")"
command -v Rscript >/dev/null || { echo "缺 R:先执行  conda install -y -c conda-forge r-base r-ape r-phylolm r-nlme"; exit 1; }
Rscript -e 'suppressMessages({library(ape);library(phylolm);library(nlme)})' 2>/dev/null || { echo "缺 R 包:conda install -y -c conda-forge r-ape r-phylolm r-nlme"; exit 1; }
# ---- 守门:输入必须齐,缺文件不要伪装成「闸门未过」 ----
MISS=0
for f in data/birdtree/pgls_data_matched.csv data/birdtree/hackett_100.nwk data/birdtree/ericson_100.nwk; do
  [ -s "$f" ] || { echo "✗ 缺输入: $f"; MISS=1; }
done
if [ "$MISS" = "1" ]; then
  cat <<'MSG'

这不是科学结论问题,是文件没同步到本机。
这三个文件在 Windows 端 FFY/data/birdtree/ 下,已打包为 pgls_inputs.tar.gz(37 MB)。
传到本机 FFY/data/birdtree/ 后解包:
    tar xzf data/birdtree/pgls_inputs.tar.gz -C data/birdtree/
再重跑 bash run_pgls.sh
MSG
  exit 2
fi
echo "✓ 输入齐备:$(wc -l < data/birdtree/pgls_data_matched.csv) 行分析表 + 两骨架各 100 棵树"

mkdir -p logs
LOG="logs/pgls_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "=========== P2 单树闸门 ==========="
Rscript src/stage6_surrogate/pgls_p2_single.R . || { echo "P2 未过闸门,按预案处理,P3 不跑。"; exit 1; }
echo; echo "=========== P3 100 树 × 2 骨架 ==========="
Rscript src/stage6_surrogate/pgls_p3_full.R .
echo "日志: $LOG"

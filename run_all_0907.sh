#!/usr/bin/env bash
# ============================================================================
# 一次跑齐:A 段 v2.3 收尾(E20/E21 换 r12 检查点) · B 段 系统发育全家桶
#   bash run_all_0907.sh
# 两段各自独立守门,A 段挂了不影响 B 段(B 段是这次的重点)。
# ============================================================================
set -uo pipefail
cd "$(dirname "$0")"
W="${WORKERS:-128}"
mkdir -p logs outputs/phylo
LOG="logs/all_0907_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
T0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-T0)/60 )) 分钟]"; }
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; echo "退出码 $?"; el; }

echo "###########################################################"
echo "# A 段 · v2.3 收尾(E18b 已完成,此处只补 E20/E21 @r12)"
echo "###########################################################"
if [ ! -f outputs/v23_e5_bio/cvae_r12.pt ]; then
  echo "✗ 缺 outputs/v23_e5_bio/cvae_r12.pt,跳过 A 段(不影响 B 段)"
else
  for SD in 0 1; do
    SRC="outputs/v23_e5_bio"; TAG=""
    [ "$SD" = "1" ] && SRC="outputs/v23_e5_bio_s1" && TAG="_s1"
    R="outputs/v23_root_r12${TAG}"; mkdir -p "$R/v2_e5_bio"
    cp -f "$SRC/cvae_r12.pt" "$SRC/model_meta.json" "$R/v2_e5_bio/" 2>/dev/null
    run "A1 · E20 @r12 seed$SD" \
      python src/stage10_v2/e20_gen_corridor.py --v21 --foot bearing --outroot "$R" \
        --mgrid 2,40,16 --anchors "5:产品下端,12:样机档,30:产品上端" \
        --nz 216 --workers "$W" --out "outputs/v23_e20_r12${TAG}"
  done
  run "A2 · E21 @r12" \
    python src/stage10_v2/e21_bird_vs_gen.py --v21 --foot bearing \
      --ckpt outputs/v23_e5_bio/cvae_r12.pt --workers "$W" --out outputs/v23_e21_r12
fi

echo
echo "###########################################################"
echo "# B 段 · 系统发育全家桶(本次重点)"
echo "###########################################################"
MISS=0
for f in data/birdtree/pgls_data_full.csv data/birdtree/hackett_100.nwk data/birdtree/ericson_100.nwk; do
  [ -s "$f" ] || { echo "✗ 缺输入: $f"; MISS=1; }
done
if [ "$MISS" = "1" ]; then
  cat <<'MSG'
B 段缺输入。pgls_data_full.csv(1.2 MB)是本次新生成的,需要从 Windows 端
FFY/data/birdtree/ 传到本机同路径。两个 .nwk 上次已传过,若也缺就一并补。
传好后重跑 bash run_all_0907.sh(A 段已完成的会重跑,不想重跑就直接
Rscript src/stage6_surrogate/phylo_full.R .)
MSG
  exit 2
fi
command -v Rscript >/dev/null || { echo "✗ 缺 R"; exit 1; }
Rscript -e 'suppressMessages({library(ape);library(phylolm)})' 2>/dev/null \
  || { echo "✗ 缺 R 包:conda install -y -c conda-forge r-ape r-phylolm"; exit 1; }
echo "✓ B 段输入与环境齐备"
run "B · phylo_full.R" Rscript src/stage6_surrogate/phylo_full.R .

echo
echo "=========== 全部结束 ==========="
echo "下载:outputs/phylo(全部 CSV) · outputs/v23_e20_r12(+_s1) · outputs/v23_e21_r12"
echo "日志: $LOG"

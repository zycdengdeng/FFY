#!/usr/bin/env bash
# A 组 + B 组一次跑完（E22–E27）。可以挂后台走人。
#
#   cd /mnt/zihanw/FFY
#   nohup bash run_ab_group.sh > /dev/null 2>&1 &
#   tail -f logs/ab_group_*.log          # 想看进度的话
#
# 约定沿用 run_v23_main.sh：不用 set -e（前段挂了后段照跑），每段完打「[累计 N 分钟]」，
# 全程 tee 进 logs/。跑完在 logs/ 下留一个 AB组结果摘要.txt，把它发回来就行。
#
# 前置：conda 环境 ffy 已激活（提示符是 (ffy) 就对了）。
set -uo pipefail
cd "$(dirname "$0")"
ROOT="$(pwd)"
W="${WORKERS:-128}"
NE22="${NE22:-3000}"      # E22 抽样重评多少个可行设计
NDES="${NDES:-24}"        # E26 每个工况生成多少个设计
export OMP_NUM_THREADS=1  # 不加的话 128 个进程各开一堆线程互相抢核

mkdir -p logs
LOG="logs/ab_group_$(date +%Y%m%d_%H%M%S).log"
SUM="logs/AB组结果摘要.txt"
exec > >(tee -a "$LOG") 2>&1

t0=$(date +%s)
el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; echo "退出码 $?"; el; }

echo "工作目录 $ROOT"
echo "workers  $W    E22 样本 $NE22    E26 每工况设计数 $NDES"
echo "日志     $LOG"

# ---------------------------------------------------------------- 0 · 前置检查
echo; echo "=========== 0/9 · 前置检查 ==========="
MISS=0
for f in src/stage10_v2/agroup_io.py src/stage10_v2/e22_dl_scaling.py \
         src/stage10_v2/e22_bio_ref.json src/stage10_v2/e26_q1_sweep.py \
         src/stage10_v2/e27_vertical_structural.py \
         src/stage10_v2/e23_box_saturation.py src/stage10_v2/e24_rebound_gate.py \
         src/stage10_v2/e25_posterior_collapse.py; do
  if [ ! -f "$f" ]; then echo "  [缺] $f"; MISS=1; else echo "  [有] $f"; fi
done
for f in outputs/v23_data_bio/factory.jsonl outputs/v23_data_bio/factory_meta.json \
         outputs/v23_data_bio/dataset.npz outputs/v23_e5_bio/cvae_r12.pt; do
  if [ ! -f "$f" ]; then echo "  [缺] $f"; MISS=1; else echo "  [有] $f"; fi
done
if [ "$MISS" = "1" ]; then
  echo
  echo "有文件缺失。脚本文件从本地传上来："
  echo "  scp src/stage10_v2/{agroup_io,e22_dl_scaling,e23_box_saturation,e24_rebound_gate,e25_posterior_collapse,e26_q1_sweep,e27_vertical_structural}.py src/stage10_v2/e22_bio_ref.json wzh@<A100>:$ROOT/src/stage10_v2/"
  exit 1
fi
python -c "import exudyn,torch,numpy,matplotlib;print('  exudyn',exudyn.__version__,'| torch',torch.__version__,'| numpy',numpy.__version__)" || {
  echo "  依赖导入失败 —— 确认 conda 环境是 ffy"; exit 1; }
el

# ---------------------------------------------------------------- A 组（秒级）
run "1/9 · E23 盒壁饱和（秒级）" \
  python src/stage10_v2/e23_box_saturation.py --fig

run "2/9 · E24 回弹软闸重打分（秒级）" \
  python src/stage10_v2/e24_rebound_gate.py --fig

run "3/9 · E25 后验塌缩（纯推理，秒级）" \
  python src/stage10_v2/e25_posterior_collapse.py --fig

# ---------------------------------------------------------------- E22（重）
run "4/9 · E22 抽样重评取 D（约 $NE22 次落震，可中断续跑）" \
  python src/stage10_v2/e22_dl_scaling.py --n "$NE22" --workers "$W"

run "5/9 · E22 标度分析（秒级，不需要 exudyn）" \
  python src/stage10_v2/e22_dl_scaling.py --analyze-only --fig

# ---------------------------------------------------------------- E26 / E27
run "6/9 · E26 纯几何预检（零仿真）" \
  python src/stage10_v2/e26_q1_sweep.py --ndes "$NDES" --geom-only

run "7/9 · E26 q1_0 扫描（约 $((NDES*6*11)) 次落震，可中断续跑）" \
  python src/stage10_v2/e26_q1_sweep.py --ndes "$NDES" --workers "$W"

run "8/9 · E26 敏感度分析（秒级）" \
  python src/stage10_v2/e26_q1_sweep.py --ndes "$NDES" --analyze-only --fig

run "9/9 · E27 竖直化的结构收益（零新增仿真，复用 E26 缓存）" \
  python src/stage10_v2/e27_vertical_structural.py --fig

# ---------------------------------------------------------------- 附：v2.3 真实时长
echo; echo "=========== 附 · run_v23_main.sh 的真实时长 ==========="
if ls logs/v23_main_*.log >/dev/null 2>&1; then
  grep -hE "^===========|^\[累计" logs/v23_main_*.log | tail -40
else
  echo "logs/ 下没有 v23_main_*.log —— 换个名字找找：ls logs/"
  ls logs/ 2>/dev/null | head -20
fi
el

# ---------------------------------------------------------------- 摘要
{
  echo "###############  A 组 + B 组结果摘要  ###############"
  echo "生成时间 $(date '+%F %T')    总耗时 $(( ($(date +%s)-t0)/60 )) 分钟"
  echo "完整日志 $LOG"
  echo
  echo "-------------------  先看这三个数  -------------------"
  echo -n "E22 夹逼合计（>20% 则该结论作废）： "
  grep -m1 "全部段合计" "$LOG" || echo "（E22 没跑出来）"
  echo -n "E26 敏感度中位（<10% 保持常数 / >25% 升第10维）： "
  grep -m1 "相对变化" "$LOG" || echo "（E26 没跑出来）"
  echo -n "E27 腿质量变化（<-15% 才算显式取舍）： "
  grep -m1 -E "^   腿质量" "$LOG" || echo "（E27 没跑出来）"
  echo
  echo "判定行："
  grep -hE "^判定：|^结论：" "$LOG" | sed 's/^/   /'
  echo
  echo "=================== E22 · D/L 标度 ==================="
  sed -n '/^\[e22\] 分析/,/e22_dl_scaling\.json/p' "$LOG"
  echo
  echo "=================== E23 · 盒壁饱和 ==================="
  sed -n '/^\[e23\] 块/,/e23_box_saturation\.json/p' "$LOG"
  echo
  echo "=================== E24 · 回弹软闸 ==================="
  sed -n '/^\[e24\] 块/,/e24_rebound_gate\.json/p' "$LOG"
  echo
  echo "=================== E25 · 后验塌缩 ==================="
  sed -n '/^\[e25\] cvae/,/e25_posterior_collapse\.json/p' "$LOG"
  echo
  echo "=================== E26a · 纯几何预检 ==================="
  sed -n '/纯几何预检/,/e26_geom\.json/p' "$LOG"
  echo
  echo "=================== E26b · q1_0 敏感度 ==================="
  sed -n '/^\[e26\] 有效点/,/e26_q1_sweep\.json/p' "$LOG"
  echo
  echo "=================== E27 · 竖直化的结构收益 ==================="
  sed -n '/^\[e27\] 有效点/,/e27_vertical_structural\.json/p' "$LOG"
  echo
  echo "=================== v2.3 真实时长 ==================="
  grep -hE "^===========|^\[累计" logs/v23_main_*.log 2>/dev/null | tail -40
  echo
  echo "=================== 各段耗时 ==================="
  grep -hE "^===========|^\[累计" "$LOG"
} > "$SUM"

# ---------------------------------------------------------------- 打包带回
PK="logs/ab_group_产物.tar.gz"
tar czf "$PK" \
  "$SUM" "$LOG" \
  outputs/v23_data_bio/e2[234]_*.json outputs/v23_data_bio/e2[234]_*.png \
  outputs/v23_e5_bio/e25_*.json outputs/v23_e5_bio/e25_*.png \
  outputs/v23_e26/e2[67]_*.json outputs/v23_e26/e2[67]_*.png 2>/dev/null

echo; echo "=========== 全部结束（$(( ($(date +%s)-t0)/60 )) 分钟）==========="
echo
echo "摘要   $SUM       ← 把这个文件的内容发回来就够了"
echo "打包   $PK"
echo
echo "取回：scp wzh@<A100>:$ROOT/$PK ."
echo
echo "重点看三个数："
echo "  · E22 ①夹逼诊断的「全部段合计」   —— 超 20% 这条结论作废"
echo "  · E26 ①的「中位」                —— <10% 保持常数 / >25% 升为第 10 维"
echo "  · E27 ②的「腿质量」              —— 减重 >15% 才算显式取舍，否则关掉"

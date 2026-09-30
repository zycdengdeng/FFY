#!/usr/bin/env bash
# E18b / E20 按修正后的统计口径重跑(2026-09-02)
#   bash run_restat.sh
#
# 改了什么:原来的可行率 f = ok/n,把三件不同的事塞进一个分母 ——
#   ok          可行
#   infeasible  求解成功、模型有效,但违反 gcap/smax/slenderness/massbudget
#   invalid     模型失效(deep_sink:足端侵入超过球半径,罚接触模型不适用)
#   unsolved    数值失败(solver/nonfinite/none/collapse)
# 后两类是"判不了",不该算进不可行。新增 f_judged = ok/(ok+infeasible),
# 旧口径 f 一并保留以便对照。
#
# 烟测已见影响不小:湿沙 2.0 有 50% 样本是 deep_sink,
# 旧口径 0.375 → 新口径 0.750,整整差一倍。
#
# 参数与 2026-09-01 夜跑完全一致,所以新旧结果逐格可比。
set -uo pipefail
cd "$(dirname "$0")"
W="${WORKERS:-128}"
ROOT="${ROOT:-outputs/v22_root}"
mkdir -p logs "$ROOT/v2_e5_bio"
cp -f outputs/v22_e5_bio/cvae_r39.pt outputs/v22_e5_bio/cvae_r40.pt \
      outputs/v22_e5_bio/model_meta.json "$ROOT/v2_e5_bio/" 2>/dev/null
LOG="logs/restat_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; echo "退出码 $?"; el; }

run "1/2 · E20 生成走廊(新统计口径)" \
  python src/stage10_v2/e20_gen_corridor.py --v21 --outroot "$ROOT" \
    --mgrid 2,40,16 --anchors "5:产品下端,12:样机档,30:产品上端" \
    --nz 216 --workers "$W" --out outputs/v22_e20_rs

run "2/2 · E18b 四臂走廊(新统计口径)" \
  python src/stage10_v2/e18b_corridor_multi.py --v21 \
    --mlo 2 --mhi 40 --nu 9 --nm 16 --nprobe 48 \
    --workers "$W" --out outputs/v22_e18b_rs

echo; echo "=========== 完成 ==========="
echo "产出: outputs/v22_e20_rs · outputs/v22_e18b_rs"
echo "新增字段: f_judged / pooled_judged(真实可行率) · r_unsolved · r_invalid"
echo "         旧字段 f / pooled 保留 —— 两者之差就是「判不了的样本占多少」。"
echo "日志: $LOG"

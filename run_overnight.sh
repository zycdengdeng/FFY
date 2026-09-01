#!/usr/bin/env bash
# 夜跑:五个实验,按「便宜且高价值」在前排序,前面挂了后面还能继续。
#   bash run_overnight.sh
# 1 P6 屈曲校核        几分钟   ← 国际会议前的真实缺口,先做
# 2 P7 关节 Zener 化   ~30 分钟 ← 串联弹性的代价,自校验通过才产出
# 3 E21 真鸟 vs 生成   ~10 分钟
# 4 E20 生成走廊       ~30 分钟
# 5 E18b 四臂走廊      ~2 小时  ← 最大,放最后
# 全部用 v2.2 口径(9 维含姿态 + 髋阻尼统一式 + 4–36 kg 模型)。
set -uo pipefail                      # 注意:不用 -e,单个实验失败不影响后续
cd "$(dirname "$0")"
W="${WORKERS:-128}"
CKPT="${CKPT:-outputs/v22_e5_bio/cvae_r39.pt}"
ROOT="${ROOT:-outputs/v22_root}"      # E20 需要 <root>/v2_e5_bio/ 结构
mkdir -p logs "$ROOT/v2_e5_bio"
cp -f outputs/v22_e5_bio/cvae_r39.pt outputs/v22_e5_bio/cvae_r40.pt \
      outputs/v22_e5_bio/model_meta.json "$ROOT/v2_e5_bio/" 2>/dev/null
LOG="logs/overnight_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; echo "退出码 $?"; el; }

run "1/5 · P6 屈曲校核(Euler + 局部壳屈曲)" \
  python src/stage10_v2/p6_buckling.py --ckpt "$CKPT" \
    --masses 5,8,12,20,30 --out outputs/v2_p6

run "2/5 · P7 关节 Zener 化(串联弹性的代价)" \
  python src/stage10_v2/p7_zener.py --ckpt "$CKPT" \
    --masses 5,12,30 --ratios 3,10,30,100,1000 --workers "$W" --out outputs/v2_p7

run "3/5 · E21 真鸟骨长 vs 生成骨长(v2.2 口径)" \
  python src/stage10_v2/e21_bird_vs_gen.py --v21 --ckpt "$CKPT" \
    --workers "$W" --out outputs/v22_e21

run "4/5 · E20 生成走廊(v2.2 口径,产品区间锚点)" \
  python src/stage10_v2/e20_gen_corridor.py --v21 --outroot "$ROOT" \
    --mgrid 2,40,16 --anchors "5:产品下端,12:样机档,30:产品上端" \
    --nz 216 --workers "$W" --out outputs/v22_e20

run "5/5 · E18b 四臂可行走廊(v2.2 口径)" \
  python src/stage10_v2/e18b_corridor_multi.py --v21 \
    --mlo 2 --mhi 40 --nu 9 --nm 16 --nprobe 48 \
    --workers "$W" --out outputs/v22_e18b

echo; echo "=========== 全部结束 ==========="
echo "产出目录: outputs/v2_p6 · v2_p7 · v22_e21 · v22_e20 · v22_e18b"
echo "口径提醒:E18b/E20/E21 已切到 v2.2(9 维 + 统一阻尼 + 新盒),"
echo "         与旧的 v2_e18b / v2_e20 / v2_e21 不可直接比,是替换不是对照。"
echo "日志: $LOG"

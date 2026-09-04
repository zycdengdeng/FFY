#!/usr/bin/env bash
# v2.3 补漏:①E18b 用修好的代码真跑一次 ②E20/E21 换漂移前的 r12 检查点重验
#   bash run_v23_fix.sh
# 三段按「先决定性、后验证性」排;每段自带守门,守门不过就停,不浪费机时。
set -uo pipefail
cd "$(dirname "$0")"
W="${WORKERS:-128}"
mkdir -p logs
LOG="logs/v23_fix_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; echo "退出码 $?"; el; }

# ---- 守门 0:确认 e18b 代码带着修复,不带就别跑 ----
if ! grep -q "FOOT_MODE = args.foot" src/stage10_v2/e18b_corridor_multi.py; then
  echo "✗ e18b_corridor_multi.py 是旧代码(缺 FOOT_MODE = args.foot),先同步再跑。"; exit 1
fi
echo "✓ e18b 代码已含修复"

# ---- 1/3 · E18b 四臂走廊(v2.3 真·解绑版) ----
run "1/3 · E18b(bearing)" \
  python src/stage10_v2/e18b_corridor_multi.py --v21 --foot bearing \
    --mlo 2 --mhi 40 --nu 9 --nm 16 --nprobe 48 --workers "$W" --out outputs/v23_e18b_fix2

# 守门 1:产出必须和 v2.2(leg)不同,相同即代码仍未生效
if cmp -s outputs/v23_e18b_fix2/e18b_turf1.2.json outputs/v22_e18b_rs/e18b_turf1.2.json; then
  echo "✗ v23_e18b_fix2 与 v22_e18b_rs 逐字节相同 —— 修复仍未生效,停。"; exit 1
fi
echo "✓ E18b 产出与 leg 口径不同,解绑生效"

# ---- 2/3 · E20 换 r12 检查点(双种子都验) ----
for SD in 0 1; do
  SRC="outputs/v23_e5_bio"; TAG=""
  [ "$SD" = "1" ] && SRC="outputs/v23_e5_bio_s1" && TAG="_s1"
  R="outputs/v23_root_r12${TAG}"
  mkdir -p "$R/v2_e5_bio"
  cp -f "$SRC/cvae_r12.pt" "$SRC/model_meta.json" "$R/v2_e5_bio/"
  run "2/3 · E20 @r12 seed$SD" \
    python src/stage10_v2/e20_gen_corridor.py --v21 --foot bearing --outroot "$R" \
      --mgrid 2,40,16 --anchors "5:产品下端,12:样机档,30:产品上端" \
      --nz 216 --workers "$W" --out "outputs/v23_e20_r12${TAG}"
done

# ---- 3/3 · E21 换 r12(seed 0) ----
run "3/3 · E21 @r12" \
  python src/stage10_v2/e21_bird_vs_gen.py --v21 --foot bearing \
    --ckpt outputs/v23_e5_bio/cvae_r12.pt --workers "$W" --out outputs/v23_e21_r12

echo; echo "=========== 全部结束 ==========="
echo "产出: outputs/v23_e18b_fix2 · v23_e20_r12(+_s1) · v23_e21_r12"
echo "下载这三(四)个目录即可;r40 的 v23_e20 / v23_e21 已作废,别混用。"
echo "日志: $LOG"

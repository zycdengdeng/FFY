#!/usr/bin/env bash
# v2.3 全量:足端解绑(r_foot 由 (m, k_c) 派生,不再绑在 L1 上)
#   bash run_v23_main.sh
#
# 唯一物理改动:foot_mode = bearing。其余(9 维含姿态、髋阻尼统一式、4–36 kg)沿用 v2.2,
# 所以跑完能干净归因到"足端解绑"这一件事。
#
# 为什么改:r_foot = 0.20·L1 把腿长与接地面积绑成同一变量,
# 四臂消融里低 b 的臂自动获得小脚 → 软地面 deep_sink → 模型失效。
# 这个劣势与被检验的异速律假设无关,是混杂。详见《足端解绑_方案与首轮验证.md》。
#
# 分五段,按「先便宜后昂贵、先决定性后补充」排;不用 set -e,前段挂了后段照跑。
set -uo pipefail
cd "$(dirname "$0")"
W="${WORKERS:-128}"; ROUNDS="${ROUNDS:-40}"; SEEDS="${SEEDS:-0 1}"
OUT_F="${OUT_F:-outputs/v23_data_bio}"; OUT_E="${OUT_E:-outputs/v23_e5_bio}"
ROOT="${ROOT:-outputs/v23_root}"
mkdir -p logs "$ROOT/v2_e5_bio"
LOG="logs/v23_main_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; echo "退出码 $?"; el; }

# ---- 0 · 先花 1 小时确认解绑在大样本下真的成立,不成立就别跑后面 5 小时 ----
run "0/5 · P8 足端解绑正式验证(64 探针 × 5 质量)" \
  python src/stage10_v2/p8_foot_decouple.py \
    --masses 5,8,12,20,30 --nprobe 64 --workers "$W" --out outputs/v2_p8

run "1/5 · 数据工厂 v2.3(足端解绑,12 万次落震)" \
  python src/stage10_v2/factory_v2.py --v21 --foot bearing --m-range 4,36 --arm bio \
    --nglobal 375 --npath 25 --K 5 --nd 120 --npass 2 --workers "$W" --out "$OUT_F"

run "2/5 · 训练集" \
  python src/stage10_v2/dataset_v2.py --factory "$OUT_F/factory.jsonl" --out "$OUT_F"

for SD in $SEEDS; do
  if [ "$SD" = "0" ]; then O="$OUT_E"; else O="${OUT_E}_s${SD}"; fi
  run "3/5 · 自提升闭环 ×$ROUNDS (seed $SD → $O)" \
    python src/stage10_v2/e5_loop_v2.py --factory "$OUT_F/factory.jsonl" --out "$O" \
      --rounds "$ROUNDS" --workers "$W" --seed "$SD"
done

cp -f "$OUT_E"/cvae_r39.pt "$OUT_E"/cvae_r40.pt "$OUT_E"/model_meta.json \
      "$ROOT/v2_e5_bio/" 2>/dev/null

run "4/5 · 两种子比对(出汇报口径)" \
  python src/stage10_v2/seed_compare.py --dirs "$OUT_E" "${OUT_E}_s1" --ref outputs/v22_e5_bio

# ---- 5 · 下游:四臂走廊是这次改动的**主要受影响对象**,必须重跑 ----
run "5a/5 · E18b 四臂走廊(v2.3 足端解绑)" \
  python src/stage10_v2/e18b_corridor_multi.py --v21 --foot bearing \
    --mlo 2 --mhi 40 --nu 9 --nm 16 --nprobe 48 --workers "$W" --out outputs/v23_e18b

run "5b/5 · E20 生成走廊(v2.3)" \
  python src/stage10_v2/e20_gen_corridor.py --v21 --foot bearing --outroot "$ROOT" \
    --mgrid 2,40,16 --anchors "5:产品下端,12:样机档,30:产品上端" \
    --nz 216 --workers "$W" --out outputs/v23_e20

run "5c/5 · E21 真鸟 vs 生成(v2.3)" \
  python src/stage10_v2/e21_bird_vs_gen.py --v21 --foot bearing --ckpt "$OUT_E/cvae_r39.pt" \
    --workers "$W" --out outputs/v23_e21

echo; echo "=========== 全部结束 ==========="
echo "产出: outputs/v2_p8 · v23_data_bio · v23_e5_bio(+_s1) · v23_e18b · v23_e20 · v23_e21"
echo
echo "⚠ 口径提醒(三条,别记混):"
echo "  ① 下游三个实验已全部带 --foot bearing,与工厂口径一致。"
echo "     与 v22_e18b_rs / v22_e20_rs(leg 模式)配对,就是足端解绑的前后对照。"
echo "  ② v2.3 与 v2.2 的物理不同,gap/可行率不可跨版本直接比;可比的是姿态、通道比、b_eff。"
echo "  ③ P8 的判据①(四臂废题率极差)若没有明显下降,说明解绑没生效,后面的都不用看。"
echo "日志: $LOG"

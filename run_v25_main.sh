#!/usr/bin/env bash
# v2.5 全量：足端摩擦真正起作用 + 水平触地速度进条件 + q1_0 进设计向量
#   cd /mnt/zihanw/FFY && nohup bash run_v25_main.sh > /dev/null 2>&1 &
#
# 三处物理改动，来源都是 A/B 组的实测结论（见《实验计划 v2.5》）：
#   ① 机体从「竖直滑轨」放开成「x-z 面内自由平动」。
#      v2.3 里水平力被 PrismaticJoint 的约束反力免费吃掉，倾斜姿态不花代价。
#      E26 因此算出 28° 的峰值只有 50° 的 42%，但那个姿态需要 μ≈0.70，真实地面上会滑。
#   ② 条件向量 5→6 维，第 6 维是 Froude 数 Fr = v_x/√(g·L_ref(m))，主工厂 Fr ∈ [0,2]。
#      不用 v_x 本身：同样 3 m/s 对 1 kg 和 30 kg 不是一回事。40% 的样本精确取 Fr=0，
#      保住与 v2.3 逐条对比的基线密度。
#   ③ 设计向量 9→10 维，第 10 维是 q1_0（跗跖骨倾角），盒取 E28 实测全距 [27.6, 54.8]°。
#      E26：只动它，峰值过载中位变化 69.7% —— 一阶因素，冻结在任何常数都不对。
#
# 判据也跟着改了三条（physics_v2.feasible_v2）：过载改判合加速度 a_res、
# 回弹软闸（E24 实测打掉 30.2%）、足端打滑硬闸（③ 的前提，没它最优会跑到会滑的姿态）。
#
# 分六段，规矩照抄 v2.3：不用 set -e，前段挂了后段照跑；每段打「[累计 N 分钟]」。
# 参考时长：v2.3 全跑 822 分钟（13.7 h），其中自提升闭环占 78%。v2.5 估 ≈18 h。
set -uo pipefail
cd "$(dirname "$0")"
W="${WORKERS:-128}"; ROUNDS="${ROUNDS:-40}"; SEEDS="${SEEDS:-0 1}"
OUT_F="${OUT_F:-outputs/v25_data_bio}"; OUT_E="${OUT_E:-outputs/v25_e5_bio}"
ROOT_D="${ROOT_D:-outputs/v25_root}"; P9="${P9:-outputs/v25_p9}"
REPORTS="reports/v25"
mkdir -p logs "$ROOT_D/v2_e5_bio" "$P9" "$REPORTS"
LOG="logs/v25_main_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }
# 注意 RC：run 里最后两条是 echo 和 el，所以调用点直接读 $? 拿到的是 echo 的 0。
# 把真实退出码存进 RC，卡关的两处判 RC。
RC=0
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; RC=$?; echo "退出码 $RC"; el; }

echo "workers=$W  rounds=$ROUNDS  seeds=$SEEDS"
echo "日志 $LOG"

# ---- 0 · 回归测试：先证明重构没动物理，不过就别往下跑 ----
# 拿 v2.3 工厂里存档的设计与工况原样重跑（planar=False, v_x=0），逐条比 peak_a。
# 这是唯一能证明「新代码在旧条件下和旧代码等价」的办法。
run "0/6 · 回归测试（对 v2.3 存档逐条比 peak_a，要求最大偏差 < 0.5%）" \
  python src/stage10_v2/p9_friction.py --mode reg --n 300 --workers "$W" \
    --factory outputs/v23_data_bio/factory.jsonl --out "$P9"
if [ "$RC" != "0" ]; then
  echo; echo "!!!! 回归测试没过。重构动到了不该动的地方，**停在这里**，别跑后面 18 小时。"
  echo "     诊断：logs/ 里这份日志 + $P9/p9_reg.json"
  exit 1
fi

# ---- 1 · 准入体检：新物理数值上站不站得住 ----
run "1/6 · P9a 准入体检（planar, Fr 0–2, q1_0 三档；五条判据）" \
  python src/stage10_v2/p9_friction.py --mode a --nprobe 64 --workers "$W" --out "$P9"
if [ "$RC" != "0" ]; then
  echo; echo "!!!! P9a 有判据不过。**停在这里**，先看 $P9/p9a_report.json。"
  echo "     尤其第 5 条：mu_demand 实测与 E26 的 tan(前倾角) 估计对不上的话，"
  echo "     q1_0 的盒子下界要按实测重定，不能直接用 27.6°。"
  exit 1
fi

# ---- 2 · 崩溃阶梯：只摸边界，不挡工厂，可以后台并行 ----
run "2/6 · P9b 崩溃阶梯（Fr 2→15，只为定位现有接触模型在哪一档崩）" \
  python src/stage10_v2/p9_friction.py --mode b --nprobe 64 --workers "$W" --out "$P9"

# ---- 3 · 数据工厂 ----
run "3/6 · 数据工厂 v2.5（10 维设计 + 6 维条件 + 面内平动）" \
  python src/stage10_v2/factory_v2.py --v25 --foot bearing --m-range 4,36 --arm bio \
    --nglobal 375 --npath 25 --K 5 --nd 120 --npass 2 --workers "$W" --out "$OUT_F"

run "4/6 · 训练集（条件 6 维，判据含回弹软闸与打滑闸）" \
  python src/stage10_v2/dataset_v2.py --factory "$OUT_F/factory.jsonl" --out "$OUT_F"

# ---- 5 · 自提升闭环 ×2 种子（占全程 78% 的机时） ----
for SD in $SEEDS; do
  if [ "$SD" = "0" ]; then O="$OUT_E"; else O="${OUT_E}_s${SD}"; fi
  run "5/6 · 自提升闭环 ×$ROUNDS (seed $SD → $O)" \
    python src/stage10_v2/e5_loop_v2.py --factory "$OUT_F/factory.jsonl" --out "$O" \
      --rounds "$ROUNDS" --workers "$W" --seed "$SD"
done
cp -f "$OUT_E"/cvae_r$((ROUNDS-1)).pt "$OUT_E"/cvae_r${ROUNDS}.pt \
      "$OUT_E"/model_meta.json "$ROOT_D/v2_e5_bio/" 2>/dev/null

run "5b/6 · 两种子比对（出汇报口径；训练类指标 ≥2 种子才报）" \
  python src/stage10_v2/seed_compare.py --dirs "$OUT_E" "${OUT_E}_s1" --ref outputs/v23_e5_bio

# ---- 6 · 下游：口径变了，必须全套重跑 ----
run "6a/6 · E18b 四臂走廊" \
  python src/stage10_v2/e18b_corridor_multi.py --v21 --foot bearing \
    --mlo 2 --mhi 40 --nu 9 --nm 16 --nprobe 48 --workers "$W" --out outputs/v25_e18b

run "6b/6 · E20 生成走廊" \
  python src/stage10_v2/e20_gen_corridor.py --v21 --foot bearing --outroot "$ROOT_D" \
    --mgrid 2,40,16 --anchors "5:产品下端,12:样机档,30:产品上端" \
    --nz 216 --workers "$W" --out outputs/v25_e20

run "6c/6 · E21 真鸟 vs 生成" \
  python src/stage10_v2/e21_bird_vs_gen.py --v21 --foot bearing \
    --ckpt "$OUT_E/cvae_r$((ROUNDS-1)).pt" --workers "$W" --out outputs/v25_e21

# ---- 附 · E22 现在是零成本了（D 已落盘） ----
run "附 · E22 D/L 标度（v2.5 工厂已落盘 D_mm，不用再重评）" \
  python src/stage10_v2/e22_dl_scaling.py --analyze-only --fig \
    --factory "$OUT_F/factory.jsonl" --out "$OUT_F"

# ---- 产物收口：reports/ 不在 .gitignore 里，push 一次就能带回本地 ----
for f in "$P9"/*.json outputs/v25_data_bio/e22_*.json outputs/v25_data_bio/e22_*.png \
         "$OUT_E"/trajectory.json "${OUT_E}_s1"/trajectory.json; do
  [ -f "$f" ] && cp -f "$f" "$REPORTS/"
done
cp -f "$LOG" "$REPORTS/run.log" 2>/dev/null

echo; echo "=========== 全部结束（$(( ($(date +%s)-t0)/60 )) 分钟）==========="
echo
echo "产物：$OUT_F · $OUT_E(+_s1) · outputs/v25_e18b · v25_e20 · v25_e21 · $P9"
echo "小文件已收进 $REPORTS —— 带回本地："
echo "  git add -A reports && git commit -m 'v2.5 结果' && git push"
echo
echo "⚠ 口径提醒（三条，别记混）："
echo "  ① v2.5 的机体能水平走了，**即使 Fr=0 也与 v2.3 不同物理** ——"
echo "     可行率/gap 不能跨版本直接比。要比就比 Fr=0 那 40% 与 v2.3 的差，"
echo "     那个差本身就是「v2.3 的竖直滑轨到底免费送了多少」的度量。"
echo "  ② 设计向量 10 维、条件 6 维，旧的 ckpt 和数据集都不能混用。"
echo "  ③ 判据多了三条（a_res / 回弹 / 打滑），可行率必然比 v2.3 低，这是预期不是回退。"

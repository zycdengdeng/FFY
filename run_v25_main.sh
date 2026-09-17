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
#
# ---- 两种构型，用 CONFIG 选 ----
#   CONFIG=bird  仿鸟：两条腿姿态相同、沿 y 并排 → 水平力**同向叠加，不抵消**
#                → 机体必须在 x-z 面内自由平动，姿态引起的水平力只能由足端摩擦承担。
#                真鸟就是这一类（两腿同姿态、都前倾约 30°，靠脚趾抓地）。
#   CONFIG=skid  对置：前后两对腿姿态镜像 → 水平力在机体内部抵消
#                → 保留竖直约束（这时 v2.3 的滑轨不是作弊，是对称性的正确等效）。
#                足端切向摩擦仍作用在腿上，只是机体不被推着走。
#                ⚠ 已知理想化：机体水平速度在冲击中不衰减，切向载荷偏保守（偏大）。
set -uo pipefail
cd "$(dirname "$0")"
CONFIG="${CONFIG:-bird}"
case "$CONFIG" in
  bird) PLANAR=1; FRCAP=2.0 ;;
  # 对置构型只能跑纯垂直：机体被竖直滑轨锁着，给它水平初速 t=0 就违反约束，
  # 求解器用一个冲量抹掉，a_res 变成上万 g 的数值垃圾（首跑实测 22816 g）。
  # 要让对置构型带水平速度，得建两条镜像腿、机体照样自由平动 —— 那是另一次改动。
  skid) PLANAR=0; FRCAP=0.0 ;;
  *) echo "CONFIG 只能是 bird 或 skid"; exit 2 ;;
esac
# 物理代码的版本戳。工厂/闭环都有续跑缓存，但缓存只在**同一版物理**下才合法：
# 首跑的 v25 产物是用错的 μ（mu_from_ground 未生效）和错的 slip（没扣足球滚动）算的，
# 直接续跑会把污染数据捡回来。目录里 .codever 不等于这个值就整体挪到 *_stale_<时间>。
# v25.3：滑移的滚动扣除符号修正（Δx − r·Δφ，此前为 +），列清单加 slip_alt。
# 旧 v25.2 工厂的 slip 列在"粘住"场景整体虚高约 2·r·Δφ，闸误杀，必须整体归档重跑。
# ---- 材料：MATERIAL=al7075 bash run_v25_main.sh ----
#   cfnylon = 打印碳纤尼龙（默认，样机可打印口径）
#   al7075  = 航空铝（与真实起落架同材料，对标口径；腿更细更轻，定尺判据不变）
#   注意：换材料改变杆件质量→回代进动力学，所以必须全量重跑，不能后处理。
#   μ(k_c) 估计式不随材料变（铝-混凝土与尼龙-混凝土同量级），文档里已标"本工作设定"。
MATERIAL="${MATERIAL:-cfnylon}"
export FFY_MATERIAL="$MATERIAL"
MSUF=""
if [ "$MATERIAL" != "cfnylon" ]; then MSUF="_${MATERIAL}"; fi
if [ "$MATERIAL" = "cfnylon" ]; then CODEVER="v25.3-slipsign"; else CODEVER="v25.4-${MATERIAL}"; fi   # cfnylon 保持旧戳，避免误归档已完成的 18h 产物
W="${WORKERS:-128}"; ROUNDS="${ROUNDS:-40}"; SEEDS="${SEEDS:-0 1}"
OUT_F="${OUT_F:-outputs/v25_${CONFIG}${MSUF}_data}"; OUT_E="${OUT_E:-outputs/v25_${CONFIG}${MSUF}_e5}"
ROOT_D="${ROOT_D:-outputs/v25_${CONFIG}${MSUF}_root}"; P9="${P9:-outputs/v25_${CONFIG}${MSUF}_p9}"
REPORTS="reports/v25_${CONFIG}${MSUF}"
mkdir -p logs "$ROOT_D/v2_e5_bio" "$P9" "$REPORTS"
LOG="logs/v25_${CONFIG}${MSUF}_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
t0=$(date +%s); el(){ echo "[累计 $(( ($(date +%s)-t0)/60 )) 分钟]"; }
fresh(){  # fresh DIR：版本不符就归档；随后建目录并盖章（同版本内的续跑不受影响）
  local d="$1"
  if [ -d "$d" ] && [ "$(cat "$d/.codever" 2>/dev/null)" != "$CODEVER" ]; then
    local t="${d}_stale_$(date +%m%d_%H%M)"
    echo "[fresh] $d 是旧版物理的产物（或无版本戳），挪到 $t"
    mv "$d" "$t"
  fi
  mkdir -p "$d"; echo "$CODEVER" > "$d/.codever"
}
# 注意 RC：run 里最后两条是 echo 和 el，所以调用点直接读 $? 拿到的是 echo 的 0。
# 把真实退出码存进 RC，卡关的两处判 RC。
RC=0
run(){ echo; echo "=========== $1 ==========="; shift; "$@"; RC=$?; echo "退出码 $RC"; el; }

echo "材料 MATERIAL=$MATERIAL  构型 CONFIG=$CONFIG (planar=$PLANAR)  workers=$W  rounds=$ROUNDS  seeds=$SEEDS  codever=$CODEVER"
for d in "$P9" "$OUT_F" "$OUT_E" "${OUT_E}_s1" "$ROOT_D" \
         "outputs/v25_${CONFIG}${MSUF}_e18b" "outputs/v25_${CONFIG}${MSUF}_e20" "outputs/v25_${CONFIG}${MSUF}_e21"; do
  fresh "$d"
done
echo "日志 $LOG"

# ---- 0 · 回归测试：先证明重构没动物理，不过就别往下跑 ----
# 拿 v2.3 工厂里存档的设计与工况原样重跑（planar=False, v_x=0），逐条比 peak_a。
# 这是唯一能证明「新代码在旧条件下和旧代码等价」的办法。
if [ "$MATERIAL" != "cfnylon" ]; then
  echo; echo "=========== 0/6 · 回归测试跳过：材料=$MATERIAL，v2.3 基线是 cfnylon，逐条比对无意义 ==========="
  echo "（代码等价性已由 cfnylon 版的回归测试保证；本次只换材料常数。）"
  RC=0
else
run "0/6 · 回归测试（对 v2.3 存档逐条比 peak_a，要求最大偏差 < 0.5%）" \
  python src/stage10_v2/p9_friction.py --mode reg --n 300 --workers "$W" \
    --factory outputs/v23_data_bio/factory.jsonl --out "$P9"
if [ "$RC" != "0" ]; then
  echo; echo "!!!! 回归测试没过。重构动到了不该动的地方，**停在这里**，别跑后面 18 小时。"
  echo "     诊断：logs/ 里这份日志 + $P9/p9_reg.json"
  exit 1
fi
fi

# ---- 1 · 准入体检：新物理数值上站不站得住 ----
run "1/6 · P9a 准入体检（三条数值判据 + 用可行率给 Fr 上界定标）" \
  python src/stage10_v2/p9_friction.py --mode a --nprobe 64 --workers "$W" \
    --planar "$PLANAR" --out "$P9"
if [ "$RC" != "0" ]; then
  echo; echo "!!!! P9a 数值判据不过。**停在这里**，先看 $P9/p9a_report.json。"
  exit 1
fi

# Fr 上界由 P9a 量出来，不是拍的；再夹到该构型的物理上限 FRCAP。
if [ "$FRCAP" != "0.0" ]; then
  FRMAX=$(python - <<PY 2>/dev/null
import json
r = json.load(open("$P9/p9a_report.json"))["fr_recommended"]
print(min($FRCAP, max(0.0, r)))
PY
)
  FRMAX="${FRMAX:-0.0}"
else
  FRMAX=0.0
fi
echo "[v25/$CONFIG] 主工厂的 Fr 上界取 $FRMAX（P9a 实测定标，上限 $FRCAP）"
if [ "$FRMAX" = "0.0" ] || [ "$FRMAX" = "0" ]; then
  echo "[v25/$CONFIG] Fr 恒为 0 → 条件仍是 6 维但第 6 维退化，等价于纯垂直着陆。"
fi

# μ 扫描：回答 E26 那张 tan(前倾角) 表到底准不准。不挡工厂，但结论要写进论文。
run "1b/6 · μ 扫描（求临界摩擦系数，Fr=0，硬地）" \
  python src/stage10_v2/p9_friction.py --mode mu --nprobe 64 --workers "$W" \
    --planar "$PLANAR" --out "$P9"

# ---- 2 · 崩溃阶梯：只摸边界，不挡工厂，可以后台并行 ----
# P9b 只对能带水平速度的构型有意义
if [ "$PLANAR" = "1" ]; then
  run "2/6 · P9b 崩溃阶梯（Fr 2→15，只为定位现有接触模型在哪一档崩）" \
    python src/stage10_v2/p9_friction.py --mode b --nprobe 64 --workers "$W" \
      --planar "$PLANAR" --out "$P9"
else
  echo; echo "=========== 2/6 · P9b 跳过（对置构型不能带水平速度，见上面的说明）==========="; el
fi

# ---- 3 · 数据工厂 ----
run "3/6 · 数据工厂 v2.5（10 维设计 + 6 维条件 + 面内平动）" \
  python src/stage10_v2/factory_v2.py --v25 --foot bearing --m-range 4,36 --arm bio \
    --planar "$PLANAR" --fr-max "$FRMAX" \
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
    --mlo 2 --mhi 40 --nu 9 --nm 16 --nprobe 48 --workers "$W" --out "outputs/v25_${CONFIG}${MSUF}_e18b"

run "6b/6 · E20 生成走廊" \
  python src/stage10_v2/e20_gen_corridor.py --v21 --foot bearing --ckpt "$OUT_E/cvae_r$((ROUNDS-1)).pt" \
    --mgrid 2,40,16 --anchors "5:产品下端,12:样机档,30:产品上端" \
    --nz 216 --workers "$W" --out "outputs/v25_${CONFIG}${MSUF}_e20"

run "6c/6 · E21 真鸟 vs 生成" \
  python src/stage10_v2/e21_bird_vs_gen.py --v21 --foot bearing \
    --ckpt "$OUT_E/cvae_r$((ROUNDS-1)).pt" --workers "$W" --out "outputs/v25_${CONFIG}${MSUF}_e21"

# ---- 附 · E22 现在是零成本了（D 已落盘） ----
run "附 · E22 D/L 标度（v2.5 工厂已落盘 D_mm，不用再重评）" \
  python src/stage10_v2/e22_dl_scaling.py --from-factory --fig \
    --factory "$OUT_F/factory.jsonl" --out "$OUT_F"

# ---- 产物收口：reports/ 不在 .gitignore 里，push 一次就能带回本地 ----
for f in "$P9"/*.json "$OUT_F"/e22_*.json "$OUT_F"/e22_*.png \
         "$OUT_F"/dataset_meta.json "$OUT_F"/factory_meta.json \
         "$OUT_E"/trajectory.json "$OUT_E"/model_meta.json; do
  [ -f "$f" ] && cp -f "$f" "$REPORTS/"
done
[ -f "${OUT_E}_s1/trajectory.json" ] && cp -f "${OUT_E}_s1/trajectory.json" "$REPORTS/trajectory_s1.json"
cp -f "$LOG" "$REPORTS/run.log" 2>/dev/null

echo; echo "=========== 全部结束（$(( ($(date +%s)-t0)/60 )) 分钟）==========="
echo
echo "构型 $CONFIG 的产物：$OUT_F · $OUT_E(+_s1) · v25_${CONFIG}${MSUF}_e18b/e20/e21 · $P9"
echo "小文件已收进 $REPORTS —— 直接下载这个目录到本地 FFY/ 下即可。"
echo "（reports/ 已在 .gitignore 里，git 不参与搬运，下载后也**不要** sync 它。）"
echo
echo "⚠ 口径提醒（三条，别记混）："
echo "  ① v2.5 的机体能水平走了，**即使 Fr=0 也与 v2.3 不同物理** ——"
echo "     可行率/gap 不能跨版本直接比。要比就比 Fr=0 那 40% 与 v2.3 的差，"
echo "     那个差本身就是「v2.3 的竖直滑轨到底免费送了多少」的度量。"
echo "  ② 设计向量 10 维、条件 6 维，旧的 ckpt 和数据集都不能混用。"
echo "  ③ 判据多了三条（a_res / 回弹 / 打滑），可行率必然比 v2.3 低，这是预期不是回退。"

#!/usr/bin/env bash
# E11 · 仿生先验消融:一键顺序执行(bio / wide / shift 三臂,共用并集标尺)
#
# 用法(A100,建议在 tmux 里裸跑):
#   bash run_e11_ablation.sh
#   WORKERS=64 ROUNDS=10 bash run_e11_ablation.sh      # 覆盖默认参数
#
# 特性:
#   - 任一步失败立即停(set -e),不会带着坏数据往下跑;
#   - 全程可断点续跑:重跑本脚本自动跳过已完成的步骤
#     (工厂与循环本身即可续跑,标尺/重考按产物存在与否跳过);
#   - 日志同时写屏与存档 logs/e11_ablation_<时间戳>.log。
set -euo pipefail

WORKERS="${WORKERS:-128}"     # 并行进程数
ROUNDS="${ROUNDS:-20}"        # wide/shift 臂自提升轮数
NREF="${NREF:-300}"           # 每盒每题参考采样数
NC="${NC:-500}"               # 工厂工况数
ND="${ND:-120}"               # 工厂每工况设计数
KGEN="${KGEN:-24}"

cd "$(dirname "$0")"
mkdir -p logs outputs/gen_abl
LOG="logs/e11_ablation_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1

export OMP_NUM_THREADS=1
UNION="outputs/gen_abl/refs_union.json"

step() { echo -e "\n=========== [$(date +%H:%M:%S)] $* ==========="; }
t0=$(date +%s)
el() { echo "[累计 $(( ($(date +%s) - t0) / 60 )) 分钟]"; }

echo "E11 消融 | workers=$WORKERS rounds=$ROUNDS nref=$NREF 工厂=${NC}×${ND}"
echo "日志: $LOG"

# ---------------------------------------------------------------- ① 公共标尺
step "① 并集参考前沿(三盒共用标尺)"
if [[ -f "$UNION" ]]; then
  echo "已存在,跳过:$UNION"
else
  python src/stage7_generative/build_union_refs.py \
    --src-refs outputs/gen_e5/refs.json --out "$UNION" \
    --nref "$NREF" --workers "$WORKERS"
fi
el

# ---------------------------------------------------------------- ② bio 臂
step "② bio 臂:复用 gen_e5 存档 r0-${ROUNDS},在并集标尺上重考"
if [[ -f outputs/gen_abl/bio_trajectory.json ]]; then
  echo "已存在,跳过:outputs/gen_abl/bio_trajectory.json"
else
  python src/stage7_generative/eval_rounds.py \
    --model-dir outputs/gen_e5 --refs "$UNION" \
    --rounds "0-${ROUNDS}" --out outputs/gen_abl/bio_trajectory.json \
    --workers "$WORKERS"
fi
el

# ---------------------------------------------------------------- ③④ 另两臂
run_arm() {                    # $1 = 盒子名(wide / shift)
  local box="$1"
  local fdir="outputs/gen_data7_${box}"
  local odir="outputs/gen_abl_${box}"

  step "${box} 臂 · 数据工厂(几何盒子=${box},刚度阻尼范围与 bio 一致)"
  python src/stage7_generative/data_factory.py \
    --nc "$NC" --nd "$ND" --dim 7 --box "$box" \
    --workers "$WORKERS" --out "$fdir"
  el

  step "${box} 臂 · 自提升循环 ${ROUNDS} 轮"
  mkdir -p "$odir"
  [[ -f "$odir/refs.json" ]] || cp "$UNION" "$odir/refs.json"   # 预置标尺 → e5_loop 直接加载,不自算
  python src/stage7_generative/e5_loop.py \
    --factory "$fdir/factory.jsonl" --data outputs/gen_data7/gen_dataset.npz \
    --out "$odir" --rounds "$ROUNDS" --kgen "$KGEN" --workers "$WORKERS"
  el
}

run_arm wide
run_arm shift

# ---------------------------------------------------------------- 汇总
step "汇总"
python - <<'PY'
import json, os
import numpy as np

def tail(fp, k=5):
    if not os.path.exists(fp):
        return None
    t = json.load(open(fp))
    g = [x["median_gap"] * 100 for x in t]
    f = [x["feas_rate"] * 100 for x in t]
    return dict(n=len(t), first=g[0], best=min(g),
                last5=float(np.median(g[-k:])), feas_last=f[-1])

arms = [("bio", "outputs/gen_abl/bio_trajectory.json"),
        ("wide", "outputs/gen_abl_wide/trajectory.json"),
        ("shift", "outputs/gen_abl_shift/trajectory.json")]
print(f"{'臂':<8}{'轮数':>5}{'r0 gap':>10}{'最好':>9}{'末5轮中位':>12}{'末轮可行率':>12}")
for name, fp in arms:
    s = tail(fp)
    if s is None:
        print(f"{name:<8}  (缺 {fp})"); continue
    print(f"{name:<8}{s['n']:>5}{s['first']:>9.1f}%{s['best']:>8.1f}%"
          f"{s['last5']:>11.1f}%{s['feas_last']:>11.0f}%")

d = "outputs/gen_abl/union_diag.json"
if os.path.exists(d):
    D = json.load(open(d))
    print("\n并集标尺诊断(各盒是否含更好设计):")
    for b in ("bio", "wide", "shift"):
        wins = sum(1 for x in D if x.get(f"ref_{b}") is not None
                   and abs(x[f"ref_{b}"] - x["ref_union"]) < 1e-9)
        fe = np.mean([x[f"feas_{b}"] for x in D]) * 100
        print(f"  {b:6s}: 提供并集最优 {wins}/{len(D)} 题, 平均可行率 {fe:.1f}%")
PY

step "全部完成"
el
echo "结果目录: outputs/gen_abl{,_wide,_shift}"

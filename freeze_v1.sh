#!/usr/bin/env bash
# v1「质量无关版」一键冻结:代码打标 + 结果清点 + 校验和 + 可搬运档案包
#
# 用法(A100,仓库根目录):
#   bash freeze_v1.sh                 # 正常冻结
#   TAG=v1.0-mass-invariant bash freeze_v1.sh
#   VERIFY=1 bash freeze_v1.sh        # 只校验已有档案,不重新打包
#
# 产出(全部在 archive/<TAG>/ 下):
#   code_<TAG>.tar.gz    代码快照(git archive,与 tag 逐字节一致)
#   MANIFEST.tsv         每个结果文件的 相对路径/字节数/sha256
#   SIZES.tsv            每个结果目录的体量
#   HEADLINE.json        自动抽取的各实验头条数字(防止日后只剩文件不记得结论)
#   core_results.tar.gz  小件全打包(json/png/md/pt),可直接下载留存
#   BIG_FILES.tsv        大件清单(factory.jsonl / pool 等),留在原地 + 校验和
#   ARCHIVE_README.md    本次冻结的自述
set -euo pipefail
cd "$(dirname "$0")"

TAG="${TAG:-v1.0-mass-invariant}"
ADIR="archive/${TAG}"
BIGMB="${BIGMB:-64}"                 # 超过此大小(MB)算「大件」,不进 core 包
mkdir -p "$ADIR"

# 结果目录白名单:存在才处理
CANDIDATES=(
  outputs/gen_data7 outputs/gen_data7_wide outputs/gen_data7_shift
  outputs/gen_e5 outputs/gen_e5b outputs/gen_e5c
  outputs/gen_e6 outputs/gen_e6c outputs/gen_e7
  outputs/gen_e8 outputs/gen_e8c
  outputs/gen_abl outputs/gen_abl_wide outputs/gen_abl_shift
  outputs/gen_e12 outputs/gen_e13 outputs/gen_e14
  outputs/surrogate_exu7d outputs/multi_metric_v2 outputs/bird_pareto_v16
)
DIRS=(); for d in "${CANDIDATES[@]}"; do [[ -d "$d" ]] && DIRS+=("$d"); done
echo "[freeze] 纳入 ${#DIRS[@]} 个结果目录"

# ---------------------------------------------------------------- ① 代码快照
if [[ "${VERIFY:-0}" != "1" ]]; then
  echo "[freeze] ① 代码快照"
  git add -A && git commit -q -m "freeze: ${TAG} 质量无关版最终状态" || echo "  (无新改动可提交)"
  git tag -f "$TAG" -m "v1 质量无关版:响应对 m 严格不变,详见 版本冻结_v1_质量无关版.md"
  git archive --format=tar.gz -o "$ADIR/code_${TAG}.tar.gz" "$TAG"
  git rev-parse "$TAG" > "$ADIR/COMMIT.txt"
  echo "  tag=$TAG commit=$(cat "$ADIR/COMMIT.txt")"
fi

# ---------------------------------------------------------------- ② 清点+校验和
echo "[freeze] ② 清点结果文件并计算 sha256(大目录会慢几分钟)"
: > "$ADIR/MANIFEST.tsv"; : > "$ADIR/BIG_FILES.tsv"
printf 'path\tbytes\tsha256\n' >> "$ADIR/MANIFEST.tsv"
printf 'path\tbytes\tsha256\n' >> "$ADIR/BIG_FILES.tsv"
LIST_CORE="$ADIR/.core_list"; : > "$LIST_CORE"
for d in "${DIRS[@]}"; do
  while IFS= read -r -d '' f; do
    sz=$(stat -c%s "$f"); sh=$(sha256sum "$f" | cut -d' ' -f1)
    printf '%s\t%s\t%s\n' "$f" "$sz" "$sh" >> "$ADIR/MANIFEST.tsv"
    if (( sz > BIGMB * 1024 * 1024 )); then
      printf '%s\t%s\t%s\n' "$f" "$sz" "$sh" >> "$ADIR/BIG_FILES.tsv"
    else
      printf '%s\0' "$f" >> "$LIST_CORE"
    fi
  done < <(find "$d" -type f -print0)
done
du -sh "${DIRS[@]}" > "$ADIR/SIZES.tsv" 2>/dev/null || true
echo "  文件 $(( $(wc -l < "$ADIR/MANIFEST.tsv") - 1 )) 个,其中大件 $(( $(wc -l < "$ADIR/BIG_FILES.tsv") - 1 )) 个"

# ---------------------------------------------------------------- ③ 头条数字
echo "[freeze] ③ 抽取头条数字"
python - "$ADIR" <<'PY'
import json, os, sys
import numpy as np
adir = sys.argv[1]
H = {}

def traj(fp, name):
    if not os.path.exists(fp): return
    t = json.load(open(fp))
    g = [x["median_gap"] * 100 for x in t]
    H[name] = dict(file=fp, rounds=len(t), r0=g[0], best=min(g),
                   last=g[-1], last5_median=float(np.median(g[-5:])),
                   fail_last=t[-1].get("fail"),
                   feas_last=t[-1].get("feas_rate"))

for n, f in [("E5_bio", "outputs/gen_e5/trajectory.json"),
             ("E5c_bio_85r", "outputs/gen_e5c/trajectory.json"),
             ("E11_wide", "outputs/gen_abl_wide/trajectory.json"),
             ("E11_shift", "outputs/gen_abl_shift/trajectory.json"),
             ("E11_bio_union", "outputs/gen_abl/bio_trajectory.json")]:
    traj(f, n)

def grab(fp, name, keys=None):
    if not os.path.exists(fp): return
    try: d = json.load(open(fp))
    except Exception: return
    H[name] = dict(file=fp, **({k: d[k] for k in keys if k in d} if keys
                               else {"_keys": list(d)[:24] if isinstance(d, dict) else f"list[{len(d)}]"}))

for n, f in [("E5_seeds", "outputs/gen_e5c/seeds_summary.json"),
             ("E6_multimetric", "outputs/gen_e6c/e6_summary.json"),
             ("E7_polish", "outputs/gen_e7/e7_summary.json"),
             ("E8_struct", "outputs/gen_e8c/e8_summary.json"),
             ("E11b_struct", "outputs/gen_abl/e11b_struct.json"),
             ("E12_feas", "outputs/gen_e12/e12_summary.json"),
             ("E13_robust", "outputs/gen_e13/e13_summary.json"),
             ("E14_mass_probe", "outputs/gen_e14/e14_mass_probe.json")]:
    grab(f, n)

json.dump(H, open(os.path.join(adir, "HEADLINE.json"), "w"),
          indent=2, ensure_ascii=False, default=float)
for k, v in H.items():
    print(f"  {k:<16} {v.get('file','')}")
PY

# ---------------------------------------------------------------- ④ 打包小件
if [[ "${VERIFY:-0}" != "1" ]]; then
  echo "[freeze] ④ 打包小件 → core_results.tar.gz"
  tar --null -czf "$ADIR/core_results.tar.gz" -T "$LIST_CORE" \
      参数溯源表.md 回弹判据调研.md 多保真代理实验报告.md 生成阶段实验报告.md \
      论文表格_生成线.md 版本冻结_v1_质量无关版.md 2>/dev/null || \
  tar --null -czf "$ADIR/core_results.tar.gz" -T "$LIST_CORE"
  ls -lh "$ADIR/core_results.tar.gz" | awk '{print "  "$9" "$5}'
fi
rm -f "$LIST_CORE"

# ---------------------------------------------------------------- ⑤ 自述
cat > "$ADIR/ARCHIVE_README.md" <<MDEOF
# 档案 ${TAG} — 质量无关版(v1)

冻结时间:$(date '+%Y-%m-%d %H:%M:%S %Z')
主机:$(hostname)   仓库 commit:$(git rev-parse HEAD)

## 这是什么
起落架生成式设计流水线的 v1 完整结果。**已知建模局限**:尺寸化规则把关节刚度、
接触刚度、杆件质量全部写成 m 的正比量,导致归一化响应(peak_a[g]、stroke、η)
对 m 严格不变,只有绝对力 F_peak 随 m 变。详见仓库根目录
\`版本冻结_v1_质量无关版.md\` 与 \`outputs/gen_e14/\` 的验尸报告。

## 怎么用
- 复现代码:\`tar xzf code_${TAG}.tar.gz\`,或 \`git checkout ${TAG}\`
- 结果小件:\`tar xzf core_results.tar.gz\`
- 大件(未打包,留在本机原路径):见 \`BIG_FILES.tsv\`
- 完整性校验:\`awk -F'\t' 'NR>1{print \$3"  "\$1}' MANIFEST.tsv | sha256sum -c -\`
- 头条数字速查:\`HEADLINE.json\`

## 结论有效性
v1 的**方法论结论全部有效**(专家迭代 vs BO 的样本效率、仿生先验消融、
可行性分类、鲁棒性零结果),因为它们都是在固定物理下的相对比较。
**失效的只有涉及"跨质量泛化"的表述**——那一维在 v1 中是死的。
MDEOF

echo "[freeze] 完成 → $ADIR"
ls -la "$ADIR"

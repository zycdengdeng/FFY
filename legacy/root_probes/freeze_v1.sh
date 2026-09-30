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

if [[ "${ONLY_INDEX:-0}" == "1" ]]; then    # 只重算索引,不动代码包与结果包
  python tools/index_results.py --roots outputs --out "$ADIR/HEADLINE.json"
  echo "[freeze] 仅重算索引完成 → $ADIR/HEADLINE.json"; exit 0
fi

# 结果目录:自动发现 outputs/ 下所有一级目录(硬编码清单会静默漏掉改过名的实验)
DIRS=()
while IFS= read -r d; do DIRS+=("$d"); done < <(find outputs -mindepth 1 -maxdepth 1 -type d | sort)
[[ -d logs ]] && DIRS+=("logs")      # 运行日志也是实验记录,一并存档
echo "[freeze] 纳入 ${#DIRS[@]} 个结果目录"

# ---------------------------------------------------------------- ① 代码快照
# 注意:**不在这里 git commit**。本脚本通常在 A100 上跑,而 A100 是只读消费端
# (代码走 Windows → GitHub → A100)。在这里提交会让本地与 origin 分叉,
# 下次 git pull 直接 fatal。快照因此直接打包工作树,不依赖提交状态。
if [[ "${VERIFY:-0}" != "1" ]]; then
  echo "[freeze] ① 代码快照(打包工作树,不提交)"
  git rev-parse HEAD > "$ADIR/COMMIT.txt" 2>/dev/null || echo "not-a-git-repo" > "$ADIR/COMMIT.txt"
  git status --porcelain > "$ADIR/DIRTY.txt" 2>/dev/null || true
  git diff HEAD > "$ADIR/uncommitted.diff" 2>/dev/null || true
  tar -czf "$ADIR/code_${TAG}.tar.gz" \
      --exclude=.git --exclude=outputs --exclude=archive --exclude=logs \
      --exclude='*.pt' --exclude='__pycache__' --exclude='*.pyc' .
  # 标签只是书签,便于 git checkout;真正权威的是上面的 tar 包
  git tag -f "$TAG" -m "v1 质量无关版:响应对 m 严格不变,见 版本冻结_v1_质量无关版.md" \
      >/dev/null 2>&1 || true
  n_dirty=$(wc -l < "$ADIR/DIRTY.txt" 2>/dev/null || echo 0)
  echo "  commit=$(cat "$ADIR/COMMIT.txt")  未提交改动 ${n_dirty} 项(已存 uncommitted.diff)"
  ls -lh "$ADIR/code_${TAG}.tar.gz" | awk '{print "  "$9" "$5}'
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
echo "[freeze] ③ 抽取头条数字(自动发现摘要文件)"
python tools/index_results.py --roots outputs --out "$ADIR/HEADLINE.json"

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
- 复现代码:\`tar xzf code_${TAG}.tar.gz\`(权威:含未提交改动)
  标签 \`${TAG}\` 只是书签;若与 tar 包有出入,以 tar 包 + \`uncommitted.diff\` 为准
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

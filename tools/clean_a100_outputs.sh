#!/usr/bin/env bash
# A100 outputs 清理(2026-09-30):不直接 rm——先挪进 outputs/_trash_0930/,确认后再整目录删。
#   用法: cd /mnt/zihanw/FFY && bash tools/clean_a100_outputs.sh
#   确认无误后: rm -rf outputs/_trash_0930
# 必须保留(论文/在跑/复现依赖):
#   e18c_scan er4b er3_full er3 g1b_e18b pgls phylo bird_pareto(FIT_JSON!)
#   v25_bird_data v25_bird_e5 v25_bird_e5_s1 v25_bird_e18b/e20/e21 v25_bird_p9* v25_bird_root
#   v25_bird_al7075_*(材料对比) g10_bio407_e5* g1_{geo,elastic,none}_e5*(SI 锚点敏感性)
#   v23_data_bio(run_v25_main 回归测试引用) v3_*(第二篇轮足预研) duong 相关
set -u
cd "$(dirname "$0")/.."
T=outputs/_trash_0930; mkdir -p "$T"
echo "清理前占用:"; du -sh outputs 2>/dev/null

# ── A 档:确定废弃 ─────────────────────────────
# 1) 所有 *_stale_*(fresh() 归档的旧版本物理产物)
for d in outputs/*_stale_*; do [ -e "$d" ] && mv "$d" "$T/"; done
# 2) v2.0–v2.3 时代整代产物(保 v23_data_bio)
for d in outputs/v2_* outputs/v21_* outputs/v22_* outputs/v23_*; do
  case "$d" in outputs/v23_data_bio) continue;; esac
  [ -e "$d" ] && mv "$d" "$T/"
done
# 3) v1 生成线与早期研究
# bird_pareto 不删!bioprior.FIT_JSON 运行时读 outputs/bird_pareto/avonet_allometry.json(缺失会静默回退硬编码值)
for d in outputs/gen_* outputs/bird_pareto_v2 outputs/multi_metric outputs/multi_metric_v2 outputs/theta_sens; do
  [ -e "$d" ] && mv "$d" "$T/"
done
# 4) 动画/演示渲染与冒烟
for d in outputs/anim_b outputs/anim_b.log outputs/anim_hard.log outputs/swan01 outputs/er3_smoke; do
  [ -e "$d" ] && mv "$d" "$T/"
done

# ── B 档:建议删,你确认(去掉 echo 前缀即执行)─────
echo; echo "B 档(建议删,自行确认后手动 mv 或解除注释):"
echo "  g10_bio407_data g10_bio407_root g10_bio407_p9   # bio407 臂中间产物,e5 结果已留"
echo "  g1_{geo,elastic,none}_data/_root/_p9            # 旧锚点臂中间产物,e5 结果已留"
echo "  v25_skid_*                                      # 对置构型,第一篇不用;若保三点式后路则留"
echo "  v26_p9w                                         # 单次探针,查无引用后删"
# for d in outputs/g10_bio407_data outputs/g10_bio407_root outputs/g10_bio407_p9 \
#          outputs/g1_geo_data outputs/g1_geo_root outputs/g1_geo_p9 \
#          outputs/g1_elastic_data outputs/g1_elastic_root outputs/g1_elastic_p9 \
#          outputs/g1_none_data outputs/g1_none_root outputs/g1_none_p9 \
#          outputs/v25_skid_* outputs/v26_p9w; do [ -e "$d" ] && mv "$d" "$T/"; done

echo; echo "已挪入 $T:"; ls "$T" | head -60
echo; echo "清理后 outputs(不含 _trash)占用:"; du -sh --exclude=_trash_0930 outputs 2>/dev/null || du -sh outputs
echo "确认无误后执行: rm -rf $T"

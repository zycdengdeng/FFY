#!/usr/bin/env bash
# E18c 绝对腿长扫描(E-R4b 分解矩阵的数据源;物理=evalcfg=v2.5 工厂同款)
#   用法(A100,tmux 里直接跑):cd /mnt/zihanw/FFY && git pull && bash run_e18c.sh
#   之后本地:python src/stage10_v2/er4b_decompose.py --scan outputs/e18c_scan --out outputs/er4b
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs outputs/e18c_scan
LOG="logs/e18c_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
export OMP_NUM_THREADS=1
W="${WORKERS:-96}"; NP="${NPROBE:-96}"
python src/stage10_v2/evalcfg.py
python src/stage10_v2/e18c_abs_scan.py --workers "$W" --nprobe "$NP" --out outputs/e18c_scan
echo "E18c 完毕:outputs/e18c_scan/e18c_*.json + *_probes.npz + stamp.json(下载整个目录)"

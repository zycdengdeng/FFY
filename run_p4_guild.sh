#!/usr/bin/env bash
# E-R1d 分组 PGLS。用法(A100): cd /mnt/zihanw/FFY && bash run_p4_guild.sh
# 与四臂训练无冲突(纯 R 单进程,不抢 workers);可与 run_g1_arms 并行。
set -uo pipefail
cd "$(dirname "$0")"
python3 src/stage6_surrogate/pgls_p4_prep.py
mkdir -p logs
NT="${NT:-25}"
Rscript src/stage6_surrogate/pgls_p4_guild.R "$NT" 2>&1 | tee "logs/pgls_p4_$(date +%Y%m%d_%H%M%S).log"

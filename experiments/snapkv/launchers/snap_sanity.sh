#!/bin/bash
source /home/tiger/cudafix.sh
export CUDA_VISIBLE_DEVICES=0
export SPRAG_MODEL_PATH=/tmp/Qwen3-30B-A3B-Instruct-2507
PY=/mlx_devbox/users/caizefeng/miniconda3/envs/clamp3/bin/python
cd /home/tiger/sprag-main
echo "##### SNAPKV SANITY $(hostname) $(date) #####"
$PY -u scripts/32_snapkv_coverage.py --mode sanity --max_new_tokens 40

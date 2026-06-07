#!/bin/bash
# SnapKV coverage shard, resumable. args: GPU SHARD NUM MAXNEW LIMIT
source /home/tiger/cudafix.sh
export CUDA_VISIBLE_DEVICES=$1
export SPRAG_MODEL_PATH=/tmp/Qwen3-30B-A3B-Instruct-2507
export SHARD_ID=$2
export NUM_SHARDS=$3
MAXNEW=$4; LIMIT=$5
PY=/mlx_devbox/users/caizefeng/miniconda3/envs/clamp3/bin/python
cd /home/tiger/sprag-main
echo "##### SNAPKV coverage shard $2/$3 gpu=$1 maxnew=$MAXNEW $(hostname) $(date) #####"
$PY -u scripts/32_snapkv_coverage.py --mode coverage \
  --data 2wikimqa hotpotqa musique --ratios 0.05 0.1 0.2 0.3 0.5 \
  --kernel 7 --limit $LIMIT --max_new_tokens $MAXNEW \
  --out data/snapkv_cov.s$2.json --resume
echo "##### SNAPKV shard $2 DONE $(hostname) $(date) #####"

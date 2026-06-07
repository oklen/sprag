#!/bin/bash
# Launch 8 RAG-SnapKV shards on local GPU0-7. args: MAXNEW LIMIT START NUM
MAXNEW=${1:-1024}; LIMIT=${2:-200}; START=${3:-0}; NUM=${4:-16}
LOGD=/home/tiger/conc_logs; mkdir -p $LOGD
cd /home/tiger/sprag-main && mkdir -p data
for s in $(tmux ls 2>/dev/null | grep -oE '^rg_[0-9]+'); do tmux kill-session -t "$s" 2>/dev/null; done
for g in 0 1 2 3 4 5 6 7; do
  sid=$((START+g))
  tmux new-session -d -s rg_$sid \
    "bash /home/tiger/rag_shard.sh $g $sid $NUM $MAXNEW $LIMIT >> $LOGD/rg_$sid.log 2>&1"
done
sleep 3
echo "launched RAGSNAP shards $START..$((START+7)) of $NUM on $(hostname) GPU0-7 maxnew=$MAXNEW limit=$LIMIT"
tmux ls | grep rg_

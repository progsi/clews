#!/bin/bash

# ===== Parse CLI args =====
JOBNAME=$1
DATASET_TRAIN=$2
DATASET_TEST=$3
MODEL=$4
NNODES=${5:-1}      # default to 1
NGPUS=${6:-1}       # default to 1

if [[ -z "$DATASET_TRAIN" || -z "$MODEL" || -z "$DATASET_TEST" ]]; then
  echo "Usage: $0 <DATASET_TRAIN> <DATASET_TEST> <MODEL> [NGPUS=1] [PARTITION=gpu] [NNODES=1]"
  exit 1
fi

MODELSUB="${DATASET_TRAIN}-${MODEL}"

# Get array of files (only files, not directories)
FILES=()
for f in "$DATASET_TEST"/*; do
    [ -f "$f" ] && FILES+=("$f")
done
# Job
for FILE in "${FILES[@]}"; do
    python test.py \
        jobname="${JOBNAME}" \
        checkpoint="logs/${MODELSUB}/checkpoint_best.ckpt" \
        conf="config/${MODELSUB}.yaml" \
        path_audio="data/audio" \
        path_meta="$FILE" \   # Use the file path directly
        nnodes="$NNODES" \
        ngpus="$NGPUS"
done
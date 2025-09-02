#!/bin/bash

# ===== Parse CLI args =====
JOBNAME_PRE=$1
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

echo "Scanning directory: $DATASET_TEST"

# Get array of files (only files, not directories)
FILES=()
for f in "cache/$DATASET_TEST"/*; do
    if [ -f "$f" ]; then
        echo "Found file: $f"
        FILES+=("$f")
    fi
done

echo "Total files found: ${#FILES[@]}"
# Job
for FILE in "${FILES[@]}"; do
    BASENAME=$(basename "$FILE")         # remove path
    NAME="${BASENAME%.*}"               # remove extension
    FLOAT=$(echo "$NAME" | grep -oP '[0-9]+(\.[0-9]+)?$')
    JOBNAME="${JOBNAME_PRE}_${FLOAT}"
    python test.py \
        jobname="${JOBNAME}" \
        checkpoint="logs/${MODELSUB}/checkpoint_best.ckpt" \
        conf="config/${MODELSUB}.yaml" \
        path_audio="data/audio" \
        path_meta="$FILE" \   # Use the file path directly
        nnodes="$NNODES" \
        ngpus="$NGPUS"
done
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

csdomains=(acoustic cover instrumental live youtube_music remix reaction tutorial)

for cs in "${csdomains[@]}"; do
    python test.py \
        jobname="${JOBNAME}" \
        checkpoint="logs/${MODELSUB}/checkpoint_best.ckpt" \
        conf="config/${MODELSUB}.yaml" \
        path_audio="data/audio" \
        path_meta="cache/${DATASET_TEST}.pt" \
        nnodes="$NNODES" \
        ngpus="$NGPUS" \
        qfilter="dvi:1" \
        cfilter=matched_concepts:"$cs"
done
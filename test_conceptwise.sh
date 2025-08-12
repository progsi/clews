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

python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=acoustic
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=background
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=cover
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=instrumental
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=live
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=professional
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=remix
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=secondary
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=tutorial
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt nnodes=$NNODES ngpus=$NGPUS qsdomain=studio csdomain=studio
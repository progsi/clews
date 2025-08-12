# ===== Parse CLI args =====
DATASET_TRAIN=$1
DATASET_TEST=$2
MODEL=$3

if [[ -z "$DATASET_TRAIN" || -z "$MODEL" || -z "$DATASET_TEST" ]]; then
  echo "Usage: $0 <DATASET_TRAIN> <DATASET_TEST> <MODEL> [NGPUS=1] [PARTITION=gpu] [NNODES=1]"
  exit 1
fi

# ===== Paths =====
MODELSUB="${DATASET_TRAIN}-${MODEL}"
JOBNAME="test-${MODELSUB}-on-${DATASET_TEST}"
LOGDIR="logs/${DATASET_TRAIN}-${MODEL}/${DATASET_TEST}/full_track/slurm/"
mkdir -p "$LOGDIR"

python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=acoustic
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=background
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=cover
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=instrumental
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=live
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=professional
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=remix
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=secondary
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=tutorial
python test.py jobname=$JOBNAME checkpoint=logs/${MODELSUB}/checkpoint_best.ckpt conf=config/${MODELSUB}.yaml path_audio=data/audio path_meta=cache/metadata-${DATASET_TEST}.pt qsdomain=studio csdomain=studio
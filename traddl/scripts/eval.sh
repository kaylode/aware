MODEL_NAME=$1
TASK=$2
DATASET_NAME="mimiciv-3.1"
ROOT=/home/mpham/workspace/source/nosos
DATA_PATH=/home/mpham/workspace/source/nosos/data/ehrfm/${DATASET_NAME}/tokenized_datasets/mimiciv-behrt

# NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
export NUM_GPUS=1
export MODEL_NAME
export DATASET_NAME
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/traddl/logs"
export TASK_NAME=${TASK}

case $DATASET_NAME in
mimiciv*)
    VOCAB_DIR="/home/mpham/workspace/source/nosos/data/ehrfm/${DATASET_NAME}/tokenized_datasets/mimiciv-behrt/train"
    ;;
eicu*)
    VOCAB_DIR="/home/mpham/workspace/source/nosos/data/ehrfm/${DATASET_NAME}/tokenized_datasets/eicu-behrt/train"
    ;;
*)
    echo "Wrong experiment: '$DATASET', available are: 'mimic', 'eicu', 'ehrshot'"
    exit 1
    ;;
esac

# Evaluate on test set
PYTHONPATH=. python tools/evaluate.py \
    --config-dir ${ROOT}/traddl/configs \
    --config-name $MODEL_NAME \
    global.save_dir=${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}/${MODEL_NAME} \
    global.exp_name=${MODEL_NAME} \
    global.exist_ok=True \
    data.dataset.test.args.input_dir=${DATA_PATH}/test/SupervisedData/${TASK} \
    data.dataset.test.args.vocab_path=${VOCAB_DIR} \
    global.resume=${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}/${MODEL_NAME}/checkpoints/best.ckpt
    
MODEL_NAME=$1
TASK=$2
DATASET_NAME="mimicpipe"
ROOT=/home/mpham/workspace/source/ehrpfn
DATA_PATH="/home/mpham/workspace/source/ehrpfn/data/processed_meds/${DATASET_NAME}/Tensors/${TASK}"
VOCAB_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/${DATASET_NAME}/train"

# mimiciv-hcai
# NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
export NUM_GPUS=1
export MODEL_NAME
export DATASET_NAME
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="/home/mpham/workspace/source/ehrpfn/outputs/logs"
export TASK_NAME="MLM"

PYTHONPATH=. python tools/train.py \
    --config-dir ${ROOT}/traddl/configs \
    --config-name $MODEL_NAME \
    global.save_dir=/home/mpham/workspace/source/ehrpfn/outputs/${DATASET_NAME}/${MODEL_NAME} \
    global.exp_name=${MODEL_NAME} \
    global.exist_ok=True \
    data.dataset.train.args.input_dir=${DATA_PATH}/train \
    data.dataset.test.args.input_dir=${DATA_PATH}/test \
    data.dataset.train.args.vocab_path=${DATA_PATH}/train
    # global.resume=/home/mpham/workspace/source/ehrpfn/outputs/traddl/mimiciv-3.1/MLM/${MODEL_NAME}/checkpoints/last-v1.ckpt
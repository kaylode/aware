MODEL_NAME=$1
TASK=$2
DATASET_NAME="mimicpipe"
ROOT=/home/mpham/workspace/source/ehrpfn
DATA_PATH="/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv/Tensors/${TASK}"
VOCAB_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv/Tensors"




export NUM_GPUS=1
export MODEL_NAME
export DATASET_NAME
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/traddl/logs"
export TASK_NAME="${TASK}"
export CUDA_LAUNCH_BLOCKING=1

set -e
echo $DATA_PATH

TORCH_USE_CUDA_DSA=1  PYTHONPATH=. python tools/tune.py \
    --config-dir ${ROOT}/traddl/configs \
    --config-name ${MODEL_NAME}_tuner \
    +task_name="${TASK}" \
    global.save_dir="${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}" \
    global.exp_name=${MODEL_NAME} \
    global.exist_ok=True \
    data.dataset.train.args.input_dir="${DATA_PATH}/train.safetensors" \
    data.dataset.val.args.input_dir="${DATA_PATH}/val.safetensors" \
    data.dataset.test.args.input_dir="${DATA_PATH}/test.safetensors" \
    data.dataset.train.args.vocab_path=${VOCAB_DIR}
    # global.pretrained=${ROOT}/outputs/traddl/${DATASET_NAME}/MLM/${MODEL_NAME}mlm/checkpoints/best-v1.ckpt
    # global.resume="/home/mpham/workspace/source/ehrpfn/outputs/traddl/mimicpipe/Length of Stay/corebehrt/checkpoints/last.ckpt"
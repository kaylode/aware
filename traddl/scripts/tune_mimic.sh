MODEL_NAME=$1
DATASET_NAME="mimicpipe"
ROOT=/home/mpham/workspace/source/ehrpfn
VOCAB_DIR="/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv/Tensors"




export NUM_GPUS=1
export MODEL_NAME
export DATASET_NAME
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/traddl/logs"
export CUDA_LAUNCH_BLOCKING=1

set -e

# "Readmission" "Mortality"
for TASK in   "Length of Stay"  ; do
    DATA_PATH="/home/mpham/workspace/source/ehrpfn/data/processed_meds/mimiciv/Tensors/${TASK}"
    echo $DATA_PATH
    export TASK_NAME="${TASK}"

    TORCH_USE_CUDA_DSA=1  PYTHONPATH=. python tools/tune.py \
        --config-dir ${ROOT}/traddl/configs \
        --config-name ${MODEL_NAME}_tuner \
        data_name=${DATASET_NAME} \
        +task_name="${TASK}" \
        global.save_dir="${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}" \
        global.exp_name=${MODEL_NAME} \
        global.exist_ok=True \
        data.dataset.train.args.input_dir="${DATA_PATH}/train.safetensors" \
        data.dataset.val.args.input_dir="${DATA_PATH}/val.safetensors" \
        data.dataset.test.args.input_dir="${DATA_PATH}/test.safetensors" \
        data.dataset.train.args.vocab_path=${VOCAB_DIR}
done

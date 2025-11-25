MODEL_NAME=$1
DATASET_NAME="eicu"
ROOT=/home/mpham/workspace/source/ehrpfn
VOCAB_DIR="${ROOT}/data/processed_meds/eicu/Tensors"
# eicu-hcai
# NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
export NUM_GPUS=1
export MODEL_NAME
export DATASET_NAME
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/traddl/logs"
export CUDA_LAUNCH_BLOCKING=1

set -e
# 1702
for SEED in  1603 2000     ; do
    for TASK in  "MORTALITY" "READMISSION"; do
        export TASK_NAME="${TASK}"
        DATA_PATH="${ROOT}/data/processed_meds/eicu/Tensors/${TASK}"
        CONFIG_DIR="${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}/optuna/${MODEL_NAME}/best_configs"
        
        TORCH_USE_CUDA_DSA=1  PYTHONPATH=. python tools/train.py \
            --config-dir "${CONFIG_DIR}" \
            --config-name $MODEL_NAME \
            global.save_dir="${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}/${SEED}" \
            global.exp_name=${MODEL_NAME} \
            global.exist_ok=True \
            data.dataset.train.args.input_dir="${DATA_PATH}/train.safetensors" \
            data.dataset.val.args.input_dir="${DATA_PATH}/val.safetensors" \
            data.dataset.test.args.input_dir="${DATA_PATH}/test.safetensors" \
            data.dataset.train.args.vocab_path=${VOCAB_DIR} \
            trainer.args.max_epochs=50 \
            +global.seed=$SEED

        PYTHONPATH=. python tools/evaluate.py \
            --config-dir "${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}/${SEED}/${MODEL_NAME}" \
            --config-name pipeline \
            global.save_dir="${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}/${SEED}" \
            global.exp_name=${MODEL_NAME} \
            global.exist_ok=True \
            data.dataset.train.args.input_dir="${DATA_PATH}/train.safetensors" \
            data.dataset.val.args.input_dir="${DATA_PATH}/val.safetensors" \
            data.dataset.test.args.input_dir="${DATA_PATH}/test.safetensors" \
            data.dataset.train.args.vocab_path=${VOCAB_DIR} \
            data.dataset.test.args.vocab_path=${VOCAB_DIR} \
            global.resume="${ROOT}/outputs/traddl/${DATASET_NAME}/${TASK}/${SEED}/${MODEL_NAME}/checkpoints/best.ckpt"
    
    done
done

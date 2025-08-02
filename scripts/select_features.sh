METHOD=$1
TASK=$2

export HYDRA_FULL_ERROR=1
ROOT=/home/mpham/workspace/source/tabpfn-ehr
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

DATA_NAME=sjh_cpe #mimiciv_infections
OUTPUT_DIR="${ROOT}/outputs/tabpfn/results"

MODEL_PATH=${OUTPUT_DIR}/${DATA_NAME}/${TASK}/${METHOD}_${DATA_NAME}_${TASK}

# Loop
PYTHONPATH=. python tabpfn_ehr/selection/shap_feat.py \
    --config-path ${MODEL_PATH} \
    --config-name config.yaml \
    method=$METHOD \
    target_column=${TASK} \
    +explainable_model_path=${OUTPUT_DIR}/${DATA_NAME}/${TASK}/${METHOD}_${DATA_NAME}_${TASK}
    # +use_shap=True
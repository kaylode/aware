#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as an error

source .env

METHOD=$1
DATASET=$2 # sjh_cpe # mimicpipe
ROOT=$(pwd)

export HYDRA_FULL_ERROR=1
export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

OUTPUT_DIR="${ROOT}/${SAVE_DIR}/results"

if [ $DATASET = "eicu" ]; then
    TASKS=("MORTALITY" "READMISSION")
elif [ $DATASET = "mimiciv" ]; then
    TASKS=("Mortality" "Readmission"  "Length of Stay"   "ICD SEPSIS")
elif [ $DATASET = "sjh_cpe" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS")
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi

for TASK in "${TASKS[@]}" ; do
    echo "Selecting features for method: $METHOD, dataset: $DATASET, task: $TASK"
    MODEL_PATH="${OUTPUT_DIR}/${DATASET}/${TASK}/${METHOD}_${DATASET}_${TASK}"
    PYTHONPATH=. uv run ehrpfn/selection.py \
        --config-path "${MODEL_PATH}" \
        --config-name config.yaml \
        +explainable_model_path="${OUTPUT_DIR}/${DATASET}/${TASK}/${METHOD}_${DATASET}_${TASK}" \
        +use_shap=True
done
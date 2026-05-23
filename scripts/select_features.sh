#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as an error

source .env

METHOD=$1
DATASET=$2 # sjh_cpe # mimicpipe
ROOT=$(pwd)

export HYDRA_FULL_ERROR=1
export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

OUTPUT_DIR="${ROOT}/${SAVE_DIR}/benchmark-reproduce"
SELECTION_DIR="${ROOT}/${SAVE_DIR}/feature-selection"

if [ $DATASET = "eicu" ]; then
    TASKS=("MORTALITY" "READMISSION")
elif [ $DATASET = "eicu2" ]; then
    TASKS=("SEPSIS" "UTI")
elif [ $DATASET = "mimiciv" ]; then
    TASKS=("Mortality" "Readmission"  "Length of Stay" "ICD SEPSIS")
elif [ $DATASET = "mimiciv_hai" ]; then
    TASKS=("ICD UTI" "ICD SEPSIS" "ICD VAP")
elif [ $DATASET = "sjh_cpe" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS")
elif [ $DATASET = "sjh_cpe_frame" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS" "NEXT_LOS" "DECEASED_NEXT_EPISODE")
elif [ $DATASET = "mimiciv_hcai" ]; then
    TASKS=("SEPSIS" "UTI"  "VAP"  ) #
elif [ $DATASET = "eicu_hcai" ]; then
    TASKS=("SEPSIS" "UTI" ) #
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi

SEEDS=("1603" "2000" "1702")

for SEED in "${SEEDS[@]}" ; do
    for TASK in "${TASKS[@]}" ; do
        echo "Selecting features for method: $METHOD, dataset: $DATASET, task: $TASK"
        MODEL_PATH="${OUTPUT_DIR}/${SEED}/${DATASET}/${TASK}/${METHOD}_${DATASET}_${TASK}"
        PYTHONPATH=. uv run ehrpfn/selection.py \
            --config-path "${MODEL_PATH}" \
            --config-name config.yaml \
            inference.output_folder="${SELECTION_DIR}/${SEED}" \
            +explainable_model_path="${OUTPUT_DIR}/${SEED}/${DATASET}/${TASK}/${METHOD}_${DATASET}_${TASK}" \
            +use_shap=True
    done
done
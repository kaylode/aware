#!/bin/bash

METHOD=$1
DATASET=$2
ROOT=$(pwd)

selection_method=null #'shap_512'

set -e  # Exit on error
set -u  # Treat unset variables as an error 

if [ $DATASET = "eicu" ]; then
    TASKS=("MORTALITY" "READMISSION")
elif [ $DATASET = "eicu2" ]; then
    TASKS=("UTI" "SEPSIS") #
elif [ $DATASET = "mimiciv_hai" ]; then
    TASKS=("ICD SEPSIS" "ICD UTI"   "ICD VAP") # 
elif [ $DATASET = "mimiciv" ]; then
    TASKS=("Mortality" "Readmission"  "Length of Stay" "ICD SEPSIS")
elif [ $DATASET = "sjh_cpe" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS")
elif [ $DATASET = "openml_ehr" ]; then
    TASKS=("43550" "4541")
elif [ $DATASET = "eicu_hcai" ]; then
    TASKS=("SEPSIS" "UTI") #
elif [ $DATASET = "mimiciv_hcai" ]; then
    TASKS=("SEPSIS" "UTI"  "VAP") #
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi

for TASK in "${TASKS[@]}" ; do
    echo "Running inference for method: $METHOD, dataset: $DATASET, task: "${TASK}""

    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. uv run train.py \
        --model_type "${METHOD}" \
        --data_name $DATASET \
        --task "${TASK}" \
        --selection_method $selection_method \
        --save_path ${ROOT}/results/hyp-tune \
        --n_trials 20 \
        --max_epoch 30 \
        --batch_size 16 \
        --tune \
        --workers 2
done
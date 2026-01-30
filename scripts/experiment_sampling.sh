#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as an error

source .env

ROOT=$(pwd)
METHOD=$1
DATASET=$2
SELECTION=$3
NUM_FEATURES=$4

export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

ML_METHODS=("xgboost" "rf" "catboost" "lightgbm" "knn" "mlp" "logistics" )
DL_METHODS=("beta" "resnet" "tabr" "modernNCA" "tabm" "tabicl" "tabtransformer" "tabnet" "excelformer" "trompt" )
RATICL_METHODS=("knn-simple" "tabdpt" "ehrpfn" "knnpfn")
TICL_METHODS=("tabpfn" "tabicl")


if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=${ROOT}/data/checkpoints/tabdpt1_1.pth
else
    PRETRAIN_PATH=null
fi

if [ $DATASET = "eicu" ]; then
    TASKS=("MORTALITY" "READMISSION")
elif [ $DATASET = "mimiciv" ]; then
    TASKS=("Mortality"  "Readmission"  "Length of Stay" "ICD SEPSIS") #
elif [ $DATASET = "sjh_cpe" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS")
elif [ $DATASET = "sjh_cpe_frame" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED") # "WILL_READMIT_WITHIN_30_DAYS" "NEXT_LOS" "DECEASED_NEXT_EPISODE")
elif [ $DATASET = "openml_ehr" ]; then
    TASKS=("43550" "4541")
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi



for SAMPLING_METHOD in null oversampling_smote undersampling_random oversampling_random; do # 
    for SEED in  1702  ; do #2000  1603 
        for TASK in "${TASKS[@]}" ; do
            echo "Running inference for method: $METHOD, dataset: $DATASET, task: "${TASK}", seed: $SEED, num_features: $NUM_FEATURES, sampling_method: $SAMPLING_METHOD"

            if [[ " ${DL_METHODS[@]} " =~ " ${METHOD} " ]]; then
                CONFIG_PATH="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/args.json"
                TUNE_CONFIG="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/${METHOD}-tuned.json"
                echo "Using config path: $CONFIG_PATH"
                echo "Using tune config: $TUNE_CONFIG"
            else
                CONFIG_PATH=null
                TUNE_CONFIG=null
            fi

            if [[ " ${RATICL_METHODS[@]} " =~ " ${METHOD} " ]]; then
                RETRIEVAL_METHOD=knn_l2
            else
                RETRIEVAL_METHOD=null
            fi

            RETRIEVAL_JSON_PATH="${ROOT}/results/ehrpfn-inference/benchmark/${DATASET}/${TASK}/logistics_${DATASET}_${TASK}/logistics_params.json" 
            FEATURE_SELECTION_PATH="${ROOT}/results/ehrpfn-inference/selection/${DATASET}/${TASK}"

            PYTHONPATH=. uv run ehrpfn/inference.py \
                --config-dir ${ROOT}/configs \
                --config-name defaults \
                data.data_name=${DATASET} \
                data.num_features=${NUM_FEATURES} \
                data.task="${TASK}" \
                data.sampling.do_sampling=true \
                data.sampling.sampling_method=${SAMPLING_METHOD} \
                data.feature_selection_path="${FEATURE_SELECTION_PATH}" \
                data.selection_method=${SELECTION} \
                retrieval.retrieval_method=${RETRIEVAL_METHOD} \
                retrieval.weight_json_path="${RETRIEVAL_JSON_PATH}" \
                inference.model=$METHOD \
                inference.output_folder="${ROOT}/${SAVE_DIR}/exp/sampling/${SAMPLING_METHOD}" \
                inference.pretrained_path="${PRETRAIN_PATH}" \
                inference.config_path="${CONFIG_PATH}" \
                inference.tune_config="${TUNE_CONFIG}" \
                inference.seed=${SEED} \
                wandb.use_wandb=false
        done
    done
done
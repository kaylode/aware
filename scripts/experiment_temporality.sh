#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as an error

source .env

ROOT=$(pwd)
METHOD=$1
DATASET=$2
SELECTION=$3
NUM_FEATURES=$4 
RETRIEVAL=$5

SAMPLING=true


export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

ML_METHODS=("xgboost" "rf" "catboost" "lightgbm" "knn" "mlp" "logistics" )
DL_METHODS=("beta" "resnet" "tabr" "modernNCA" "tabm" "tabicl" "tabtransformer" "tabnet" "excelformer" "trompt" )
PFN_METHODS=("knn-simple" "tabpfn" "tabdpt" "ehrpfn" "knnpfn")


if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=${ROOT}/data/checkpoints/tabdpt1_1.pth
else
    PRETRAIN_PATH=null
fi

if [ $DATASET = "mimiciv" ]; then
    TASKS=("ICD SEPSIS")
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi


for TIMEBINS in 3 6 12 24 36; do #
    for SEED in  1702 2000  1603  ; do 
        for TASK in "${TASKS[@]}" ; do
            echo "Running inference for method: $METHOD, dataset: "${DATASET}-${TIMEBINS}bins", task: "${TASK}", seed: $SEED, num_features: $NUM_FEATURES"

            if [[ " ${DL_METHODS[@]} " =~ " ${METHOD} " ]]; then
                CONFIG_PATH="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/args.json"
                TUNE_CONFIG="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/${METHOD}-tuned.json"
                echo "Using config path: $CONFIG_PATH"
                echo "Using tune config: $TUNE_CONFIG"
            else
                CONFIG_PATH=null
                TUNE_CONFIG=null
            fi

            RETRIEVAL_JSON_PATH="${ROOT}/results/ehrpfn-inference/benchmark/${DATASET}/${TASK}/logistics_${DATASET}_${TASK}/logistics_params.json" 
            FEATURE_SELECTION_PATH="${ROOT}/results/ehrpfn-inference/selection/${DATASET}/${TASK}"

            PYTHONPATH=. uv run ehrpfn/inference.py \
                --config-dir ${ROOT}/configs \
                --config-name defaults \
                data.data_name=mimiciv_temporal \
                data.num_features=${NUM_FEATURES} \
                data.task="${TASK}" \
                data.sampling.do_sampling=${SAMPLING} \
                data.feature_selection_path="${FEATURE_SELECTION_PATH}" \
                data.selection_method=${SELECTION} \
                retrieval.retrieval_method=${RETRIEVAL} \
                retrieval.weight_json_path="${RETRIEVAL_JSON_PATH}" \
                retrieval.inf_batch_size=512 \
                inference.model=$METHOD \
                inference.output_folder="${ROOT}/${SAVE_DIR}/exp/temporality/${TIMEBINS}bins" \
                inference.pretrained_path="${PRETRAIN_PATH}" \
                inference.config_path="${CONFIG_PATH}" \
                inference.tune_config="${TUNE_CONFIG}" \
                inference.seed=${SEED} \
                wandb.use_wandb=false \
                +data.time_bins=${TIMEBINS}
        done
    done
done
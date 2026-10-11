#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as an error

source .env

ROOT=$(pwd)
METHOD=$1
DATASET=$2
SELECTION=$3
RETRIEVAL=$4
NUM_FEATURES=$5

SAMPLING=true


export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

ML_METHODS=("xgboost" "rf" "catboost" "lightgbm" "knn" "mlp" "logistics" )
DL_METHODS=("beta" "resnet" "tabr" "modernNCA" "tabm" "tabtransformer" "tabnet" "excelformer" "trompt" )
PFN_METHODS=("knn-simple" "tabpfn" "tabdpt" "tabicl" "ehrpfn" "knnpfn")


if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=${ROOT}/data/checkpoints/tabdpt1_1.pth
else
    PRETRAIN_PATH=null
fi


if [ $DATASET = "eicu" ]; then
    TASKS=("MORTALITY" "READMISSION")
elif [ $DATASET = "eicu2" ]; then
    TASKS=("UTI" "SEPSIS") #  
elif [ $DATASET = "mimiciv" ]; then
    TASKS=("Mortality" "Readmission"  "Length of Stay" )
elif [ $DATASET = "mimiciv_hai" ]; then
    TASKS=("ICD UTI" "ICD VAP" "ICD SEPSIS" ) #
elif [ $DATASET = "sjh_cpe" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS")
elif [ $DATASET = "sjh_cpe_frame" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS" "NEXT_LOS" "DECEASED_NEXT_EPISODE")
elif [ $DATASET = "openml_ehr" ]; then
    TASKS=("") #"43550" "4541") #"4531" "1097" "46612"
elif [ $DATASET = "synthea" ]; then
    TASKS=("stroke" "lung_cancer")
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi

INF_BATCH_SIZE=512
MAX_SAMPLES=10000

for IMBALANCE_RATIO in 5 10 20 50 100 200 300 400 500; do #
    for SEED in  1702  ; do #2000  1603 
        for TASK in "${TASKS[@]}" ; do
            echo "Running inference for method: $METHOD, dataset: $DATASET, task: "${TASK}", seed: $SEED, num_features: $NUM_FEATURES, imbalance_ratio: $IMBALANCE_RATIO"

            if [[ " ${DL_METHODS[@]} " =~ " ${METHOD} " ]]; then
                CONFIG_PATH="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/args.json"
                TUNE_CONFIG="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/${METHOD}-tuned.json"
                INF_BATCH_SIZE=16
                echo "Using config path: $CONFIG_PATH"
                echo "Using tune config: $TUNE_CONFIG"
            else
                CONFIG_PATH=null
                TUNE_CONFIG=null
            fi

            RETRIEVAL_JSON_PATH="${ROOT}/results/benchmark/${DATASET}/${TASK}/logistics_${DATASET}_${TASK}/logistics_params.json" 

            PYTHONPATH=. uv run ehrpfn/inference.py \
                --config-dir ${ROOT}/configs \
                --config-name defaults \
                data.data_name=${DATASET} \
                data.num_features=${NUM_FEATURES} \
                data.task="${TASK}" \
                data.sampling.do_sampling=${SAMPLING} \
                data.sampling.max_samples=${MAX_SAMPLES} \
                data.sampling.imbalance_ratio=${IMBALANCE_RATIO} \
                retrieval.inf_batch_size=${INF_BATCH_SIZE} \
                data.selection_method=${SELECTION} \
                retrieval.retrieval_method=${RETRIEVAL} \
                retrieval.weight_json_path="${RETRIEVAL_JSON_PATH}" \
                inference.model=$METHOD \
                inference.output_folder="${ROOT}/${SAVE_DIR}/exp/imbalance/${IMBALANCE_RATIO}" \
                inference.pretrained_path="${PRETRAIN_PATH}" \
                inference.config_path="${CONFIG_PATH}" \
                inference.tune_config="${TUNE_CONFIG}" \
                inference.seed=${SEED} \
                wandb.use_wandb=false
        done
    done
done
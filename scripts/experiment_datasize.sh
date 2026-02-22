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
PFN_METHODS=("knn-simple" "tabpfn" "tabdpt" "tabicl"  "ehrpfn" "knnpfn")


if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=${ROOT}/data/checkpoints/tabdpt1_1.pth
    NUM_FEATURES=100
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


for MAX_SAMPLES in 1000 2000 5000 10000 30000 50000  ; do #
    for SEED in  1702 ; do #2000  1603  
        for TASK in "${TASKS[@]}" ; do
            echo "Running inference for method: $METHOD, dataset: $DATASET, task: "${TASK}", seed: $SEED, max_samples: $MAX_SAMPLES"

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
                retrieval.inf_batch_size=${INF_BATCH_SIZE} \
                data.sampling.do_sampling=${SAMPLING} \
                data.sampling.max_samples=${MAX_SAMPLES} \
                data.selection_method=${SELECTION} \
                retrieval.retrieval_method=${RETRIEVAL} \
                retrieval.weight_json_path="${RETRIEVAL_JSON_PATH}" \
                inference.model=$METHOD \
                inference.output_folder="${ROOT}/${SAVE_DIR}/exp/samplesize/${MAX_SAMPLES}" \
                inference.pretrained_path="${PRETRAIN_PATH}" \
                inference.config_path="${CONFIG_PATH}" \
                inference.tune_config="${TUNE_CONFIG}" \
                inference.seed=${SEED} \
                wandb.use_wandb=false
        done
    done
done
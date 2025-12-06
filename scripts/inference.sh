#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as an error

source .env

ROOT=$(pwd)
METHOD=$1
DATASET=$2
SELECTION=null
SAMPLING=null
RETRIEVAL=knn 
NUM_FEATURES=null

export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

ML_METHODS=("xgboost" "rf" "catboost" "lightgbm" "knn" "mlp" "logistics" )
DL_METHODS=("beta" "resnet" "tabr" "modernNCA" "tabm" "tabicl" "tabtransformer" "tabnet" "excelformer" "trompt" )
RETRIEVAL_METHODS=("knn-simple" "tabpfn" "tabdpt" "ehrpfn")

if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=${ROOT}/data/checkpoints/tabdpt1_1.pth
    NUM_FEATURES=100
else
    PRETRAIN_PATH=null
fi

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


for SEED in  1702 2000  1603  ; do
    for TASK in "${TASKS[@]}" ; do
        echo "Running inference for method: $METHOD, dataset: $DATASET, task: $TASK, seed: $SEED"

        if [[ " ${DL_METHODS[@]} " =~ " ${METHOD} " ]]; then
            CONFIG_PATH="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/args.json"
            TUNE_CONFIG="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/${METHOD}-tuned.json"
            echo "Using config path: $CONFIG_PATH"
            echo "Using tune config: $TUNE_CONFIG"
        else
            CONFIG_PATH=null
            TUNE_CONFIG=null
        fi

        if [[ $RETRIEVAL = "weighted_logistics" ]]; then
            PRETRAIN_PATH=${ROOT}/data/checkpoints/weighted_logistics.pth
            # data.weight_json_path=f"{os.getcwd()}/outputs/all/results/{dataset_name}/{TASK}/logistics_{dataset_name}_{TASK}/logistics_params.json" \
            echo "Using pretrained path: $PRETRAIN_PATH"
        fi

        PYTHONPATH=. uv run ehrpfn/inference.py \
            --config-dir ${ROOT}/configs \
            --config-name defaults \
            data.data_name=${DATASET} \
            data.num_features=${NUM_FEATURES} \
            data.task=${TASK} \
            data.sampling.do_sampling=${SAMPLING} \
            data.selection_method=${SELECTION} \
            retrieval.retrieval_method=${RETRIEVAL} \
            inference.model=$METHOD \
            inference.output_folder="${ROOT}/${SAVE_DIR}" \
            inference.pretrained_path=${PRETRAIN_PATH} \
            inference.config_path=${CONFIG_PATH} \
            inference.tune_config=${TUNE_CONFIG} \
            inference.seed=${SEED} \
            wandb.use_wandb=false
    done
done

#!/bin/bash

set -e  # Exit on error
set -u  # Treat unset variables as an error

source .env

ROOT=$(pwd)
METHOD=$1
DATASET=$2
RETRIEVAL=$3 
SELECTION=shap
NUM_FEATURES=100
CONTEXT_SIZE=1024
INF_BATCH_SIZE=512
SAMPLING=null
DO_SAMPLING=false

export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

PFN_METHODS=("knn-simple" "tabpfn" "tabdpt" "ehrpfn" "knnpfn" "tabicl" "knnpfnv2")

if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=${ROOT}/data/checkpoints/tabdpt1_1.pth
    NUM_FEATURES=100
else
    PRETRAIN_PATH=null
fi

if [ $METHOD = "tabpfn" ]; then
    NUM_MAX_SAMPLES=10000 #tabpfn v2.0 limit
else
    NUM_MAX_SAMPLES=null
fi
PRETRAIN_RETRIEVER_PATH=null

if [ $DATASET = "eicu" ]; then
    TASKS=("MORTALITY" "READMISSION")
elif [ $DATASET = "eicu2" ]; then
    TASKS=("SEPSIS" "UTI" ) #   
elif [ $DATASET = "mimiciv" ]; then
    TASKS=("Mortality" "Readmission"  "Length of Stay" )
elif [ $DATASET = "mimiciv_hai" ]; then
    TASKS=("ICD UTI" "ICD SEPSIS"  "ICD VAP"  ) #
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


for SEED in 1603  2000 1702  ; do # 
    for TASK in "${TASKS[@]}" ; do
        echo "Running inference for method: $METHOD, dataset: $DATASET, task: "${TASK}", seed: $SEED"

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

        RETRIEVAL_JSON_PATH="${ROOT}/${SAVE_DIR}/benchmark/${DATASET}/${TASK}/logistics_${DATASET}_${TASK}/logistics_params.json" 
        PRETRAIN_RETRIEVER_PATH="${ROOT}/results/ehr-retrieval/${DATASET}/${TASK}/${SEED}/${RETRIEVAL}/retriever.pth"

        PYTHONPATH=. uv run ehrpfn/inference.py \
            --config-dir ${ROOT}/configs \
            --config-name defaults \
            data.data_name=${DATASET} \
            data.num_features=${NUM_FEATURES} \
            data.task="${TASK}" \
            data.sampling.do_sampling=${DO_SAMPLING} \
            data.sampling.sampling_method=${SAMPLING} \
            data.sampling.max_samples=${NUM_MAX_SAMPLES} \
            data.selection_method=${SELECTION} \
            retrieval.context_size=${CONTEXT_SIZE} \
            retrieval.inf_batch_size=${INF_BATCH_SIZE} \
            retrieval.retrieval_method="${RETRIEVAL}" \
            retrieval.pretrained_path="${PRETRAIN_RETRIEVER_PATH}" \
            retrieval.weight_json_path="${RETRIEVAL_JSON_PATH}" \
            inference.model="${METHOD}" \
            inference.output_folder="${ROOT}/${SAVE_DIR}/benchmark/retrieval/${SEED}" \
            inference.pretrained_path="${PRETRAIN_PATH}" \
            inference.config_path="${CONFIG_PATH}" \
            inference.tune_config="${TUNE_CONFIG}" \
            inference.seed=${SEED} \
            wandb.use_wandb=false
            # inference.output_folder="${ROOT}/${SAVE_DIR}/benchmark-reproduce/${SEED}" \
    done
done



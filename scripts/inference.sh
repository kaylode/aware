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
CONTEXT_SIZE=${6:-1024}
INF_BATCH_SIZE=${7:-512}
SAMPLING=${8:-null}

if [ $SAMPLING = "null" ] || [ -z "$SAMPLING" ]; then
    DO_SAMPLING=false
else
    DO_SAMPLING=true
fi

export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

ML_METHODS=("xgboost" "rf" "catboost" "lightgbm" "mlp" "logistics" )
DL_METHODS=("danets" "beta" "resnet" "tabr" "modernNCA" "tabm"  "tabtransformer" "tabnet" "excelformer" "trompt" )
PFN_METHODS=("knn-simple" "tabpfn" "tabdpt" "ehrpfn" "knnpfn" "tabicl" "knnpfnv2")

if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=${ROOT}/data/checkpoints/tabdpt1_1.pth
#     NUM_FEATURES=100
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
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS") # "NEXT_LOS" "DECEASED_NEXT_EPISODE")
elif [ $DATASET = "openml_ehr" ]; then
    TASKS=("") #"43550" "46612" "46612" "4531" "1097"
elif [ $DATASET = "synthea" ]; then
    TASKS=("stroke" "lung_cancer")
elif [ $DATASET = "mimiciv_hcai" ]; then
    TASKS=("SEPSIS" "UTI"  "VAP"  ) #
elif [ $DATASET = "eicu_hcai" ]; then
    TASKS=("SEPSIS" "UTI" ) #
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi


for SEED in 1603 2000 1702  ; do #   
    for TASK in "${TASKS[@]}" ; do
        echo "Running inference for method: $METHOD, dataset: $DATASET, task: "${TASK}", seed: $SEED"

        if [[ " ${DL_METHODS[@]} " =~ " ${METHOD} " ]]; then
            CONFIG_PATH="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/args.json"
            TUNE_CONFIG="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/${METHOD}-tuned.json"
            INF_BATCH_SIZE=16

            if [ $DATASET = "openml_ehr" ]; then
                if [ $SEED -eq 1603 ]; then
                    CONFIG_PATH="${ROOT}/results/old/all/hyp-tune/mimiciv_hai/ICD SEPSIS/${METHOD}/args.json"
                    TUNE_CONFIG="${ROOT}/results/old/all/hyp-tune/mimiciv_hai/ICD SEPSIS/${METHOD}/${METHOD}-tuned.json"
                elif [ $SEED -eq 2000 ]; then
                    CONFIG_PATH="${ROOT}/results/old/all/hyp-tune/eicu2/SEPSIS/${METHOD}/args.json"
                    TUNE_CONFIG="${ROOT}/results/old/all/hyp-tune/eicu2/SEPSIS/${METHOD}/${METHOD}-tuned.json"
                elif [ $SEED -eq 1702 ]; then
                    CONFIG_PATH="${ROOT}/results/old/all/hyp-tune/mimiciv/Mortality/${METHOD}/args.json"
                    TUNE_CONFIG="${ROOT}/results/old/all/hyp-tune/mimiciv/Mortality/${METHOD}/${METHOD}-tuned.json"
                fi
            fi

            echo "Using config path: $CONFIG_PATH"
            echo "Using tune config: $TUNE_CONFIG"
        elif [[ " ${ML_METHODS[@]} " =~ " ${METHOD} " ]]; then
            CONFIG_PATH=null
            TUNE_CONFIG="${ROOT}/${SAVE_DIR}/benchmark/hcai/${DATASET}/2000"
            echo "Using tune config: $TUNE_CONFIG"
        else
            CONFIG_PATH=null
            TUNE_CONFIG=null
        fi

        RETRIEVAL_JSON_PATH="${ROOT}/${SAVE_DIR}/benchmark/${DATASET}/${TASK}/logistics_${DATASET}_${TASK}/logistics_params.json" 
        
        if [ $DATASET != "openml_ehr" ]; then
            PRETRAIN_RETRIEVER_PATH="${ROOT}/results/ehr-retrieval/${DATASET}/${TASK}/${SEED}/${RETRIEVAL}/retriever.pth"
        else
            PRETRAIN_RETRIEVER_PATH="${ROOT}/results/ehr-retrieval/${DATASET}/${SEED}/${RETRIEVAL}/retriever.pth"
        fi

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
            inference.output_folder="${ROOT}/${SAVE_DIR}/benchmark-reproduce/${SEED}" \
            inference.pretrained_path="${PRETRAIN_PATH}" \
            inference.config_path="${CONFIG_PATH}" \
            inference.tune_config="${TUNE_CONFIG}" \
            inference.seed=${SEED} \
            wandb.use_wandb=false
    done
done



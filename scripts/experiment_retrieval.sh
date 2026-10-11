#!/bin/bash

# set -e  # Exit on error
# set -u  # Treat unset variables as an error

source .env

ROOT=$(pwd)
METHOD=$1
DATASET=$2
SELECTION=$3
NUM_FEATURES=$4

export LOGGING_DIR=${ROOT}/${LOGGING_DIR}/inference/

ML_METHODS=("xgboost" "rf" "catboost" "lightgbm" "knn" "mlp" "logistics" )
DL_METHODS=("beta" "resnet" "tabr" "modernNCA" "tabm" "tabtransformer" "tabnet" "excelformer" "trompt" )
PFN_METHODS=("knn-simple" "tabpfn" "tabdpt" "ehrpfn" "knnpfn" "tabicl" )


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
    TASKS=("Mortality" "Readmission"  "Length of Stay")
elif [ $DATASET = "mimiciv_hai" ]; then
    TASKS=("ICD UTI" "ICD VAP"  "ICD SEPSIS" ) #
elif [ $DATASET = "sjh_cpe" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS")
elif [ $DATASET = "sjh_cpe_frame" ]; then
    TASKS=("EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS" "NEXT_LOS" "DECEASED_NEXT_EPISODE")
elif [ $DATASET = "openml_ehr" ]; then
    TASKS=("43550" "4541")
elif [ $DATASET = "synthea" ]; then
    TASKS=("stroke" "lung_cancer")
else
    echo "Unknown dataset: $DATASET"
    exit 1  
fi
# snnl snnlattnc snnlattne snnlattne_ens snnlattne_focal snnlattnc_weighted
# bash scripts/experiment_retrieval.sh tabdpt mimiciv_hai shap 100
# random knn_l2 weighted_logistics bsnnl snnl attention balance minority_prioritized inverse_frequency balanced_snnl clinical_snnl
for RETRIEVAL_METHOD in \
    knn_l2 siamese supcon unsupcon \
    aware_attn_only \
    aware_snnl_only \
    aware_snnl_attn_only \
    aware_snnl_attn_bln_only \
    aware_balance_only \
    aware_ensemble_only \
    aware_adapter_only \
    aware_no_attn \
    aware_no_snnl \
    aware_no_balance \
    aware_no_ensemble \
    aware_imb; do 
    for SEED in 1702 1603  2000  ; do #
        for TASK in "${TASKS[@]}" ; do
            echo "Running inference for method: $METHOD, dataset: $DATASET, task: "${TASK}", seed: $SEED, num_features: $NUM_FEATURES, retrieval_method: $RETRIEVAL_METHOD"

            if [[ " ${DL_METHODS[@]} " =~ " ${METHOD} " ]]; then
                CONFIG_PATH="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/args.json"
                TUNE_CONFIG="${ROOT}/${TALENT_TUNED_DIR}/${DATASET}/${TASK}/${METHOD}/${METHOD}-tuned.json"
                echo "Using config path: $CONFIG_PATH"
                echo "Using tune config: $TUNE_CONFIG"
            else
                CONFIG_PATH=null
                TUNE_CONFIG=null
            fi

            RETRIEVAL_JSON_PATH="${ROOT}/${SAVE_DIR}/benchmark/${DATASET}/${TASK}/logistics_${DATASET}_${TASK}/logistics_params.json" 

            PYTHONPATH=. uv run ehrpfn/inference.py \
                --config-dir ${ROOT}/configs \
                --config-name defaults \
                data.data_name=${DATASET} \
                data.num_features=${NUM_FEATURES} \
                data.task="${TASK}" \
                data.sampling.do_sampling=false \
                data.selection_method=${SELECTION} \
                retrieval.retrieval_method=${RETRIEVAL_METHOD} \
                retrieval.weight_json_path="${RETRIEVAL_JSON_PATH}" \
                inference.model=$METHOD \
                inference.output_folder="${ROOT}/${SAVE_DIR}/exp/retrieval/${SEED}/${RETRIEVAL_METHOD}" \
                inference.pretrained_path="${PRETRAIN_PATH}" \
                inference.config_path="${CONFIG_PATH}" \
                inference.tune_config="${TUNE_CONFIG}" \
                inference.seed=${SEED} \
                wandb.use_wandb=false
        done
    done
done
METHOD=$1
# TASK=$2

export HYDRA_FULL_ERROR=1
ROOT=/home/mpham/workspace/source/ehrpfn
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

DATA_NAME=sjh_cpe # eicu #
OUTPUT_DIR="${ROOT}/outputs/all"


# Loop "ICD VAP" "ICD SSI" "ICD UTI" "ICD BSI"  
# for TASK in "Readmission" ; do
# for TASK in "ICD SEPSIS" ; do
# for TASK in "READMISSION" "MORTALITY"; do
# for TASK in "Mortality" "Length of Stay" "ICD UTI"; do
for TASK in "EPISODE_CPE_INFECTION_FLAGGED"; do
    MODEL_PATH="${OUTPUT_DIR}/${DATA_NAME}/${TASK}/${METHOD}_${DATA_NAME}_${TASK}"
    PYTHONPATH=. python ehrpfn/selection/shap_feat.py \
        --config-path "${MODEL_PATH}" \
        --config-name config.yaml \
        output_folder="${OUTPUT_DIR}" \
        +explainable_model_path="${OUTPUT_DIR}/${DATA_NAME}/${TASK}/${METHOD}_${DATA_NAME}_${TASK}" \
        +use_shap=True
done
METHOD=$1
# TASK=$2

export HYDRA_FULL_ERROR=1
ROOT=/home/mpham/workspace/source/ehrpfn
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

DATA_NAME=mimicpipe #mimiciv_outcomes #mimiciv_infections
OUTPUT_DIR="${ROOT}/outputs/tabpfn-test/results"


# Loop "ICD VAP" "ICD SSI" "ICD UTI" "ICD BSI"  "Mortality" "Length of Stay"
for TASK in "Readmission" ; do
# for TASK in "ICD SEPSIS" ; do
    MODEL_PATH="${OUTPUT_DIR}/${DATA_NAME}/${TASK}/${METHOD}_${DATA_NAME}_${TASK}"
    PYTHONPATH=. python ehrpfn/selection/shap_feat.py \
        --config-path "${MODEL_PATH}" \
        --config-name config.yaml \
        +explainable_model_path="${OUTPUT_DIR}/${DATA_NAME}/${TASK}/${METHOD}_${DATA_NAME}_${TASK}" \
        +use_shap=True
done

# train_label_path="${ROOT}/data/labels/PATIENT_OUTCOME/${TASK}_train_labels.csv" \
# test_label_path="${ROOT}/data/labels/PATIENT_OUTCOME/${TASK}_test_labels.csv" \
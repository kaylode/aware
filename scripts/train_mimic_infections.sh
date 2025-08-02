METHOD=$1

ROOT=/home/mpham/workspace/source/tabpfn-ehr
DATA_FOLDER="${ROOT}/data/mimiciv-icu-gbms-events"
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=shap
SAMPLING=null
RETRIEVAL=trajectories #activities_tfidf_kmeans #trajectories #activities random #
PRETRAIN_MODE=HEALTH

# Loop
# for loop
# for SELECTION in null "xgboost" "rf" "shap"
for TASK in "SEPSIS" "VAP" "BSI"; do
    PYTHONPATH=. python tabpfn_ehr/inference.py --config-name mimiciv_infections \
        train_label_path=${ROOT}/outputs/tabpfn/labels/PATIENT_INFECTION/${TASK}_train_labels.csv \
        test_label_path=${ROOT}/outputs/tabpfn/labels/PATIENT_INFECTION/${TASK}_test_labels.csv \
        method=$METHOD \
        num_features=100 \
        target_column=${TASK} \
        sampling=${SAMPLING} \
        selection_method=${SELECTION} \
        retrieval_method=${RETRIEVAL} \
        feature_selection_path=${ROOT}/outputs/tabpfn/selection/mimiciv_infections/${TASK} \
        n_process_variants=50 \
        context_size=1024 \
        pretrain_mode=${PRETRAIN_MODE}
done
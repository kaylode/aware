METHOD=$1

ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=shap
SAMPLING=null
RETRIEVAL=null #knn #activities  #activities_tfidf_kmeans #trajectories #activities random #
PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/talent/mimicpipe/Readmission/tabicl/epoch-last-14.dill


# Loop "ICD SEPSIS" "ICD VAP" "ICD SSI" "ICD UTI" "ICD BSI" 
# Loop  "Readmission" "Mortality" "Length of Stay"
    # CONFIG_PATH="${ROOT}/outputs/talent/mimicpipe/${TASK}/${METHOD}/args.json"
        # +config_path=${CONFIG_PATH}
        # pretrained_path=${PRETRAIN_PATH}
for TASK in "ICD SEPSIS" "Readmission"; do
    PYTHONPATH=. python ehrpfn/inference.py --config-name mimiciv \
        data_csv="${ROOT}/data/processed/mimiciv/${TASK}/all_data.csv" \
        train_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/train_labels.csv" \
        test_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/test_labels.csv" \
        method=$METHOD \
        num_features=100 \
        target_column="${TASK}" \
        sampling=${SAMPLING} \
        selection_method=${SELECTION} \
        retrieval_method=${RETRIEVAL} \
        feature_selection_path="${ROOT}/outputs/tabpfn/selection/mimicpipe/${TASK}" \
        embedding_file_path=null \
        context_size=1024
done

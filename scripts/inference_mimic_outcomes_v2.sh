METHOD=$1

ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=null
SAMPLING=null
RETRIEVAL=knn #faiss #activities_tfidf_kmeans #trajectories #activities random #

if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/data/checkpoints/tabdpt1_1.pth
else
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-GENERAL-HEALTH-SYNTHETIC-100/best_cls-health-auc.ckpt
fi

# Loop  "Readmission"
for TASK in  "Mortality" "Length of Stay" ; do
    PYTHONPATH=. python ehrpfn/inference.py --config-name mimiciv_outcomes_v2 \
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
        process_file_path="/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/${TASK}/process" \
        trajectory_file_path="/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/${TASK}/trajectories" \
        n_process_variants=8 \
        context_size=1024 \
        pretrained_path=${PRETRAIN_PATH}
done

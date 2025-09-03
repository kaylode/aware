METHOD=$1

ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=shap
SAMPLING=null
RETRIEVAL=knn #faiss #activities_tfidf_kmeans #trajectories #activities random #

if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/data/checkpoints/tabdpt1_1.pth
else
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-GENERAL-HEALTH-SYNTHETIC-100/best_cls-health-auc.ckpt
fi


for TASK in "READMISSION" "MORTALITY"; do
PYTHONPATH=. python ehrpfn/inference.py --config-name eicu \
    train_admission_id_file=${ROOT}/data/processed_muse/eicu/task:${TASK}/train_admission_ids.txt \
    val_admission_id_file=${ROOT}/data/processed_muse/eicu/task:${TASK}/val_admission_ids.txt \
    test_admission_id_file=${ROOT}/data/processed_muse/eicu/task:${TASK}/test_admission_ids.txt \
    method=$METHOD \
    num_features=100 \
    target_column=${TASK} \
    sampling=${SAMPLING} \
    selection_method=${SELECTION} \
    retrieval_method=${RETRIEVAL} \
    feature_selection_path="${ROOT}/outputs/tabpfn/selection/eicu/${TASK}" \
    context_size=1024 \
    pretrained_path=${PRETRAIN_PATH}
done

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
PYTHONPATH=. python ehrpfn/inference.py --config-name openml_ehr \
    method=$METHOD \
    num_features=500 \
    n_trials=5 \
    target_column=openml_ehr \
    sampling=${SAMPLING} \
    selection_method=null \
    retrieval_method=${RETRIEVAL} \
    feature_selection_path=null \
    context_size=1024 \
    pretrained_path=${PRETRAIN_PATH} \
    seed=1702

METHOD=$1

ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=random
SAMPLING=null
RETRIEVAL=activities #activities_tfidf_kmeans #trajectories #activities random #
PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-HEALTH-MIMICIV-SYNTHETIC-100/latest.ckpt


for TASK in "EPISODE_CPE_INFECTION_FLAGGED" "WILL_READMIT_WITHIN_30_DAYS" "DECEASED_NEXT_EPISODE" "NEXT_LOS"; do
    PYTHONPATH=. python ehrpfn/inference.py --config-name sjh_cpe \
        method=$METHOD \
        num_features=100 \
        target_column=${TASK} \
        sampling=${SAMPLING} \
        selection_method=${SELECTION} \
        retrieval_method=${RETRIEVAL} \
        feature_selection_path=${ROOT}/outputs/tabpfn/selection/sjh_cpe/${TASK} \
        n_process_variants=8 \
        context_size=1024 \
        pretrained_path=${PRETRAIN_PATH}
done
METHOD=$1

ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=shap
SAMPLING=null
RETRIEVAL=knn #knn dbr weighted random # weighted_logistics # weighted_feattype
NUM_FEATURES=500

if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/data/checkpoints/tabdpt1_1.pth
    NUM_FEATURES=100
elif [ $METHOD = "ehrpfn500" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/TabDPT-GENERAL-500/best_cls-cc18-auc.ckpt
    NUM_FEATURES=500
elif [ $METHOD = "ehrpfn-mimiciv-synthetic" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-GENERAL-HEALTH-MIMICIV-SYNTHETIC-500/best_cls-cc18-auc.ckpt
    NUM_FEATURES=500
else
    PRETRAIN_PATH=null
fi
## "DECEASED_NEXT_EPISODE" "NEXT_LOS"  1603 2000
# 2000  1603 2000 "EPISODE_CPE_INFECTION_FLAGGED"   "WILL_READMIT_WITHIN_30_DAYS"
for SEED in 1603 2000  ; do
    for TASK in "EPISODE_CPE_INFECTION_FLAGGED" ; do
        CONFIG_PATH="${ROOT}/outputs/all/hyp-tune/sjh_cpe/${TASK}/${METHOD}/args.json"
        TUNE_CONFIG="${ROOT}/outputs/all/hyp-tune/sjh_cpe/${TASK}/${METHOD}/${METHOD}-tuned.json"
                
        PYTHONPATH=. python ehrpfn/inference.py \
            --config-dir ${ROOT}/configs/data \
            --config-name sjh_cpe \
            method=$METHOD \
            num_features=${NUM_FEATURES} \
            target_column=${TASK} \
            sampling=${SAMPLING} \
            selection_method=${SELECTION} \
            retrieval_method=${RETRIEVAL} \
            feature_selection_path="${ROOT}/outputs/all/selection/sjh_cpe/${TASK}" \
            context_size=1024 \
            pretrained_path=${PRETRAIN_PATH} \
            +config_path="${CONFIG_PATH}" \
            +tune_config="${TUNE_CONFIG}" \
            output_folder="/home/mpham/workspace/source/ehrpfn/outputs/all" \
            seed=${SEED}
    done
done
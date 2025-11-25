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
elif [ $METHOD = "ehrpfn-synthetic" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/all/training/pretrain/EHRPFN-GENERAL-SYNTHETIC-500/best_auc.ckpt
    NUM_FEATURES=500
elif [ $METHOD = "ehrpfn-finetune" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/all/training/finetune/eicu/READMISSION/tabdpt_11/best_auc.ckpt
    NUM_FEATURES=500
else
    PRETRAIN_PATH=null
fi
# 2000  1603  "MORTALITY" 
# PRETRAIN_PATH="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/results/eicu/MORTALITY/beta_eicu_MORTALITY_sel:shap_500/best-val-2000.pth"
for SEED in  1702 ; do
    for TASK in "READMISSION" ; do
        CONFIG_PATH="${ROOT}/outputs/all/hyp-tune/eicu/${TASK}/${METHOD}/args.json"
        TUNE_CONFIG="${ROOT}/outputs/all/hyp-tune/eicu/${TASK}/${METHOD}/${METHOD}-tuned.json"
        PYTHONPATH=. python ehrpfn/inference.py \
            --config-dir ${ROOT}/configs/data \
            --config-name eicu \
            train_admission_id_file=${ROOT}/data/processed/eicu/task:${TASK}/train_admission_ids.txt \
            val_admission_id_file=${ROOT}/data/processed/eicu/task:${TASK}/val_admission_ids.txt \
            test_admission_id_file=${ROOT}/data/processed/eicu/task:${TASK}/test_admission_ids.txt \
            method=$METHOD \
            num_features=${NUM_FEATURES} \
            target_column=${TASK} \
            sampling=${SAMPLING} \
            selection_method=${SELECTION} \
            retrieval_method=${RETRIEVAL} \
            feature_selection_path="${ROOT}/outputs/all/selection/eicu/${TASK}" \
            embedding_file_path="${ROOT}/data/processed/eicu/task:${TASK}/trajectories" \
            output_folder="/home/mpham/workspace/source/ehrpfn/outputs/all" \
            context_size=1024 \
            pretrained_path=${PRETRAIN_PATH} \
            +config_path=${CONFIG_PATH} \
            +tune_config=${TUNE_CONFIG} \
            seed=${SEED}
    done
done

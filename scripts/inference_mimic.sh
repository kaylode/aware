METHOD=$1

ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=shap
SAMPLING=null
RETRIEVAL=weighted_logistics #knn dbr weighted random # weighted_logistics # weighted_feattype
NUM_FEATURES=500

if [ $METHOD = "tabdpt" ]; then
    PRETRAIN_PATH="/home/mpham/workspace/source/ehrpfn/data/checkpoints/tabdpt1_1.pth"
    NUM_FEATURES=100
elif [ $METHOD = "tabdptv2" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/data/checkpoints/tabdpt1_1.pth
    NUM_FEATURES=100
elif [ $METHOD = "ehrpfn" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-MIMICIV-100/best_cls-mimicpipe-auc.ckpt
    NUM_FEATURES=100
elif [ $METHOD = "ehrpfn-mimiciv-synthetic" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-MIMICIV-SYNTHETIC-500/best_cls-mimicpipe-auc.ckpt
    NUM_FEATURES=100
    METHOD="ehrpfn"
else
    PRETRAIN_PATH=null
fi
#  
for SEED in 2000 1702 1603  ; do
    for TASK in  "Length of Stay" "Mortality" "Readmission" "ICD SEPSIS" ; do
        CONFIG_PATH="${ROOT}/outputs/talent-tune/mimicpipe/${TASK}/${METHOD}/args.json"
        TUNE_CONFIG="${ROOT}/outputs/talent-tune/mimicpipe/${TASK}/${METHOD}/${METHOD}-tuned.json"
        # PRETRAIN_PATH="/home/mpham/workspace/source/ehrpfn/outputs/talent-tune/mimicpipe/${TASK}/tabicl/epoch-best-${SEED}.dill"
        PYTHONPATH=. python ehrpfn/inference.py --config-name mimiciv \
            data_csv="${ROOT}/data/processed/mimiciv/${TASK}/all_data.csv" \
            train_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/train_labels.csv" \
            test_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/test_labels.csv" \
            method=$METHOD \
            num_features=$NUM_FEATURES \
            target_column="${TASK}" \
            sampling=${SAMPLING} \
            selection_method=${SELECTION} \
            retrieval_method=${RETRIEVAL} \
            feature_selection_path="${ROOT}/outputs/tabpfn/selection/mimicpipe/${TASK}" \
            context_size=1024 \
            pretrained_path="${PRETRAIN_PATH}" \
            embedding_file_path="/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/${TASK}/trajectories" \
            output_folder="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn" \
            seed=${SEED} \
            +config_path="${CONFIG_PATH}" \
            +tune_config="${TUNE_CONFIG}"
    done
done

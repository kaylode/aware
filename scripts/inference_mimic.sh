METHOD=$1

ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=shap
SAMPLING=null
RETRIEVAL=weighted_feattype #knn dbr weighted random # weighted_logistics # weighted_feattype
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

#      1702  
for SEED in 1603  ; do
    for TASK in  "Mortality" "Readmission"  "Length of Stay"   "ICD SEPSIS"  ; do
        # CONFIG_PATH="${ROOT}/outputs/talent-tune/mimicpipe/${TASK}/${METHOD}/args.json"
        # TUNE_CONFIG="${ROOT}/outputs/talent-tune/mimicpipe/${TASK}/${METHOD}/${METHOD}-tuned.json"
        # PRETRAIN_PATH="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/results/mimicpipe/ICD SEPSIS/beta_mimicpipe_ICD SEPSIS_sel:shap_500/best-val-2000.pth"
        PYTHONPATH=. python ehrpfn/inference.py \
            --config-dir ${ROOT}/configs/data \
            --config-name mimiciv \
            data_csv="${ROOT}/data/processed/mimiciv/${TASK}/all_data.csv" \
            train_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/train_labels.csv" \
            test_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/test_labels.csv" \
            method=$METHOD \
            num_features=$NUM_FEATURES \
            target_column="${TASK}" \
            sampling=${SAMPLING} \
            selection_method=${SELECTION} \
            retrieval_method=${RETRIEVAL} \
            feature_selection_path="${ROOT}/outputs/all/selection/mimicpipe/${TASK}" \
            context_size=1024 \
            pretrained_path="${PRETRAIN_PATH}" \
            embedding_file_path="/home/mpham/workspace/source/ehrpfn/data/processed/mimiciv/${TASK}/trajectories" \
            output_folder="/home/mpham/workspace/source/ehrpfn/outputs/all/results" \
            seed=${SEED}
            # +config_path="${CONFIG_PATH}" \
            # +tune_config="${TUNE_CONFIG}"
    done
done

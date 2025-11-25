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
elif [ $METHOD = "ehrpfn" ]; then
    PRETRAIN_PATH=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-GENERAL-HEALTH-SYNTHETIC-100/best_cls-health-auc.ckpt
else
    PRETRAIN_PATH=null
fi


for SEED in 1992 1963 1702 ; do
    # PRETRAIN_PATH="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/results/openml_ehr/COVID-19-Hospitals-Treatment-Plan/beta_COVID-19-Hospitals-Treatment-Plan_COVID-19-Hospitals-Treatment-Plan/best-val-${SEED}.pth"

    if [ $SEED = 1702 ]; then
        TASK="Readmission"
        DATA='mimicpipe'
    elif [ $SEED = 1603 ]; then
        TASK="ICD SEPSIS"
        DATA='mimicpipe'
    elif [ $SEED = 2000 ]; then
        TASK="Mortality"
        DATA='mimicpipe'
    elif [ $SEED = 1992 ]; then
        TASK="READMISSION"
        DATA='eicu'
    else
        TASK="MORTALITY"
        DATA='eicu'
    fi

    CONFIG_PATH="${ROOT}/outputs/all/hyp-tune/${DATA}/${TASK}/${METHOD}/args.json"
    TUNE_CONFIG="${ROOT}/outputs/all/hyp-tune/${DATA}/${TASK}/${METHOD}/${METHOD}-tuned.json"

    PYTHONPATH=. python ehrpfn/inference.py \
        --config-dir ${ROOT}/configs/data \
        --config-name openml_ehr \
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
        output_folder="/home/mpham/workspace/source/ehrpfn/outputs/all/results" \
        seed=${SEED} \
        +config_path="${CONFIG_PATH}" \
        +tune_config="${TUNE_CONFIG}"
done

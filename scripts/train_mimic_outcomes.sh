METHOD=$1

ROOT=/home/mpham/workspace/source/tabpfn-ehr
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

set -e  # Exit on error
set -u  # Treat unset variables as an error

SELECTION=random
SAMPLING=null
RETRIEVAL=faiss #activities_tfidf_kmeans #trajectories #activities random #
PRETRAIN_MODE=HEALTH

#  
# Loop 
for TASK in "DEATH" "WILL_READMIT_WITHIN_7_DAYS"  "NEXT_ADMISSION_LOS" ; do
    PYTHONPATH=. python tabpfn_ehr/inference.py --config-name mimiciv_outcomes \
        train_label_path=${ROOT}/data/mimiciv-icu-gbms-events/train/SupervisedData/PATIENT_OUTCOME/${TASK}_prediction_labels.csv \
        test_label_path=${ROOT}/data/mimiciv-icu-gbms-events/test/SupervisedData/PATIENT_OUTCOME/${TASK}_prediction_labels.csv \
        method=$METHOD \
        num_features=100 \
        target_column=${TASK} \
        sampling=${SAMPLING} \
        selection_method=${SELECTION} \
        retrieval_method=${RETRIEVAL} \
        feature_selection_path=${ROOT}/outputs/tabpfn/selection/mimiciv_outcomes/${TASK} \
        n_process_variants=8 \
        context_size=1024 \
        pretrain_mode=${PRETRAIN_MODE}
done

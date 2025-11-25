DATA_NAME=$1
ROOT=/home/mpham/workspace/source/ehrpfn
export HYDRA_FULL_ERROR=1
export LOGGING_DIR="${ROOT}/outputs/tabpfn/logs"

SELECTION='shap_500'
SAMPLING=null
RETRIEVAL=knn #knn dbr weighted random # weighted_logistics # weighted_feattype
SEED=1702
set -e  # Exit on error
set -u  # Treat unset variables as an error 

if [ "$DATA_NAME" = "mimicpipe" ]; then
    # MIMICIV
    for TASK in "Readmission" "ICD SEPSIS" "Mortality" "Length of Stay" "ICD UTI"; do
        echo "${TASK}"
        feature_selection_path="${ROOT}/outputs/all/selection/${DATA_NAME}/${TASK}"
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ehrpfn/tune_tabpfn.py --config-name mimiciv \
            data_csv="${ROOT}/data/processed/mimiciv/${TASK}/all_data.csv" \
            train_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/train_labels.csv" \
            test_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/test_labels.csv" \
            method=tabpfn \
            num_features=500 \
            target_column="${TASK}" \
            sampling=${SAMPLING} \
            selection_method=${SELECTION} \
            retrieval_method=${RETRIEVAL} \
            feature_selection_path="${ROOT}/outputs/all/selection/mimicpipe/${TASK}" \
            context_size=1024 \
            output_folder="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn-tune" \
            seed=${SEED}
    done
elif [ "$DATA_NAME" = "eicu" ]; then
    # eICU
    for TASK in "READMISSION" "MORTALITY"; do
        feature_selection_path="${ROOT}/outputs/all/selection/${DATA_NAME}/${TASK}"
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ehrpfn/tune_tabpfn.py --config-name eicu \
            train_admission_id_file=${ROOT}/data/processed/eicu/task:${TASK}/train_admission_ids.txt \
            val_admission_id_file=${ROOT}/data/processed/eicu/task:${TASK}/val_admission_ids.txt \
            test_admission_id_file=${ROOT}/data/processed/eicu/task:${TASK}/test_admission_ids.txt \
            method=tabpfn \
            num_features=500 \
            target_column=${TASK} \
            sampling=${SAMPLING} \
            selection_method=${SELECTION} \
            retrieval_method=${RETRIEVAL} \
            feature_selection_path="${ROOT}/outputs/all/selection/eicu/${TASK}" \
            context_size=1024 \
            output_folder="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn-tune" \
            seed=${SEED}
    done
else
    # OpenML EHRs
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ehrpfn/tune_tabpfn.py \
        --data_name $DATA_NAME \
        --task openml_ehr \
        --data_folder "/home/mpham/workspace/source/ehrpfn/data/pretraining" \
        --selection_method null \
        --feature_selection_path null
fi

# tabr & modernNCA
# assert(args.cat_policy == 'tabr_ohe')
# assert(args.num_policy == 'none')

# tabicl 
# assert(args.normalization == 'none')
# assert(args.cat_policy == 'indices')
# assert(args.num_policy == 'none')
# assert(args.tune != True)

# tabm
# assert(args.cat_policy == 'indices')
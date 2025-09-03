MODEL_NAME=$1
DATA_NAME=$2
ROOT=/home/mpham/workspace/source/ehrpfn

selection_method='shap_1024'

set -e  # Exit on error
set -u  # Treat unset variables as an error 

if [ "$DATA_NAME" = "mimicpipe" ]; then
    # MIMICIV
    for TASK in "Readmission" "ICD SEPSIS" "Mortality" "Length of Stay" "ICD UTI"; do
        echo "${TASK}"
        feature_selection_path="${ROOT}/outputs/tabpfn/selection/${DATA_NAME}/${TASK}"
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ehrpfn/train_talent.py \
            --data_name "${DATA_NAME}" \
            --task "${TASK}" \
            --data_csv "${ROOT}/data/processed/mimiciv/${TASK}/all_data.csv" \
            --train_label_path "${ROOT}/data/processed/mimiciv/${TASK}/split/train_labels.csv" \
            --test_label_path "${ROOT}/data/processed/mimiciv/${TASK}/split/test_labels.csv" \
            --selection_method "${selection_method}" \
            --feature_selection_path "${feature_selection_path}" \
            --model_type "${MODEL_NAME}" \
            --save_path /home/mpham/workspace/source/ehrpfn/outputs/talent-tune \
            --tune
    done
elif [ "$DATA_NAME" = "eicu" ]; then
    # eICU
    for TASK in "READMISSION" "MORTALITY"; do
        feature_selection_path="${ROOT}/outputs/tabpfn/selection/${DATA_NAME}/${TASK}"
        CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ehrpfn/train_talent.py \
            --data_name $DATA_NAME \
            --task $TASK \
            --data_csv /home/mpham/workspace/source/ehrpfn/data/processed_muse/eicu/all_data.csv \
            --label_path /home/mpham/workspace/source/ehrpfn/data/processed_muse/eicu/all_labels.csv \
            --train_admission_id_file ${ROOT}/data/processed_muse/eicu/task:${TASK}/train_admission_ids.txt \
            --val_admission_id_file ${ROOT}/data/processed_muse/eicu/task:${TASK}/val_admission_ids.txt \
            --test_admission_id_file ${ROOT}/data/processed_muse/eicu/task:${TASK}/test_admission_ids.txt \
            --selection_method $selection_method \
            --feature_selection_path $feature_selection_path \
            --model_type $MODEL_NAME \
            --save_path /home/mpham/workspace/source/ehrpfn/outputs/talent-tune \
            --tune
    done
else
    # OpenML EHRs
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ehrpfn/train_talent.py \
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
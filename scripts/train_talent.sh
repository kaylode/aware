MODEL_NAME=$1
TASK=$2
ROOT=/home/mpham/workspace/source/ehrpfn
data_csv="${ROOT}/data/processed/mimiciv/${TASK}/all_data.csv"
train_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/train_labels.csv"
test_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/test_labels.csv"
feature_selection_path="${ROOT}/outputs/tabpfn/selection/mimicpipe/${TASK}"
selection_method='shap_100'

set -e  # Exit on error
set -u  # Treat unset variables as an error

CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. python ehrpfn/train_talent.py \
    --task $TASK \
    --data_csv $data_csv \
    --train_label_path $train_label_path \
    --test_label_path $test_label_path \
    --selection_method $selection_method \
    --feature_selection_path $feature_selection_path \
    --model_type $MODEL_NAME \
    --save_path /home/mpham/workspace/source/ehrpfn/outputs/talent \
    --cat_policy tabr_ohe \
    --num_policy none
    # --normalization none


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

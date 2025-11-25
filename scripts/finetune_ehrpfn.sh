export LOGGING_DIR="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/logs"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

FREEZE_TRANSFORMER=true

NUM_FEATURES=500
NUM_MODEL_UPDATES=512
EVAL_EVERY=1
NUM_EPOCHS=5
EVAL_ONLY=true
RESUME=null #//home/mpham/workspace/source/ehrpfn/outputs/all/training/pretrain/EHRPFN-MIMICIV-SYNTHETIC-500
PRETRAINED=/home/mpham/workspace/source/ehrpfn/outputs/all/training/finetune/eicu/READMISSION/tabdpt_11/best_auc.ckpt
ROOT="/home/mpham/workspace/source/ehrpfn"
MODEL_NAME=ehrpfn

DATA_NAME=$1
TASK_NAME=$2

if [ "$DATA_NAME" = "eicu" ]; then
  USE_EICU=true
  USE_MIMICIV=false
elif [ "$DATA_NAME" = "mimiciv" ]; then
  USE_EICU=false
  USE_MIMICIV=true
fi

export DATA_NAME
export MODEL_NAME

CUDA_VISIBLE_DEVICES=0 python ehrpfn/training/train.py \
  --config-dir ${ROOT}/configs/training \
  --config-name ${MODEL_NAME} \
  folder="${ROOT}/outputs/all/training/finetune/${DATA_NAME}/${TASK_NAME}" \
  exp_name="${MODEL_NAME}" \
  model.max_num_features=$NUM_FEATURES \
  model.freeze_transformer=$FREEZE_TRANSFORMER \
  training.batch_size=16 \
  training.run_pretrain=false \
  training.run_finetune=true \
  training.min_eval_pos=1023 \
  training.max_eval_pos=1023 \
  training.seq_len=1024 \
  data.task_name="${TASK_NAME}" \
  training.num_epochs=$NUM_EPOCHS \
  training.resume=$RESUME \
  training.pretrained=$PRETRAINED \
  training.num_model_updates=$NUM_MODEL_UPDATES \
  logging.eval_every=$EVAL_EVERY \
  training.eval_only=$EVAL_ONLY \
  training.lr=0.00001 \
  data.use_mimiciv=$USE_MIMICIV \
  data.mimiciv.data_csv="${ROOT}/data/processed/mimiciv/${TASK_NAME}/all_data.csv" \
  data.mimiciv.train_label_path="${ROOT}/data/processed/mimiciv/${TASK_NAME}/split/train_labels.csv" \
  data.mimiciv.test_label_path="${ROOT}/data/processed/mimiciv/${TASK_NAME}/split/test_labels.csv" \
  data.mimiciv.feature_selection_path="${ROOT}/outputs/all/selection/mimicpipe/${TASK_NAME}" \
  data.mimiciv.selection_method="shap_${NUM_FEATURES}" \
  data.use_eicu=$USE_EICU \
  data.eicu.task="${TASK_NAME}" \
  data.eicu.csv_file="${ROOT}/data/processed/eicu/all_data.csv" \
  data.eicu.label_file="${ROOT}/data/processed/eicu/all_labels.csv" \
  data.eicu.train_admission_id_file="${ROOT}/data/processed/eicu/task:${TASK_NAME}/train_admission_ids.txt" \
  data.eicu.val_admission_id_file="${ROOT}/data/processed/eicu/task:${TASK_NAME}/val_admission_ids.txt" \
  data.eicu.test_admission_id_file="${ROOT}/data/processed/eicu/task:${TASK_NAME}/test_admission_ids.txt" \
  data.eicu.feature_selection_path="${ROOT}/outputs/all/selection/eicu/${TASK_NAME}" \
  data.eicu.selection_method="shap_${NUM_FEATURES}"

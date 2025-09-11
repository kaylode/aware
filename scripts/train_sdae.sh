export LOGGING_DIR="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/logs"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

ROOT="/home/mpham/workspace/source/ehrpfn"
TASK="ICD SEPSIS"

CUDA_VISIBLE_DEVICES=0 python ehrpfn/representation/train.py \
  training.task="${TASK}" \
  model.loss_function=mse \
  training.data_csv="${ROOT}/data/processed/mimiciv/${TASK}/all_data.csv" \
  training.train_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/train_labels.csv" \
  training.test_label_path="${ROOT}/data/processed/mimiciv/${TASK}/split/test_labels.csv" \
  training.save_path="${ROOT}/outputs/pretraining/ptsdae/${TASK}" \
  training.feature_selection_path="${ROOT}/outputs/tabpfn/selection/mimicpipe/${TASK}" \
  training.selection_method=shap_1024
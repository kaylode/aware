export LOGGING_DIR="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/logs"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1
export DATA_NAME="pretrain"

FREEZE_TRANSFORMER=false
USE_GENERAL_DATA=true
USE_HEALTH_DATA=false
USE_MIMICIV=false
USE_SYNTHETIC=false
NUM_FEATURES=500
NUM_MODEL_UPDATES=512
EVAL_EVERY=1
NUM_EPOCHS=2048
EVAL_ONLY=true
RESUME=null #/home/mpham/workspace/source/ehrpfn/outputs/all/training/pretrain/EHRPFN-GENERAL-SYNTHETIC-500
PRETRAINED=/home/mpham/workspace/source/ehrpfn/outputs/all/training/pretrain/EHRPFN-GENERAL-SYNTHETIC-500/best_auc.ckpt
ROOT="/home/mpham/workspace/source/ehrpfn"


MODEL_NAME_=NanoTabPFN

if $USE_GENERAL_DATA; then
  MODEL_NAME_="${MODEL_NAME_}-GENERAL"
fi

if $USE_HEALTH_DATA; then
  MODEL_NAME_="${MODEL_NAME_}-HEALTH"
fi

if $USE_MIMICIV; then
  MODEL_NAME_="${MODEL_NAME_}-MIMICIV"
fi

if $USE_SYNTHETIC; then
  MODEL_NAME_="${MODEL_NAME_}-SYNTHETIC"
fi

export MODEL_NAME=$MODEL_NAME_

EXP_NAME="$MODEL_NAME-$NUM_FEATURES"

CUDA_VISIBLE_DEVICES=0 python ehrpfn/training/train.py \
  --config-dir /home/mpham/workspace/source/ehrpfn/configs/training \
  --config-name nanotabpfn \
  folder="${ROOT}/outputs/all/training/pretrain" \
  exp_name=$EXP_NAME \
  model.max_num_features=$NUM_FEATURES \
  training.batch_size=64 \
  training.run_pretrain=true \
  training.run_finetune=false \
  data.use_general_openml=$USE_GENERAL_DATA \
  data.use_health_openml=$USE_HEALTH_DATA \
  data.use_mimiciv=$USE_MIMICIV \
  data.use_synthetic=$USE_SYNTHETIC \
  training.num_epochs=$NUM_EPOCHS \
  training.resume=$RESUME \
  training.pretrained=$PRETRAINED \
  training.num_model_updates=$NUM_MODEL_UPDATES \
  logging.eval_every=$EVAL_EVERY \
  training.eval_only=$EVAL_ONLY \
  model.freeze_transformer=$FREEZE_TRANSFORMER

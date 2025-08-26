export LOGGING_DIR="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/logs"
export NCCL_DEBUG=INFO
export NCCL_TIMEOUT=1800
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
export CUDA_LAUNCH_BLOCKING=1

FREEZE_TRANSFORMER=false
USE_GENERAL_DATA=false
USE_HEALTH_DATA=false
USE_MIMICIV=true
USE_SYNTHETIC=true
NUM_FEATURES=500
NUM_MODEL_UPDATES=512
EVAL_EVERY=1
NUM_EPOCHS=512
EVAL_ONLY=false
RESUME=null #/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-MIMICIV-100
PRETRAINED=null #/home/mpham/workspace/source/ehrpfn/outputs/pretraining/TabDPT-100/tabdpt1_1.pth

MODEL_NAME=TabDPT

if $USE_SYNTHETIC || $USE_HEALTH_DATA || $USE_MIMICIV; then
  MODEL_NAME=EHRPFN
fi

if $USE_GENERAL_DATA; then
  MODEL_NAME="${MODEL_NAME}-GENERAL"
fi

if $USE_HEALTH_DATA; then
  MODEL_NAME="${MODEL_NAME}-HEALTH"
fi

if $USE_MIMICIV; then
  MODEL_NAME="${MODEL_NAME}-MIMICIV"
fi

if $USE_SYNTHETIC; then
  MODEL_NAME="${MODEL_NAME}-SYNTHETIC"
fi

EXP_NAME="$MODEL_NAME-$NUM_FEATURES"

CUDA_VISIBLE_DEVICES=0 python ehrpfn/train.py \
  exp_name=$EXP_NAME \
  model.max_num_features=$NUM_FEATURES \
  training.batch_size=64 \
  data.use_general_data=$USE_GENERAL_DATA \
  data.use_health_data=$USE_HEALTH_DATA \
  data.use_mimiciv=$USE_MIMICIV \
  data.use_synthetic=$USE_SYNTHETIC \
  training.num_epochs=$NUM_EPOCHS \
  training.resume=$RESUME \
  training.pretrained=$PRETRAINED \
  training.num_model_updates=$NUM_MODEL_UPDATES \
  logging.eval_every=$EVAL_EVERY \
  training.eval_only=$EVAL_ONLY \
  model.freeze_transformer=$FREEZE_TRANSFORMER


# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nnodes=1 \
#   --nproc_per_node=1 \
#   --rdzv_endpoint=localhost:29500 ehrpfn/train.py \
#   env.gpus="[0]" \
#   exp_name=$EXP_NAME \
#   model.max_num_features=$NUM_FEATURES \
#   training.batch_size=64 \
#   data.use_general_data=$USE_GENERAL_DATA \
#   data.use_health_data=$USE_HEALTH_DATA \
#   data.use_mimiciv=$USE_MIMICIV \
#   data.use_synthetic=$USE_SYNTHETIC \
#   training.num_epochs=2048 \
#   training.resume=$RESUME \
#   training.pretrained=$PRETRAINED \
#   training.num_model_updates=$NUM_MODEL_UPDATES \
#   logging.eval_every=$EVAL_EVERY \
#   training.eval_only=false


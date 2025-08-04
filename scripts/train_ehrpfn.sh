export LOGGING_DIR="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/logs"
# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nproc_per_node=1 \
#   --rdzv_endpoint=localhost:29500 train.py \
#   env.gpus="[0]" \
#   exp_name="tabdpt" \
#   model.max_num_features=100


USE_HEALTH_DATA=true
USE_MIMICIV=false
USE_SYNTHETIC=false
NUM_FEATURES=100
RESUME=null #/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-HEALTH-MIMICIV-SYNTHETIC-100
PRETRAINED=/home/mpham/workspace/source/ehrpfn/data/checkpoints/tabdpt1_1.pth

if [ -n "$PRETRAINED" ]; then
  MODEL_NAME=EHRPFN
else
  MODEL_NAME=TabDPT
fi

if $USE_SYNTHETIC || $USE_HEALTH_DATA || $USE_MIMICIV; then
  MODEL_NAME=EHRPFN
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
  data.use_health_data=$USE_HEALTH_DATA \
  data.use_mimiciv=$USE_MIMICIV \
  data.use_synthetic=$USE_SYNTHETIC \
  training.num_epochs=2048 \
  training.resume=$RESUME \
  training.pretrained=$PRETRAINED
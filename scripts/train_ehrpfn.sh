export LOGGING_DIR="/home/mpham/workspace/source/ehrpfn/outputs/tabpfn/logs"
# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nproc_per_node=1 \
#   --rdzv_endpoint=localhost:29500 train.py \
#   env.gpus="[0]" \
#   exp_name="tabdpt" \
#   model.max_num_features=100


USE_HEALTH_DATA=true
USE_MIMICIV=true
USE_SYNTHETIC=true
NUM_FEATURES=300
RESUME=/home/mpham/workspace/source/ehrpfn/outputs/pretraining/EHRPFN-HEALTH-MIMICIV-SYNTHETIC-300


PRETRAINED=null
if [ -n "$PRETRAINED" ]; then
  MODEL_NAME=EHRPFN
else
  MODEL_NAME=TabDPT
fi

if [ -n "$USE_SYNTHETIC" ] || [ -n "$USE_HEALTH_DATA" ] || [ -n "$USE_MIMICIV" ]; then
  MODEL_NAME=EHRPFN
fi

if [ -n "$USE_HEALTH_DATA" ]; then
  MODEL_NAME="${MODEL_NAME}-HEALTH"
fi

if [ -n "$USE_MIMICIV" ]; then
  MODEL_NAME="${MODEL_NAME}-MIMICIV"
fi

if [ -n "$USE_SYNTHETIC" ]; then
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
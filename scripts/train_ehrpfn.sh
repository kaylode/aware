# CUDA_VISIBLE_DEVICES=0 torchrun \
#   --nproc_per_node=1 \
#   --rdzv_endpoint=localhost:29500 train.py \
#   env.gpus="[0]" \
#   exp_name="tabdpt" \
#   model.max_num_features=100


CUDA_VISIBLE_DEVICES=0 python ehrpfn/train.py \
  exp_name="TabDPT-300" \
  model.max_num_features=300 \
  training.batch_size=128 \
  training.reset_policy=cnt \
  data.use_mimic_only=false \
  data.use_health_data=false \
  data.use_synthetic=false \
  model.emsize=768 \
  model.max_num_classes=10 \
  model.nhead=8 \
  model.nhid_factor=2 \
  model.nlayers=16 \
  training.num_epochs=2048
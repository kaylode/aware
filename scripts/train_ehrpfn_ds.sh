#!/bin/bash
NUM_NODES=1
NUM_GPUS=1

deepspeed --num_nodes=${NUM_NODES} \
          --num_gpus=${NUM_GPUS} \
          --bind_cores_to_rank ehrpfn/training_deepspeed/example.py --deepspeed $@ \
          exp_name="TabDPT-300" \
          model.max_num_features=300 \
          training.batch_size=128 \
          training.reset_policy=rm \
          data.use_mimic_only=false \
          data.use_health_data=false \
          data.use_synthetic=false \
          model.emsize=768 \
          model.max_num_classes=10 \
          model.nhead=8 \
          model.nhid_factor=2 \
          model.nlayers=16 \
          training.num_epochs=2048


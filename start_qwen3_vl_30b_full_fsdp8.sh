#!/usr/bin/bash
set -e

# A5
source /home/pkg/b061/ascend-toolkit/set_env.sh

# A3
# source /home/cann/CANN_8_5_1_B050/ascend-toolkit/set_env.sh

export MASTER_ADDR=localhost
export NPROC_PER_NODE=8

export PYTORCH_NPU_ALLOC_CONF='expandable_segments:True'
export MULTI_STREAM_MEMORY_REUSE=2
export TASK_QUEUE_ENABLE=2
export CPU_AFFINITY_CONF=1

# Create temporary config file
cat > ./training_config.yaml << 'EOF'
data:
  source_name: sharegpt4v_sft
  chat_template: qwen3vl
  data_type: conversation
  dataloader:
    type: native
    num_workers: 8
  datasets_type: iterable
  max_seq_len: 16384
  train_path: /home/g00878120/dataset/ShareGPT4V/sharegpt4v_instruct_gpt4-vision_cap100k_coco_abs.json
  train_size: 80000000
  mm_configs:
    image_max_pixels: 602112 # 28 * 28 * 768
    video_max_pixels: 602112 # 28 * 28 * 768
    video_total_pixels: 2889523 # max_seq_len * 28^2 * 0.9, dynamic per-frame budget
    max_frames: 16
    fps: 2.0
    use_audio_in_video: false
model:
  model_path: /home/g00878120/weights/Qwen3-VL-30B-A3B-Instruct
  ops_implementation:
    rms_norm_implementation: npu
    rotary_pos_emb_implementation: npu
    swiglu_mlp_implementation: eager
    attn_implementation: flash_attention_2
    moe_implementation: fused
    cross_entropy_loss_implementation: chunk_loss
    load_balancing_loss_implementation: eager
train:
  enable_full_determinism: true
  accelerator:
    ep_size: 1
    fsdp_config:
      fsdp_mode: fsdp2
      full_shard: true
    ulysses_size: 1
  checkpoint:
    manager: dcp
    output_dir: ./
    save_hf_weights: false
    save_steps: 999
  dyn_bsz_margin: 0
  global_batch_size: 8
  init_device: meta
  max_steps: 100
  micro_batch_size: 1
  optimizer:
    lr: 0.0003
    lr_decay_ratio: 1.0
    lr_decay_style: constant
    lr_warmup_ratio: 0.007
    max_grad_norm: 1.0
    weight_decay: 0.01
  profile:
    enable: true
    end_step: 6
    start_step: 5
    with_stack: false
    rank0_only: false
  rmpad: false
  rmpad_with_pos_ids: true
  wandb:
    enable: false
EOF

bash train.sh tasks/train_vlm.py ./training_config.yaml

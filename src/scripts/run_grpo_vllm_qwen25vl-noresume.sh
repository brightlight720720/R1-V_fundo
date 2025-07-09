#!/bin/bash


# The latest vllm==0.7.3 is required for this script: pip3 install vllm==0.7.3
# The latest transformers is required too, install by: pip install git+https://github.com/huggingface/transformers.git@a40f1ac602fe900281722254c52ce3773f28eb0e


export HF_HOME=/workspace/hf_cache

export DEBUG_MODE="true"
export LOG_PATH="./vllm_run.txt"

QWEN_PATH="/workspace/models/Qwen2.5-VL-3B-Instruct"
#QWEN_PATH="/workspace/outputs/grpo_run"
HF_DATASET="Leeyuyu/fundo_400"
OUTPUT_DIR="/workspace/outputs/grpo_run"
if [ ! -d "$OUTPUT_DIR" ]; then
 mkdir -p "$OUTPUT_DIR"
fi
RUN_NAME="grpo_qwen25vl_3b_nothink"
DS_CONFIG="/workspace/R1-V_fundo/src/r1-v/local_scripts/zero1_no_optimizer.json"

# NOTE: you are expected to use X + 1 cards for X training proc and 1 vLLM proc 
# e.g., the visible devices should be 0,1,2,3,4 for 5 cards, and  --nproc_per_node="4"

# GPU configuration for 2 GPUs: 
# - GPU 0 for training
# - GPU 1 for vLLM inference

export VLLM_QWEN_VL_DUMMY_H=256
export VLLM_QWEN_VL_DUMMY_W=256
export VLLM_QWEN_VL_DUMMY_FRAMES=4

CUDA_VISIBLE_DEVICES="0,1,2,3" torchrun \
    --nproc_per_node="3" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port="12345" \
    /workspace/R1-V_fundo/src/r1-v/src/open_r1/grpo_25_nothink.py \
    --use_vllm true \
    --output_dir ${OUTPUT_DIR} \
    --model_name_or_path ${QWEN_PATH} \
    --dataset_name ${HF_DATASET} \
    --push_to_hub true \
    --hub_model_id Leeyuyu/Qwen2.5-GRPO-fundo-nothink \
    --hub_strategy every_save \
    --max_prompt_length 4096 \
    --max_completion_length 512 \
    --learning_rate 2e-6 \
    --lr_scheduler_type cosine \
    --warmup_steps 40 \
    --weight_decay 0.00 \
    --logging_steps 1 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --max_steps 300 \
    --gradient_accumulation_steps 48 \
    --per_device_train_batch_size 2 \
    --bf16 true \
    --max_grad_norm 1.0 \
    --min_pixels 3136 \
    --max_pixels 501760 \
    --num_train_epochs 2 \
    --run_name ${RUN_NAME} \
    --save_steps 50 \
    --save_total_limit 1 \
    --save_strategy steps \
    --report_to wandb \
    --temperature 0.7 \
    --num_generations 4 \
    --vllm_device "cuda:3" \
    --vllm_gpu_memory_utilization 0.85 \
    --deepspeed ${DS_CONFIG} \
    2>&1 | tee "${OUTPUT_DIR}/training_log.txt"


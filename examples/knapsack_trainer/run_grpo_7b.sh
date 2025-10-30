#!/usr/bin/env bash
set -xeuo pipefail

################################################################################
# Configuration
################################################################################

PROJECT_NAME='GRPO-allocation'
EXP_NAME='a100_GRPO-Qwen2.5-Math-7B-DAPO_17K'

# Advantage Estimator Configuration
ADV_ESTIMATOR='grpo'
ADV_CLIP_VALUE=5.0

# KL Divergence Configuration
USE_KL_IN_REWARD=false
KL_COEF=0.0
USE_KL_LOSS=true
KL_LOSS_COEF=0.0

# PPO Clipping Configuration
CLIP_RATIO_LOW=0.2
CLIP_RATIO_HIGH=0.28
IMP_RATIO_CAP=2.0

# Rollout Configuration
ROLLOUT_ENGINE='vllm'
GPU_MEMORY_UTILIZATION=0.85

# Sequence Length Configuration
MAX_PROMPT_LENGTH=$((1024 * 2))
MAX_RESPONSE_LENGTH=$((1024 * 4))

# Training Configuration
LOSS_AGG_MODE='token-mean'
TOTAL_BSZ=2048
OFF_POLICY=8
N_RESP_PER_PROMPT=8
TRAIN_PROMPT_BSZ=$((TOTAL_BSZ / N_RESP_PER_PROMPT))
PPO_MINI_BSZ=$((TOTAL_BSZ / OFF_POLICY / N_RESP_PER_PROMPT))
MAX_DATA_SIZE=$((1024 * 20))

# Training Steps
TOTAL_EPOCHS=10000
TOTAL_TRAINING_STEPS=1000

# Distributed Training Configuration
NNODES="${NNODES:-1}"
NGPUS_PER_NODE="${NGPUS_PER_NODE:-8}"

# Path Configuration
RAY_DATA_HOME="${RAY_DATA_HOME:-/opt/tiger/verl}"
RAY_SAVE_HOME="${RAY_SAVE_HOME:-/opt/tiger/verl}"
RAY_MODEL_HOME="${RAY_MODEL_HOME:-/opt/tiger/verl}"

MODEL_PATH="${MODEL_PATH:-${RAY_MODEL_HOME}/Qwen2.5-Math-7B}"
CKPTS_DIR="${CKPTS_DIR:-${RAY_SAVE_HOME}/ckpts/${PROJECT_NAME}/${EXP_NAME}}"
TRAIN_FILE="${TRAIN_FILE:-${RAY_DATA_HOME}/DAPO-Math-17k/train.parquet}"

# Test files as comma-separated list
TEST_FILE="${TEST_FILE:-[${RAY_DATA_HOME}/deepscaler/aime.parquet,${RAY_DATA_HOME}/deepscaler/aime2025.parquet,${RAY_DATA_HOME}/deepscaler/amc_fixed.parquet,${RAY_DATA_HOME}/deepscaler/math_100.parquet,${RAY_DATA_HOME}/deepscaler/minerva_fixed_100.parquet,${RAY_DATA_HOME}/deepscaler/olympiad_bench_fixed_100.parquet,${RAY_DATA_HOME}/r1/gpqa_diamond_100.parquet]}"

# Sampling Configuration
TEMPERATURE=1.0
TOP_P=1.0
TOP_K=-1  # 0 for HF rollout, -1 for vLLM rollout
VAL_TEMPERATURE=0.6
VAL_TOP_P=1.0
VAL_N=16

# Adaptive Allocation Configuration
ADAPTIVE_ALLOCATION=false
EPOCH_ESTIMATION=true
ROBUST_EPOCH_ESTIMATION=true
REWARD_BUFFER_LENGTH=16
MIN_BUFFER_LENGTH=8

# Performance Configuration
SP_SIZE=1
USE_DYNAMIC_BSZ=true
ACTOR_PPO_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 4))
INFER_PPO_MAX_TOKEN_LEN=$(((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH) * 4))
OFFLOAD=false
GEN_TP=1
FSDP_SIZE=-1

mkdir -p "$CKPTS_DIR"

################################################################################
# Main Training Command
################################################################################

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${TEST_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.train_batch_size="${TRAIN_PROMPT_BSZ}" \
    +data.max_data_size="${MAX_DATA_SIZE}" \
    +data.epoch_estimation="${EPOCH_ESTIMATION}" \
    +data.robust_epoch_estimation="${ROBUST_EPOCH_ESTIMATION}" \
    +data.reward_buffer_length="${REWARD_BUFFER_LENGTH}" \
    +data.min_buffer_length="${MIN_BUFFER_LENGTH}" \
    actor_rollout_ref.rollout.n="${N_RESP_PER_PROMPT}" \
    algorithm.adv_estimator="${ADV_ESTIMATOR}" \
    algorithm.use_kl_in_reward="${USE_KL_IN_REWARD}" \
    algorithm.kl_ctrl.kl_coef="${KL_COEF}" \
    +algorithm.adv_clip_value="${ADV_CLIP_VALUE}" \
    actor_rollout_ref.actor.use_kl_loss="${USE_KL_LOSS}" \
    actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.clip_ratio_low="${CLIP_RATIO_LOW}" \
    actor_rollout_ref.actor.clip_ratio_high="${CLIP_RATIO_HIGH}" \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    +actor_rollout_ref.actor.imp_ratio_cap="${IMP_RATIO_CAP}" \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz="${USE_DYNAMIC_BSZ}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${INFER_PPO_MAX_TOKEN_LEN}" \
    actor_rollout_ref.model.path="${MODEL_PATH}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=0 \
    actor_rollout_ref.actor.optim.weight_decay=0.0 \
    actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BSZ}" \
    actor_rollout_ref.actor.fsdp_config.param_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${OFFLOAD}" \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode="${LOSS_AGG_MODE}" \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.rollout.name="${ROLLOUT_ENGINE}" \
    actor_rollout_ref.rollout.gpu_memory_utilization="${GPU_MEMORY_UTILIZATION}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size="${GEN_TP}" \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((MAX_PROMPT_LENGTH + MAX_RESPONSE_LENGTH)) \
    actor_rollout_ref.rollout.temperature="${TEMPERATURE}" \
    actor_rollout_ref.rollout.top_p="${TOP_P}" \
    actor_rollout_ref.rollout.top_k="${TOP_K}" \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE}" \
    actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P}" \
    actor_rollout_ref.rollout.val_kwargs.top_k="${TOP_K}" \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n="${VAL_N}" \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.ref.fsdp_config.param_offload="${OFFLOAD}" \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size="${SP_SIZE}" \
    actor_rollout_ref.actor.fsdp_config.fsdp_size="${FSDP_SIZE}" \
    +trainer.adaptive_allocation=${ADAPTIVE_ALLOCATION} \
    trainer.logger='[console,swanlab]' \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXP_NAME}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=10 \
    trainer.save_freq=20 \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.default_local_dir="${CKPTS_DIR}" \
    trainer.rollout_data_dir="${CKPTS_DIR}/rollout" \
    trainer.validation_data_dir="${CKPTS_DIR}/validation" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=50 \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.max_actor_ckpt_to_keep=1 
set -x
# online==================

export GPUS_PER_NODE=8
export NNODES=8

#=========================

simplerl_train_path=deepscaler_rl_data/simplelr_math_35/train.parquet
simplerl_test_path=deepscaler_rl_data/simplelr_math_35/test.parquet
deepscaler_train_path=deepscaler_rl_data/deepscaler/train.parquet
aime_test_path=deepscaler_rl_data/deepscaler/aime.parquet
DATA_ROOT_DIR="['$simplerl_train_path','$deepscaler_train_path']"
VAL_DATA_DIR="['$aime_test_path','$simplerl_test_path']"

CKPT_ROOT_DIR=/ckpt
PROJECT_NAME=srpo

EXPERIMENT_NAME=srpo_v6_test_adv
BASE_MODEL=/map-vepfs/models/Qwen2.5-Math-7B

export VLLM_ATTENTION_BACKEND=XFORMERS
# https://github.com/Unakar/Logic-RL/blob/main/main_grpo.sh
# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export HYDRA_FULL_ERROR=1
# 默认参数：kl_loss_coef=0.001  srpo_wrong_sample_save_rate=0.2
#  python -m verl.trainer.main_ppo
# ./verl/trainer/main_ppo.py
# actor_rollout_ref.actor.loss_agg_mode=token-mean/seq-mean-token-sum
python3 -u /map-vepfs/wanggeng/verl/verl/trainer/main_ppo.py \
    algorithm.adv_estimator=srpo \
    data.train_files=$DATA_ROOT_DIR \
    data.val_files=$VAL_DATA_DIR \
    data.train_batch_size=32 \
    data.val_batch_size=256 \
    data.max_prompt_length=1024 \
    data.max_response_length=3072 \
    data.truncation=right \
    actor_rollout_ref.model.path=$BASE_MODEL \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.fix_ppo_mini_batch_nums=1 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.000 \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    actor_rollout_ref.actor.cliprange_low=0.2 \
    actor_rollout_ref.actor.cliprange_high=0.3 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.first_n=4 \
    actor_rollout_ref.rollout.use_dynamic_sampling=True \
    actor_rollout_ref.rollout.srpo_correct_sample_max_num=1 \
    actor_rollout_ref.rollout.srpo_wrong_sample_max_num=1 \
    actor_rollout_ref.rollout.srpo_max_block_num=4 \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=16 \
    actor_rollout_ref.rollout.val_kwargs.temperature=1.0 \
    actor_rollout_ref.rollout.adv_no_use_std=True \
    actor_rollout_ref.rollout.separate_by_prob=True \
    actor_rollout_ref.rollout.second_reward_by_first_reward=False \
    actor_rollout_ref.rollout.sharpen=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=32 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.kl_ctrl.kl_coef=0.00 \
    trainer.critic_warmup=0 \
    trainer.logger=['console'] \
    trainer.project_name="${PROJECT_NAME}" \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.default_local_dir="${CKPT_ROOT_DIR}/${PROJECT_NAME}/${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node=$GPUS_PER_NODE \
    trainer.nnodes=$NNODES \
    trainer.save_freq=100 \
    trainer.test_freq=50 \
    trainer.total_epochs=10 $@

# trainer.n_gpus_per_node=$GPUS_PER_NODE \
#trainer.nnodes=$NNODES \

# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import ray
import numpy as np
from tqdm import tqdm
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProto
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.tracking import ValidationGenerationsLogger
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader
import verl.utils.torch_functional as verl_F
from verl.utils.model import compute_position_id_with_mask
import random
from collections import defaultdict

WorkerType = Type[Worker]
import logging
# 设置日志级别为 DEBUG
logging.basicConfig(level=logging.DEBUG)

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'
    SRPO = 'srpo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

        self._check_resource_available()

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]

    def _check_resource_available(self):
        """Check if the resource pool can be satisfied in this ray cluster."""
        node_available_resources = ray.state.available_resources_per_node()
        node_available_gpus = {node: node_info.get('GPU', 0) for node, node_info in node_available_resources.items()}

        # check total required gpus can be satisfied
        total_available_gpus = sum(node_available_gpus.values())
        total_required_gpus = sum(
            [num_gpus for process_on_nodes in self.resource_pool_spec.values() for num_gpus in process_on_nodes])
        if total_available_gpus < total_required_gpus:
            raise ValueError(
                f"Total available GPUs {total_available_gpus} is less than total desired GPUs {total_required_gpus}")

        # check each resource pool can be satisfied, O(#resource_pools * #nodes)
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            num_gpus, num_nodes = process_on_nodes[0], len(process_on_nodes)
            for node, available_gpus in node_available_gpus.items():
                if available_gpus >= num_gpus:
                    node_available_gpus[node] -= num_gpus
                    num_nodes -= 1
                    if num_nodes == 0:
                        break
            if num_nodes > 0:
                raise ValueError(
                    f"Resource pool {resource_pool_name}: {num_gpus}*{num_nodes} cannot be satisfied in this ray cluster"
                )


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1, adv_no_use_std=False, sharpen=False):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index, adv_no_use_std=adv_no_use_std)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    
    elif adv_estimator == AdvantageEstimator.SRPO:
        token_level_rewards = data.batch['token_level_rewards']
        # breakpoint()
        index = data.non_tensor_batch['uid'] # 一个问题的多个rollout index index = rollout_n * 1
        group_index = data.non_tensor_batch['index'] # 组的index，传进来  group_index = rollout_n * block_num,所以要聚类的话，必须是uuid值
        responses = data.batch['responses']  # 384条prompt
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_srpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index,
                                                                        group_index=group_index,
                                                                        rollout_n=num_repeat,
                                                                        adv_no_use_std=adv_no_use_std,
                                                                        sharpen=sharpen)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


def compute_throughout_metrics(batch, timing_raw, n_gpus):
    total_num_tokens = sum(batch.meta_info['global_token_num'])
    time = timing_raw["step"]
    # estimated_flops, promised_flops = flops_function.estimate_flops(num_tokens, time)
    # f'Actual TFLOPs/s/GPU​': estimated_flops/(n_gpus),
    # f'Theoretical TFLOPs/s/GPU​': promised_flops,
    return {
        f'total_num_tokens': total_num_tokens,
        f'time_per_step': time,
        f'Tokens/Sec/GPU': total_num_tokens / (time * n_gpus),
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'
        '''
        role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
}
        '''
        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls
        self.validation_generations_logger = ValidationGenerationsLogger()

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO,AdvantageEstimator.SRPO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        assert config.actor_rollout_ref.actor.loss_agg_mode in [
            "token-mean", "seq-mean-token-sum", "seq-mean-token-mean"
        ], f"Invalid loss_agg_mode: {config.actor_rollout_ref.actor.loss_agg_mode}"
        
        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                         tokenizer=self.tokenizer,
                                         processor=self.processor,
                                         prompt_key=self.config.data.prompt_key,
                                         image_key=self.config.data.get('image_key', 'images'),
                                         max_prompt_length=self.config.data.max_prompt_length,
                                         filter_prompts=True,
                                         return_raw_chat=self.config.data.get('return_raw_chat', False),
                                         truncation=self.config.data.truncation)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=self.config.data.train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)

        self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                       tokenizer=self.tokenizer,
                                       processor=self.processor,
                                       prompt_key=self.config.data.prompt_key,
                                       image_key=self.config.data.get('image_key', 'images'),
                                       max_prompt_length=self.config.data.max_prompt_length,
                                       filter_prompts=True,
                                       return_raw_chat=self.config.data.get('return_raw_chat', False),
                                       truncation='error')
        self.val_dataloader = StatefulDataLoader(
            dataset=self.val_dataset,
            # Validation datasets are sent to inference engines as a whole batch,
            # which will schedule the memory themselves.
            batch_size=len(self.val_dataset),
            num_workers=8,
            shuffle=False,
            drop_last=False,
            collate_fn=collate_fn)

        assert len(self.train_dataloader) >= 1
        assert len(
            self.val_dataloader
        ) == 1, "Validation dataloader must have a single batch, which inference engines will schedule the memory themselves."

        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_val_generations(self, inputs, outputs, scores):
        """Log a table of validation samples to the configured logger (wandb or swanlab)"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        import numpy as np

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(inputs, outputs, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Log to each configured logger
        self.validation_generations_logger.log(self.config.trainer.logger, samples, self.global_steps)

    def compute_pass_at_k(self, results: list[list[bool]], k: int):
        """
        Compute the average pass@k metric for a list of problem results.

        Args:
            results: A list of lists of booleans, where each sublist represents the success of samples for a problem.
            k: The number of samples to consider (k in pass@k).

        Returns:
            The average pass@k score across all problems.
        """

        if k < 1:
            raise ValueError("k must be at least 1")
        
        pass_rates = []
        for problem in results:
            n = len(problem)
            if n < k:
                raise ValueError(f"Each problem must have at least {k} samples, found {n}")
            
            correct = sum(problem)
            if correct == 0:
                pass_rates.append(0.0)
                continue
            
            # Calculate the probability of failing all k trials
            fail_prob = 1.0
            for i in range(k):
                fail_prob *= (n - correct - i) / (n - i)
            
            pass_rates.append(1 - fail_prob)
        
        return sum(pass_rates) / len(pass_rates)

    def _validate(self):
        reward_tensor_lst = []
        reward_extra_info_dict: Dict[str,
                                     list[list[float]]] = None  # the values are of shape (num_of_batch, batch_size)
        data_source_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_scores = []

        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            # repeat test batch
            test_batch = test_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.val_kwargs.n,
                                           interleave=True)

            # we only do validation on rule-based rm
            if self.config.reward_model.enable and test_batch[0].non_tensor_batch['reward_model']['style'] == 'model':
                return {}

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)

            test_gen_batch = test_batch.pop(['input_ids', 'attention_mask', 'position_ids'])
            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': self.config.actor_rollout_ref.rollout.val_kwargs.do_sample,
                'validate': True,
                'rollout_n': 1,
            }
            # print(f'test_gen_batch meta info: {test_gen_batch.meta_info}')

            # pad to be divisible by dp_size
            # print(f"[_validate] test_gen_batch: {test_gen_batch}")
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            # print(f"[_validate] test_gen_batch_padded: {test_gen_batch_padded}")
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # print(f"[_validate] test_output_gen_batch_padded: {test_output_gen_batch_padded}")

            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            # print(f"[_validate] test_output_gen_batch: {test_output_gen_batch_padded}")
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)

            test_batch = test_batch.union(test_output_gen_batch) 
            # evaluate using reward_function
            reward_result = self.val_reward_fn(test_batch)

            # Handle both scalar and dictionary returns
            if isinstance(reward_result, dict):
                reward_tensor = reward_result['reward_tensor']
                cur_data_source = test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])
                if 'extra_info' in reward_result:
                    if reward_extra_info_dict is None:
                        reward_extra_info_dict = {}
                        for key, extra_reward in reward_result['extra_info'].items():
                            for i, data_source in enumerate(cur_data_source):
                                composed_key = f'{key}_{data_source}'
                                if composed_key not in reward_extra_info_dict:
                                    reward_extra_info_dict[composed_key] = []
                                reward_extra_info_dict[composed_key].append(extra_reward[i])
            else:
                reward_tensor = reward_result
                cur_data_source = test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0])

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)

            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(cur_data_source)

        self._maybe_log_val_generations(inputs=sample_inputs, outputs=sample_outputs, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)

        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())
                    
        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            assert len(rewards) % self.config.actor_rollout_ref.rollout.val_kwargs.n == 0
            metric_dict[f'val/test_score/{data_source}'] = np.mean(rewards)
            print(f"""Calculating pass@k rate for {data_source} with k={self.config.actor_rollout_ref.rollout.val_kwargs.k}""")
            reward_per_test_sample = np.reshape(rewards, (-1, self.config.actor_rollout_ref.rollout.val_kwargs.n)) # [N, n_val]
            pass_at_k_rate = self.compute_pass_at_k(reward_per_test_sample, k=self.config.actor_rollout_ref.rollout.val_kwargs.k)
            print(f"[{data_source}]pass_at_k_rate:", pass_at_k_rate)
            metric_dict[f'val/test_score/{data_source}_pass@k'] = pass_at_k_rate

        if reward_extra_info_dict is not None:
            for key, extra_info_dict in reward_extra_info_dict.items():
                metric_dict[f'val/test_score_extra/{key}'] = np.mean(extra_info_dict)

        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # print(f"[_balance_batch] global_partition_lst: {global_partition_lst}")
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        # print(f"[_balance_batch] batch: {global_partition_lst}")
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def separate_responses_str(self, save_dict_list, separate_flag="\n\n"):
        def split_list(lst, block_num):
            n = len(lst)
            quotient, remainder = divmod(n, block_num)
            result = []
            start = 0
            for i in range(block_num):
                # 计算当前部分的长度
                part_length = quotient + (1 if i < remainder else 0)
                # 提取当前部分
                end = start + part_length
                tmp = lst[start:end]
                result.append(separate_flag.join(tmp))
                start = end
            return result

        separate_block_num_list = []
        for dic in save_dict_list:
            separated_response = dic["response"].split(separate_flag)
            if len(separated_response) > 1:
                separated_response = separated_response[:-1]  # 最后一个位置是answer
            if len(separated_response) > self.config.actor_rollout_ref.rollout.srpo_max_block_num:
                separated_response = split_list(separated_response, self.config.actor_rollout_ref.rollout.srpo_max_block_num)
                # for i in range(len(separated_response)):
                #     print(f"[separate_responses_str] separated_response {i}:{separated_response[i]}")
                # print(f"[separate_responses_str] separated_response:{len(separated_response)}")
                # raise
            separate_block_num_list.append(len(separated_response))
            prefix_sum_str = ""
            for sub_str in separated_response:
                prefix_sum_str += sub_str + separate_flag
                if "prefix_sum_str" not in dic:
                    dic["prefix_sum_str"] = []
                dic["prefix_sum_str"].append(prefix_sum_str) 
            # print(f"[separate_responses_str] idx: {dic['idx']} {'='*10}")
            # print(f"[separate_responses_str] prompt: {dic['prompt']}")
            # print(f"[separate_responses_str] separated_response: {dic['prefix_sum_str']}")
        return {
            "rollout/separate_block_num/mean":np.mean(separate_block_num_list),
            "rollout/separate_block_num/min":np.min(separate_block_num_list),
            "rollout/separate_block_num/max":np.max(separate_block_num_list)
        }  

    def separate_responses_str_by_prob(self, save_dict_list, batch):
        assert "trajectory_prob" in save_dict_list[0], f"{save_dict_list}"
        separate_point_dict = {}
        for i in range(self.config.actor_rollout_ref.rollout.srpo_max_block_num): separate_point_dict[f"rollout/separate_block_num_{i}"] = []
        
        for dic in save_dict_list:
            trajectory_prob = dic["trajectory_prob"]
            # print("[separate_responses_str_by_prob] raw trajectory_prob: ", trajectory_prob, trajectory_prob.shape)
            idx = dic["idx"]
            data_item = batch[idx]
            prompt_ids = data_item.batch['prompts']
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            response_ids = data_item.batch['responses']
            # print("[separate_responses_str_by_prob] response_ids: ", response_ids, response_ids.shape)
            valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            valid_trajectory_prob = trajectory_prob[:valid_response_length]
            # print("[separate_responses_str_by_prob] valid_trajectory_prob: ", valid_trajectory_prob, valid_trajectory_prob.shape)
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            response_str = self.tokenizer.decode(valid_response_ids)
            # assert len(valid_trajectory_prob) > self.config.actor_rollout_ref.rollout.srpo_max_block_num, \
            #     f"[separate_responses_str_by_prob] valid_trajectory_prob len:{len(valid_trajectory_prob)} \n prompt:{prompt_str} \nresponse:{response_str} is too short"
            if len(valid_trajectory_prob) > self.config.actor_rollout_ref.rollout.srpo_max_block_num:
                srpo_max_block_num = self.config.actor_rollout_ref.rollout.srpo_max_block_num
            else:
                srpo_max_block_num = 1
                print(f"[separate_responses_str_by_prob] valid_trajectory_prob len:{len(valid_trajectory_prob)} \n prompt:{prompt_str} \nresponse:{response_str} is too short")
            # print("[separate_responses_str_by_prob] final response_str: ", response_str.replace("\n",""))
            values, indices = torch.topk(valid_trajectory_prob, k=srpo_max_block_num, largest=False)
            sorted_indices = torch.sort(indices)[0]
            sorted_indices = torch.concat([torch.tensor([0], device=sorted_indices.device, dtype=sorted_indices.dtype), sorted_indices])
            # print("[separate_responses_str_by_prob] sorted_indices: ", sorted_indices, sorted_indices.shape)
            prefix_sum_str = ""
            for i in range(srpo_max_block_num):
                start_idx, end_idx = sorted_indices[i], sorted_indices[i+1]
                separate_point_dict[f"rollout/separate_block_num_{i}"].append(end_idx)
                sub_str_ids = valid_response_ids[start_idx:end_idx]
                sub_str = self.tokenizer.decode(sub_str_ids)
                prefix_sum_str += sub_str
                if "prefix_sum_str" not in dic:
                    dic["prefix_sum_str"] = []
                dic["prefix_sum_str"].append(prefix_sum_str) 
                # print(f"[separate_responses_str_by_prob] sub_str {i}: ", sub_str.replace("\n",""))
            # print(f"[separate_responses_str_by_prob] sub_str last: ", self.tokenizer.decode(valid_response_ids[end_idx:]).replace("\n",""))
            # print(f"[separate_responses_str_by_prob] prefix_sum_str dic: ", dic["prefix_sum_str"]) 
        for i in range(self.config.actor_rollout_ref.rollout.srpo_max_block_num): 
            separate_point_dict[f"rollout/separate_block_num_{i}"] = np.mean(separate_point_dict[f"rollout/separate_block_num_{i}"])
        # print(f"[separate_responses_str_by_prob] separate_point_dict: ", separate_point_dict) 
        return separate_point_dict        

    def create_new_batch_data(self, save_dict_list, batch, separate_flag="\n\n"):
        # new_batch = batch.pop(
        #             batch_keys=['input_ids', 'attention_mask', 'position_ids'],
        #         )
        # new_batch.non_tensor_batch['raw_prompt_ids'] = raw_prompt_ids
        # print(f"[create_new_batch_data] new_batch: {new_batch}")

        def cal_second_reward_rate(first_rollout_right_rate):
            if first_rollout_right_rate is None:
                return 1.0
            rollout_n = self.config.actor_rollout_ref.rollout.n
            assert rollout_n >=4, f"[cal_second_rollout_num] rollout n must bigger than 4"
            first_rollout_right_num = int(first_rollout_right_rate*rollout_n)
            res = -1
            if first_rollout_right_num ==0 or first_rollout_right_num==1:
                res = 2
            elif first_rollout_right_num==rollout_n or first_rollout_right_num==rollout_n-1:
                res = 1
            elif first_rollout_right_num == rollout_n //2:
                res = 1.5
            elif first_rollout_right_num < rollout_n //2:
                res = 1.5
            elif first_rollout_right_num > rollout_n // 2:
                res = 1
            else:
                raise NameError(f"cal_second_rollout_num error, first_rollout_right_num={first_rollout_right_num} N={rollout_n}")
            return float(res)
        
        all_dict = {
            "input_ids": [],
            "attention_mask": [],
            "position_ids": [],
            "raw_prompt_ids": [],
            'data_source': [],
            'ability': [],
            'reward_model': [],
            'extra_info': [],
            'index': [],
            'first_rollout_reward':[],
            'second_reward_rate': []
        }
        for dic in save_dict_list:
            idx = dic["idx"]
            data_item = batch[idx]
            prompt_length = data_item.batch['prompts'].shape[-1]
            valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
            # 左pad
            valid_prompt_ids = data_item.batch['prompts'][-valid_prompt_length:]
            # 使用tokenizer decode valid_prompt_ids
            prompt_str = self.tokenizer.decode(valid_prompt_ids)
            prefix_sum_str = dic["prefix_sum_str"]
            assert len(prefix_sum_str) <= self.config.actor_rollout_ref.rollout.srpo_max_block_num, f"prefix_sum_str: {len(prefix_sum_str)}" 
            # 原始prompt也重新rollout一次
            prefix_sum_str = [""] + prefix_sum_str
            second_reward_rate = cal_second_reward_rate(dic['first_rollout_right_rate'])
            # print(f"[create_new_batch_data] first_rollout_right_rate:{dic['first_rollout_right_rate']} second_reward_rate:{second_reward_rate}")
            # if len(prefix_sum_str) == self.config.actor_rollout_ref.rollout.srpo_max_block_num:
            #     print(f"[create_new_batch_data] prefix_sum_str: {prefix_sum_str}")
            #     raise 
            for sub_idx, sub_str in enumerate(prefix_sum_str):
                new_prompt_str = prompt_str + sub_str
                # print(f"[new_prompt_str]: {new_prompt_str}")
                # print(f"[create_new_batch_data] prompt_str {idx}  {sub_idx}: {new_prompt_str}") 
                # pad了
                input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(prompt=new_prompt_str,
                                                                         tokenizer=self.tokenizer,
                                                                         # 输入里有部分respones的内容
                                                                         max_length=self.config.data.max_prompt_length+self.config.data.max_response_length,
                                                                         pad_token_id=self.tokenizer.pad_token_id,
                                                                         left_pad=True,
                                                                         truncation=self.config.data.truncation)
                position_ids = compute_position_id_with_mask(attention_mask)
                all_dict['input_ids'].append(input_ids[0])
                all_dict['attention_mask'].append(attention_mask[0])
                all_dict['position_ids'].append(position_ids[0])
                all_dict['raw_prompt_ids'].append(list(self.tokenizer.encode(new_prompt_str, add_special_tokens=False)))
                all_dict['data_source'].append(dic['data_source'])
                all_dict['ability'].append(dic['ability'])
                all_dict['reward_model'].append(dic['reward_model'])
                all_dict['extra_info'].append(dic['extra_info'])
                all_dict['index'].append(idx) 
                all_dict['first_rollout_reward'].append(dic['first_rollout_reward'])
                all_dict['second_reward_rate'].append(second_reward_rate)
        # 必须平均分配
        worker_num = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
        need_num = (len(all_dict['input_ids'])//worker_num)*worker_num
        all_dict['input_ids'] = torch.stack(all_dict['input_ids'],dim=0)[:need_num]
        all_dict['attention_mask'] = torch.stack(all_dict['attention_mask'],dim=0)[:need_num]
        all_dict['position_ids'] = torch.stack(all_dict['position_ids'],dim=0)[:need_num]
        all_dict['raw_prompt_ids'] = np.array(all_dict['raw_prompt_ids'],  dtype=object)[:need_num]
        all_dict['data_source'] = np.array(all_dict['data_source'],  dtype=object)[:need_num]
        all_dict['ability'] = np.array(all_dict['ability'],  dtype=object)[:need_num]
        all_dict['reward_model'] = np.array(all_dict['reward_model'],  dtype=object)[:need_num]
        all_dict['extra_info'] = np.array(all_dict['extra_info'],  dtype=object)[:need_num]
        all_dict['index'] = np.array(all_dict['index'],  dtype=object)[:need_num]
        all_dict['first_rollout_reward'] = np.array(all_dict['first_rollout_reward'], dtype=object)[:need_num]
        all_dict['second_reward_rate'] = np.array(all_dict['second_reward_rate'], dtype=object)[:need_num]
        new_batch = DataProto.from_single_dict(all_dict)
        del batch
        return new_batch


    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None
        accum_batch = None
        num_gen_batches = 0
        timing_raw = {}

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # print(f"[fit] batch: {batch}")
                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                # raw_prompt_ids = np.repeat(raw_prompt_ids, self.config.actor_rollout_ref.rollout.n, axis=0)
                is_last_step = self.global_steps >= self.total_training_steps
                # print(f"[fit] gen_batch: {gen_batch}") 
                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch.meta_info['rollout_n'] = self.config.actor_rollout_ref.rollout.first_n
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    # print(f"[fit] {self.actor_rollout_wg.__class__.__name__}")
                    # print(f"[fit] gen_batch_output: {gen_batch_output}")
                    # print(f"[fit] batch: {batch}")

                    
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.first_n, interleave=True)
                    batch = batch.union(gen_batch_output)

                    if self.config.actor_rollout_ref.rollout.first_roll_use_dynamic_sampling:
                        raise NotImplementedError
                        # 丢掉group全0/1的数据---
                        first_rollout_data_reward_tensor = self.reward_fn(batch)
                        first_rollout_data_reward_tensor = first_rollout_data_reward_tensor.sum(dim=-1)
                        # reward只能是1/0
                        assert torch.all(torch.logical_or(first_rollout_data_reward_tensor == 1, first_rollout_data_reward_tensor == 0)), \
                                f"reward only 1 or 0 {first_rollout_data_reward_tensor}"
                        # print(f"[fit] seconde_rollout_data_reward_tensor 1: {second_rollout_data_reward_tensor} {second_rollout_data_reward_tensor.shape}")
                        first_rollout_data_reward_tensor = first_rollout_data_reward_tensor.reshape([-1,  self.config.actor_rollout_ref.rollout.n])
                        first_rollout_data_reward_count = first_rollout_data_reward_tensor.sum(-1)
                        # print(f"[fit] second_rollout_data_reward_count 1: {second_rollout_data_reward_count} {second_rollout_data_reward_count.shape}")
                        # 哪个group非全对/错
                        need_idx = (first_rollout_data_reward_count!=self.config.actor_rollout_ref.rollout.n) & (first_rollout_data_reward_count!=0)
                        metrics.update({"rollout/first_rollout_all_right_group_num": 
                                        torch.sum(first_rollout_data_reward_count==self.config.actor_rollout_ref.rollout.n).detach().item()})
                        metrics.update({"rollout/first_rollout_all_wrong_group_num": torch.sum(first_rollout_data_reward_count==0).detach().item()})
                        need_idx = need_idx.repeat_interleave(self.config.actor_rollout_ref.rollout.n, dim=0)
                        worker_num = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                        need_num = (torch.sum(need_idx)//worker_num)*worker_num
                        # print(f"[fit] need_idx: {need_idx} {need_idx.shape}")
                        metrics.update({"rollout/first_rollout_sample_num_before_dynamic_choice": len(batch)})
                        # 有可能全对或者全错导致任务挂掉
                        if need_num >= worker_num:
                            batch = batch.gather(need_idx)
                        else:
                            print(f"[fit] first_roll_use_dynamic_sampling need num: {need_num}")
                            need_num = (len(batch)//worker_num)*worker_num
                        assert len(batch) >= worker_num, f"each worker need at least one sample. worker num: {worker_num} samples: {len(batch)}"
                        # print(f"[fit] after gather: {len(batch)} {worker_num} {torch.sum(need_idx)}")
                        need_idx2 = torch.zeros((len(batch),), dtype=torch.bool, device=need_idx.device)
                        need_idx2[:need_num] = True
                        batch = batch.gather(need_idx2)
                        metrics.update({"rollout/first_rollout_sample_num_after_dynamic_choice": len(batch)})
                        print(f"[fit] after first roll filter: {len(batch)}") 


                    SEPARATE_BY_PROB = self.config.actor_rollout_ref.rollout.separate_by_prob
                    # todo bigger clip
                    # 去除std
                    # 如果第二次rollout对了给更大的分
                    # 大bz大学习率
                    if SEPARATE_BY_PROB:
                        first_rollout_trajectory_prob = self.actor_rollout_wg.compute_log_prob(batch)
                    # print(f"[fit] batch 2: {batch}") 
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    first_rollout_response_info = _compute_response_info(batch)
                    metrics.update({
                        "rollout/first_rollout_length/mean":torch.mean(first_rollout_response_info["response_length"]).detach().item(),
                        "rollout/first_rollout_length/min":torch.min(first_rollout_response_info["response_length"]).detach().item(),
                        "rollout/first_rollout_length/max":torch.max(first_rollout_response_info["response_length"]).detach().item()
                    })
                    self._balance_batch(batch, metrics=metrics)
                    # print(f"[fit] batch after balance: {batch}") # print东西太多
                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()


                    # 第一次rollout，roll出正确答案数据
                    pre_rollout_data_reward_tensor = self.reward_fn(batch)
                    pre_rollout_data_reward_tensor = pre_rollout_data_reward_tensor.sum(dim=-1)
                    pre_rollout_right_rate = pre_rollout_data_reward_tensor.reshape(-1, self.config.actor_rollout_ref.rollout.n).sum(-1) / self.config.actor_rollout_ref.rollout.n
                    print(f"[fit] pre_rollout_right_rate: {pre_rollout_right_rate} {pre_rollout_right_rate.shape}")
                    pre_rollout_right_rate = pre_rollout_right_rate.repeat_interleave(self.config.actor_rollout_ref.rollout.n, dim=0)
                    print(f"[fit] pre_rollout_right_rate repeat: {pre_rollout_right_rate} {pre_rollout_right_rate.shape}")
                    # print(f"[fix] raw reward right num: {pre_rollout_data_reward_tensor.sum()}")
                    # 正确答案太少了，随机赋予20个为1
                    # todo: 对的只留一条
                    # todo：多次rollout，看能不能rollout对
                    # todo：参数：每个state rollout次数（非\n\n粒度，\n\n是最小粒度），总state rollout次数
                    # 冷启动模型setting：
                    # 数据集：openr1
                    # model size：7B
                    # 外部传参
                    # 追踪指标
                    # 应该在group粒度保留
                    # FILL_NUM = 20
                    # if pre_rollout_data_reward_tensor.sum() < FILL_NUM:
                    #     random_indices = torch.randperm(pre_rollout_data_reward_tensor.numel())[:FILL_NUM]
                    #     pre_rollout_data_reward_tensor[random_indices] = 1
                    # print(f"[fix] reward: {pre_rollout_data_reward_tensor} \n {pre_rollout_data_reward_tensor.shape}")
                    
                    # 抽带\n\n的答案
                    SAVE_FLAG = "\n\n"
                    assert len(pre_rollout_data_reward_tensor) == len(batch.batch['attention_mask']), \
                            f"{len(pre_rollout_data_reward_tensor)} != {len(batch.batch['attention_mask'])}"
                    
                    save_dict_list = []
                    uid_set = set()
                    correct_counter_dict = defaultdict(int)
                    wrong_counter_dict = defaultdict(int)
                    FALSE_SAVE_RATE = self.config.actor_rollout_ref.rollout.srpo_wrong_sample_save_rate
                    CORRECT_MAX_NUM = self.config.actor_rollout_ref.rollout.srpo_correct_sample_max_num
                    WRONG_MAX_NUM = self.config.actor_rollout_ref.rollout.srpo_wrong_sample_max_num
                    for i in range(len(batch)):
                        # 保存正确的/按概率保存错误的，每个sample只保存一个样本
                        if CORRECT_MAX_NUM <= 0 and WRONG_MAX_NUM<=0:
                            random_num = random.random()
                            if pre_rollout_data_reward_tensor[i] <=0 and random_num > FALSE_SAVE_RATE:
                                continue
                            uid = batch.non_tensor_batch['uid'][i]
                            if uid in uid_set:
                                continue
                            uid_set.add(uid)
                        # 正/负例最多保存CORRECT_MAX_NUM/WRONG_MAX_NUM个
                        else:
                            assert CORRECT_MAX_NUM > 0 or WRONG_MAX_NUM > 0
                            uid = batch.non_tensor_batch['uid'][i]
                            if pre_rollout_data_reward_tensor[i] > 0:
                                if correct_counter_dict[uid] < CORRECT_MAX_NUM:
                                    correct_counter_dict[uid] += 1
                                else:
                                    continue
                            else:
                                if wrong_counter_dict[uid] < WRONG_MAX_NUM:
                                    wrong_counter_dict[uid] += 1
                                else:
                                    continue
                        data_item = batch[i]  # DataProtoItem
                        prompt_ids = data_item.batch['prompts']
                        prompt_length = prompt_ids.shape[-1]
                        valid_prompt_length = data_item.batch['attention_mask'][:prompt_length].sum()
                        valid_prompt_ids = prompt_ids[-valid_prompt_length:]
                        response_ids = data_item.batch['responses']
                        valid_response_length = data_item.batch['attention_mask'][prompt_length:].sum()
                        valid_response_ids = response_ids[:valid_response_length]
                        prompt_str = self.tokenizer.decode(valid_prompt_ids)
                        response_str = self.tokenizer.decode(valid_response_ids)
                        if not SEPARATE_BY_PROB and SAVE_FLAG not in response_str:
                            continue
                        save_dict_list.append({
                            'prompt': prompt_str,
                            'response': response_str,
                            'idx': i,
                            'data_source': batch.non_tensor_batch['data_source'][i],
                            'ability': batch.non_tensor_batch['ability'][i],
                            'reward_model': batch.non_tensor_batch['reward_model'][i],
                            'extra_info': batch.non_tensor_batch['extra_info'][i],
                            'index': batch.non_tensor_batch['index'][i],
                            'trajectory_prob':first_rollout_trajectory_prob[i].batch["old_log_probs"] if SEPARATE_BY_PROB else None,
                            'first_rollout_reward': pre_rollout_data_reward_tensor[i] if self.config.actor_rollout_ref.rollout.second_reward_by_first_reward else None,
                            'first_rollout_right_rate':pre_rollout_right_rate[i] if self.config.actor_rollout_ref.rollout.second_reward_by_first_right_rate else None,
                        })
                        # print(f"[fit] idx: {i}, count:{response_str.count(SAVE_FLAG)}, response: {response_str}")
                    print(f"[fit] before separate_responses_str size: {len(save_dict_list)} right_num: {pre_rollout_data_reward_tensor.sum()}")
                    if SEPARATE_BY_PROB:
                        separate_block_dict = self.separate_responses_str_by_prob(save_dict_list, batch)
                    else:
                        separate_block_dict = self.separate_responses_str(save_dict_list, separate_flag=SAVE_FLAG)
                    metrics.update(separate_block_dict)
                    batch = self.create_new_batch_data(save_dict_list, batch, separate_flag=SAVE_FLAG) 
                    print(f"[fit] after separate_responses_str size: {len(batch.batch['input_ids'])}")

                    #第二次rollout======================================
                    # no multi_modal
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                    is_last_step = self.global_steps >= self.total_training_steps
                    with _timer('step', timing_raw):
                        # generate a batch
                        with _timer('gen', timing_raw):
                            gen_batch.meta_info['rollout_n'] = self.config.actor_rollout_ref.rollout.n
                            gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    #--------------------
                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                             dtype=object)
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    if self.config.actor_rollout_ref.rollout.use_dynamic_sampling:
                        # 丢掉group全0/1的数据---
                        second_rollout_data_reward_tensor = self.reward_fn(batch)
                        second_rollout_data_reward_tensor = second_rollout_data_reward_tensor.sum(dim=-1)
                        # reward只能是1/0: 开启second_reward_by_first_reward时候是2======
                        # assert torch.all(torch.logical_or(second_rollout_data_reward_tensor == 1, second_rollout_data_reward_tensor == 0)), \
                        #         f"reward only 1 or 0 {second_rollout_data_reward_tensor}"
                        # print(f"[fit] seconde_rollout_data_reward_tensor 1: {second_rollout_data_reward_tensor} {second_rollout_data_reward_tensor.shape}")
                        second_rollout_data_reward_tensor = second_rollout_data_reward_tensor.reshape([-1,  self.config.actor_rollout_ref.rollout.n])
                        second_rollout_data_reward_count = second_rollout_data_reward_tensor.sum(-1)
                        # print(f"[fit] second_rollout_data_reward_count 1: {second_rollout_data_reward_count} {second_rollout_data_reward_count.shape}")
                        # 哪个group非全对/错
                        need_idx = (second_rollout_data_reward_count!=self.config.actor_rollout_ref.rollout.n) & (second_rollout_data_reward_count!=0)
                        metrics.update({"rollout/second_rollout_all_right_group_num": 
                                        torch.sum(second_rollout_data_reward_count==self.config.actor_rollout_ref.rollout.n).detach().item()})
                        metrics.update({"rollout/second_rollout_all_wrong_group_num": torch.sum(second_rollout_data_reward_count==0).detach().item()})
                        need_idx = need_idx.repeat_interleave(self.config.actor_rollout_ref.rollout.n, dim=0)
                        worker_num = self.config.trainer.n_gpus_per_node * self.config.trainer.nnodes
                        # need_num = (torch.sum(need_idx)//worker_num)*worker_num
                        # print(f"[fit] need_idx: {need_idx} {need_idx.shape}")
                        metrics.update({"rollout/sample_num_before_dynamic_choice": len(batch)})
                        batch = batch.gather(need_idx)
                        # 后边有攒batch
                        # assert len(batch) >= worker_num, f"each worker need at least one sample. \
                        #     worker num: {worker_num} samples: {len(batch)}"
                        # print(f"[fit] after gather: {len(batch)} {worker_num} {torch.sum(need_idx)}")
                        # need_idx2 = torch.zeros((len(batch),), dtype=torch.bool, device=need_idx.device)
                        # need_idx2[:need_num] = True
                        # batch = batch.gather(need_idx2)
                        metrics.update({"rollout/sample_num_after_dynamic_choice": len(batch)})
                        print(f"[fit] after second roll filter: {len(batch)}") 
                    #--------------------
                    accum_batch = batch if accum_batch is None else DataProto.concat([accum_batch, batch])
                    num_prompt_in_batch = len(np.unique(accum_batch.non_tensor_batch['uid']))
                    prompt_bsz = int(self.config.data.train_batch_size * self.config.algorithm.save_batch_threshold)
                    print(f"[fit] num_prompt_in_batch: {num_prompt_in_batch} prompt_bsz: {prompt_bsz}")
                    if num_prompt_in_batch < prompt_bsz:
                        print(f"[fit] num_prompt_in_batch: {num_prompt_in_batch=} < {prompt_bsz=}")
                        max_num_gen_batches = self.config.algorithm.max_num_gen_batches
                        if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                            print(f"{num_gen_batches=}. Keep generating...")
                            continue
                        else:
                            raise ValueError(f"{num_gen_batches=} >= {max_num_gen_batches=}." + " Generated too many. Please check if your data are too difficult." + " You could also try set max_num_gen_batches=0 to enable endless trials.")
                    else:
                        batch = accum_batch
                        need_num = (len(batch)//worker_num)*worker_num
                        need_idx2 = torch.zeros((len(batch),), dtype=torch.bool, device=need_idx.device)
                        need_idx2[:need_num] = True
                        batch = batch.gather(need_idx2)
                        metrics.update({"rollout/accum_batch_size": len(batch)})
                        metrics.update({"rollout/num_prompt_in_batch": num_prompt_in_batch})
                    
                    self._balance_batch(batch, metrics=metrics)
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()

                    #====================================================
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        # 返回了advantages和returns
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n,
                                                  adv_no_use_std=self.config.actor_rollout_ref.rollout.adv_no_use_std,
                                                  sharpen=self.config.actor_rollout_ref.rollout.sharpen)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            # print(f"[update_actor] outter bz: {len(batch)}")
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics) 
                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                config = self.config
                n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
                # Implement actual tflpo and theoretical tflpo
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))
                timing_raw = {}
                metrics["train/num_gen_batches"] = num_gen_batches
                accum_batch = None
                num_gen_batches = 0
                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return

                self.global_steps += 1

    def fit_vanilla(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf
        logger = Tracking(project_name=self.config.trainer.project_name,
                        experiment_name=self.config.trainer.experiment_name,
                        default_backend=self.config.trainer.logger,
                        config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True):
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return
        progress_bar = tqdm(total=self.total_training_steps, initial=self.global_steps, desc="Training Progress")
        
        # we start from step 1
        self.global_steps += 1
        last_val_metrics = None

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                batch: DataProto = DataProto.from_single_dict(batch_dict)
                # print(f"[fit] batch: {batch}")
                # pop those keys for generation
                if 'multi_modal_inputs' in batch.non_tensor_batch.keys():
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data', 'multi_modal_inputs'],
                    )
                else:
                    gen_batch = batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )
                # raw_prompt_ids = np.repeat(raw_prompt_ids, self.config.actor_rollout_ref.rollout.n, axis=0)
                is_last_step = self.global_steps >= self.total_training_steps
                # print(f"[fit] gen_batch: {gen_batch}") 
                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)
                    # print(f"[fit] {self.actor_rollout_wg.__class__.__name__}")
                    # print(f"[fit] gen_batch_output: {gen_batch_output}")
                    # print(f"[fit] batch: {batch}")

                    
                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            batch = batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(batch.batch))],
                                                            dtype=object)
                    # repeat to align with repeated responses in rollout
                    batch = batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    batch = batch.union(gen_batch_output)
                    # print(f"[fit] batch 2: {batch}") 
                    # balance the number of valid tokens on each dp rank.
                    # Note that this breaks the order of data inside the batch.
                    # Please take care when you implement group based adv computation such as GRPO and rloo
                    self._balance_batch(batch, metrics=metrics)
                    # print(f"[fit] batch after balance: {batch}") # print东西太多
                    # compute global_valid tokens
                    batch.meta_info['global_token_num'] = torch.sum(batch.batch['attention_mask'], dim=-1).tolist()
                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # we first compute reward model score
                            reward_tensor = self.rm_wg.compute_rm_score(batch)
                            batch = batch.union(reward_tensor)

                        # we combine with rule-based rm
                        reward_tensor = self.reward_fn(batch)
                        batch.batch['token_level_scores'] = reward_tensor

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                kl_ctrl=self.kl_ctrl,
                                                                kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                adv_estimator=self.config.algorithm.adv_estimator,
                                                gamma=self.config.algorithm.gamma,
                                                lam=self.config.algorithm.lam,
                                                num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:
                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        (is_last_step or  self.global_steps % self.config.trainer.test_freq == 0):
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                            if is_last_step:
                                last_val_metrics = val_metrics
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and ( is_last_step or \
                            self.global_steps % self.config.trainer.save_freq == 0):
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                config = self.config
                n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes
                # Implement actual tflpo and theoretical tflpo
                metrics.update(compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus))

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                if is_last_step:
                    pprint(f'Final validation metrics: {last_val_metrics}')
                    return
                
                progress_bar.update(1)
                self.global_steps += 1
from typing import Callable, Optional, Union, Any, List, Dict, Tuple

import torch
from datasets import Dataset, IterableDataset
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
)
from transformers.utils import is_peft_available
from trl import GRPOConfig

from verifiers.envs.environment import Environment
from verifiers.trainers.grpo_env_trainer import GRPOEnvTrainer, RewardFunc

if is_peft_available():
    from peft import PeftConfig  # type: ignore

if is_wandb_available():
    import wandb

class MSGRPOEnvTrainer(GRPOEnvTrainer):
    """
    Multi-Step GRPO Environment Trainer that calculates separate advantages for 
    step rewards and outcome rewards. Tokens before a '<result>' tag get
    both advantages, tokens after only get outcome advantage.
    """
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            step_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            outcome_reward_funcs: Union[RewardFunc, List[RewardFunc]],
            step_reward_weights: Optional[List[float]] = None,
            outcome_reward_weights: Optional[List[float]] = None,
            # optional, default is 0.5
            step_advantage_coe: Optional[float] = 0.5,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[List[TrainerCallback]] = None,
            optimizers: Tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):
        # Convert single reward functions to lists
        if callable(step_reward_funcs):
            step_reward_funcs = [step_reward_funcs]
        if callable(outcome_reward_funcs):
            outcome_reward_funcs = [outcome_reward_funcs]
            
        # Create combined reward funcs for parent class
        self.step_reward_funcs = step_reward_funcs
        self.outcome_reward_funcs = outcome_reward_funcs
        combined_reward_funcs = step_reward_funcs + outcome_reward_funcs
        
        # Set up reward weights
        self.num_step_funcs = len(step_reward_funcs)
        self.num_outcome_funcs = len(outcome_reward_funcs)
        
        # all ones
        self.step_reward_weights = torch.ones(self.num_step_funcs)
        self.outcome_reward_weights = torch.ones(self.num_outcome_funcs)
        
        # Step advantage coefficient
        self.step_advantage_coe = step_advantage_coe
        
        # Combined weights for parent class (these won't be used directly in our implementation)
        #combined_weights = step_reward_weights + outcome_reward_weights
        
        super().__init__(
            model=model,
            env=env,
            reward_funcs=combined_reward_funcs,
            #reward_weights=combined_weights,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )

    def _generate_and_score_completions(
         self, inputs: Dict[str, Union[torch.Tensor, Any]]   
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]  # type: ignore 
        
        # Generate completions using the environment
        # This part is the same as the parent class
        prompt_inputs, prompt_ids, prompt_mask = self._prepare_prompt_inputs(inputs)
        
        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Generate completions using the environment
        all_prompts, completion_ids, completion_messages, completion_mask = self._generate_completions(prompts)
        
        # Prepare model inputs
        prompt_completion_ids, attention_mask, logits_to_keep = self._prepare_model_inputs(
            prompt_ids, prompt_mask, completion_ids, completion_mask
        )
        
        # Compute logps
        old_per_token_logps, ref_per_token_logps = self._compute_logps(
            prompt_completion_ids, attention_mask, logits_to_keep
        )
        
        # Calculate step rewards and outcome rewards separately
        rewards_step = self._calculate_rewards(
            prompts, completion_messages, self.step_reward_funcs, inputs
        )
        
        rewards_outcome = self._calculate_rewards(
            prompts, completion_messages, self.outcome_reward_funcs, inputs
        )
        
        # Apply weights and sum
        step_rewards = (rewards_step * self.step_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        outcome_rewards = (rewards_outcome * self.outcome_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        
        # Compute normalized advantages
        step_advantages = self._compute_normalized_advantages(step_rewards)
        outcome_advantages = self._compute_normalized_advantages(outcome_rewards)
        
        # Find the positions of <result> tags in each completion
        result_positions = self._find_result_positions(completion_ids, completion_messages)
        
        # Apply the combined advantages based on <result> tag positions
        # If there's a <result>, tokens before get step+outcome advantage, after get only outcome
        # If no <result>, all tokens get only outcome advantage
        combined_advantages = self._combine_advantages(
            completion_mask, step_advantages, outcome_advantages, result_positions
        )
        
        # Log metrics
        self._log_metrics(
            prompts, completion_messages, completion_mask, 
            rewards_step, rewards_outcome, step_rewards, outcome_rewards
        )
        
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": combined_advantages,
        }
    
    def _prepare_prompt_inputs(self, inputs):
        """Prepare the prompt inputs for the model."""
        from trl.data_utils import maybe_apply_chat_template
        from transformers import Trainer
        
        prompts = [x["prompt"] for x in inputs]  # type: ignore
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs]  # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
        )
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            
        return prompt_inputs, prompt_ids, prompt_mask
    
    def _generate_completions(self, prompts):
        """Generate completions using the environment and broadcast the results."""
        from accelerate.utils import broadcast_object_list, gather_object
        
        all_prompts = gather_object(prompts)
        if self.accelerator.is_main_process:
            env_result = self.env.generate(
                prompts=all_prompts,
                llm=self.llm,
                sampling_params=self.sampling_params,
            )
            completion_ids = env_result['ids']
            completion_messages = env_result['messages']
            completion_mask = env_result['mask']
        else:
            completion_ids = [None] * len(all_prompts)
            completion_messages = [None] * len(all_prompts)
            completion_mask = [None] * len(all_prompts)

        completion_ids = broadcast_object_list(completion_ids, from_process=0)
        completion_messages = broadcast_object_list(completion_messages, from_process=0)
        completion_mask = broadcast_object_list(completion_mask, from_process=0)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )

        completion_ids = completion_ids[process_slice]
        completion_messages = completion_messages[process_slice]
        completion_mask = completion_mask[process_slice]
        
        # Convert to tensors and pad
        device = self.accelerator.device
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = self._pad(completion_ids, padding_value=self.processing_class.pad_token_id)

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = self._pad(completion_mask, padding_value=0)
        
        return all_prompts, completion_ids, completion_messages, completion_mask
    
    def _pad(self, tensors, padding_value=0):
        """Pad a list of tensors to the same length."""
        from trl.trainer.utils import pad
        return pad(tensors, padding_value=padding_value)
    
    def _prepare_model_inputs(self, prompt_ids, prompt_mask, completion_ids, completion_mask):
        """Prepare the model inputs for logit computation."""
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        return prompt_completion_ids, attention_mask, logits_to_keep
    
    def _compute_logps(self, prompt_completion_ids, attention_mask, logits_to_keep):
        """Compute log probabilities using the model and reference model."""
        with torch.no_grad():
            # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's
            # computation here, and use per_token_logps.detach() instead.
            if self.num_iterations > 1:
                old_per_token_logps = self._get_per_token_logps(
                    self.model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                old_per_token_logps = None

            if self.beta == 0.0:
                ref_per_token_logps = None
            elif self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
                    
        return old_per_token_logps, ref_per_token_logps
    
    def _calculate_rewards(self, prompts, completions, reward_funcs, inputs):
        """Calculate rewards for a set of reward functions."""
        device = self.accelerator.device
        rewards_per_func = torch.zeros(len(prompts), len(reward_funcs), device=device)
        
        for i, reward_func in enumerate(reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]]  # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys}  # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs)  # type: ignore
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)
            
        from accelerate.utils import gather
        return gather(rewards_per_func)
    
    def _compute_normalized_advantages(self, rewards):
        """Compute normalized advantages from rewards."""
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * len(rewards) // self.accelerator.num_processes,
            (self.accelerator.process_index + 1) * len(rewards) // self.accelerator.num_processes,
        )
        return advantages[process_slice]
    
    def _find_result_positions(self, completion_ids, completion_messages):
        """
        在环境响应中查找<result>标签并确定分割点。
        
        如果环境响应中包含<result>标签，则返回该环境响应的开始位置，
        这样该位置之前的所有token都会获得step_advantage + outcome_advantage，
        之后的token只获得outcome_advantage。
        
        如果没有找到结果标签，则返回-1，表示整个序列只使用outcome_advantage。
        """
        device = self.accelerator.device
        result_positions = []
        
        for i, completion in enumerate(completion_messages):
            ids = completion_ids[i]
            result_pos = -1
            
            # 处理对话历史形式的完成内容
            if isinstance(completion, list):
                # 寻找assistant消息后跟着user消息(env response)的模式
                for j, msg in enumerate(completion):
                    if msg.get('role') == 'assistant':
                        # 检查是否有后续的环境响应
                        if j + 1 < len(completion) and completion[j + 1].get('role') == 'user':
                            user_msg = completion[j + 1].get('content', '')
                            
                            # 检查环境响应中是否包含<result>标签
                            if '<result>' in user_msg:
                                # 计算环境响应开始的token位置
                                token_pos = 0
                                # 计算所有之前消息的token长度
                                for k in range(j + 1):
                                    token_pos += len(self.processing_class.encode(
                                        str(completion[k].get('content', ''))))
                                
                                # 将分割点设为环境响应的开始位置
                                result_pos = min(token_pos, len(ids) - 1)
                                break
            
            # 处理单个字符串形式的完成内容（兼容性保留）
            elif isinstance(completion, str):
                # raise error
                raise ValueError("Completion is a string, which is not supported.")
            
            result_positions.append(result_pos)
            
        return result_positions
    
    def _combine_advantages(self, completion_mask, step_advantages, outcome_advantages, result_positions):
        """
        Combine step and outcome advantages based on result positions.
        - If result_pos > 0: tokens before get step+outcome, after get only outcome
        - If result_pos = -1: all tokens get only outcome advantage
        
        The step_advantage_coe parameter controls the weight of step advantage.
        """
        device = self.accelerator.device
        batch_size, seq_len = completion_mask.shape
        combined_advantages = torch.zeros_like(completion_mask, dtype=torch.float32)
        
        for i in range(batch_size):
            result_pos = result_positions[i]
            if result_pos > 0:
                # Create a mask for tokens before the result tag
                before_result_mask = torch.zeros(seq_len, device=device)
                before_result_mask[:result_pos] = 1.0
                before_result_mask = before_result_mask * completion_mask[i]
                
                # Apply combined advantage before result, outcome advantage after
                # Use step_advantage_coe to control the weight of step advantage
                # 将标量扩展到序列长度
                outcome_advantage_expanded = outcome_advantages[i].item() * torch.ones_like(completion_mask[i], dtype=torch.float32)
                step_advantage_expanded = step_advantages[i].item() * torch.ones_like(before_result_mask, dtype=torch.float32)
                
                combined_advantages[i] = outcome_advantage_expanded + self.step_advantage_coe * step_advantage_expanded * before_result_mask
            else:
                # No result tag found, use only outcome advantage
                # 将标量扩展到序列长度
                outcome_advantage_expanded = outcome_advantages[i].item() * torch.ones_like(completion_mask[i], dtype=torch.float32)
                combined_advantages[i] = outcome_advantage_expanded
                
        return combined_advantages
    
    def _log_metrics(self, prompts, completions, completion_mask, 
                    rewards_step, rewards_outcome, step_rewards, outcome_rewards):
        """Log metrics for both step and outcome rewards."""
        mode = "eval" if self.control.should_evaluate else "train"

        # Log completion length
        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        self._metrics[mode]["completion_length"].append(completion_length)

        # Log individual reward functions
        for i, reward_func in enumerate(self.step_reward_funcs):
            reward_func_name = getattr(reward_func, "__name__", f"step_reward_{i}")
            self._metrics[mode][f"rewards/step/{reward_func_name}"].append(rewards_step.mean(0)[i].item())
            
        for i, reward_func in enumerate(self.outcome_reward_funcs):
            reward_func_name = getattr(reward_func, "__name__", f"outcome_reward_{i}")
            self._metrics[mode][f"rewards/outcome/{reward_func_name}"].append(rewards_outcome.mean(0)[i].item())

        # Log overall rewards
        self._metrics[mode]["reward/step"].append(step_rewards.mean().item())
        self._metrics[mode]["reward/outcome"].append(outcome_rewards.mean().item())
        
        # Log combined reward (step + outcome)
        combined_rewards = step_rewards + outcome_rewards
        self._metrics[mode]["reward"].append(combined_rewards.mean().item())
        
        # Log samples if needed
        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            self._log_completion_samples(prompts, completions, combined_rewards)
            
    def _log_completion_samples(self, prompts, completions, rewards):
        """Log completion samples to console and wandb if available."""
        from accelerate.utils import gather_object
        
        prompts_to_log = gather_object(prompts)
        completions_to_log = gather_object(completions)
        rewards_to_log = rewards.tolist()  # 直接在这里转换为Python列表

        if self.accelerator.is_main_process:
            if len(prompts_to_log) > 0:
                from trl.import_utils import is_rich_available
                
                if is_rich_available():
                    from verifiers.utils.logging_utils import print_prompt_completions_sample
                    
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                    
                if self.args.report_to and "wandb" in self.args.report_to and is_wandb_available() and wandb.run is not None:
                    import pandas as pd
                    
                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),  # 再次确保是Python列表
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)})

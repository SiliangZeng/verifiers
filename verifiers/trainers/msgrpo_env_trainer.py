from typing import Callable, Optional, Union, Any, List

from accelerate.utils import broadcast_object_list, gather, gather_object
from datasets import Dataset, IterableDataset
import torch
from torch import nn
from transformers import (
    PreTrainedModel,
    PreTrainedTokenizerBase,
    TrainerCallback,
    is_wandb_available,
    Trainer,
)
from transformers.utils import is_peft_available
from trl import GRPOTrainer, GRPOConfig
from trl.data_utils import apply_chat_template, maybe_apply_chat_template
from trl.import_utils import is_rich_available
from trl.trainer.utils import pad

from verifiers.envs.environment import Environment
from verifiers.utils.logging_utils import print_prompt_completions_sample

from .grpo_env_trainer import GRPOEnvTrainer

if is_peft_available():
    from peft import PeftConfig # type: ignore

if is_wandb_available():
    import wandb

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

class MSGRPOEnvTrainer(GRPOEnvTrainer):
    def __init__(
            self,
            model: Union[str, PreTrainedModel],
            env: Environment,
            reward_funcs: Union[RewardFunc, list[RewardFunc]],
            step_advantage_coef: float = 0.0,
            use_step_rewards: bool = False,
            args: Optional[GRPOConfig] = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            callbacks: Optional[list[TrainerCallback]] = None,
            optimizers: tuple[Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
            peft_config: Optional["PeftConfig"] = None,
            **kwargs,
    ):

        super().__init__(
            model=model,
            env=env,
            reward_funcs=reward_funcs,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
            peft_config=peft_config,
            **kwargs,
        )
        
        self.step_advantage_coef = step_advantage_coef
        self.use_step_rewards = use_step_rewards

        self.step_reward_funcs = self.reward_funcs[:2]  # First two reward functions
        self.outcome_reward_funcs = self.reward_funcs[2:]  # Last four reward functions
        self.num_step_funcs = len(self.step_reward_funcs)
        self.num_outcome_funcs = len(self.outcome_reward_funcs)
        self.step_reward_weights = torch.ones(self.num_step_funcs)
        self.outcome_reward_weights = torch.ones(self.num_outcome_funcs)


    def _generate_and_score_completions(
         self, inputs: dict[str, Union[torch.Tensor, Any]]   
    ) -> dict[str, Union[torch.Tensor, Any]]:
        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs] # type: ignore
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)["prompt"] for example in inputs] # type: ignore
        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False # type: ignore
        ) # type: ignore
        prompt_inputs = Trainer._prepare_inputs(self, prompt_inputs) # type: ignore
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length :]
            prompt_mask = prompt_mask[:, -self.max_prompt_length :]

        if self.state.global_step != self._last_loaded_step:
            self._move_model_to_vllm()
            self._last_loaded_step = self.state.global_step

        # Gather the original prompts in message dict form, not the text form
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

        # Pad + mask after per-sequence EOS tokens
        completion_ids = [torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(completion_ids, padding_value=self.processing_class.pad_token_id) # type: ignore

        completion_mask = [torch.tensor(mask, device=device) for mask in completion_mask]
        completion_mask = pad(completion_mask, padding_value=0)

        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1) # (B, P+C)
        
        logits_to_keep = completion_ids.size(1)

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

        # use message dicts for reward function inputs
        completions = completion_messages
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)
        for i, reward_func in enumerate(self.reward_funcs):
            # Repeat all input columns (but "prompt" and "completion") to match the number of generations
            keys = [key for key in inputs[0] if key not in ["prompt", "completion"]] # type: ignore
            reward_kwargs = {key: [example[key] for example in inputs] for key in keys} # type: ignore
            output_reward_func = reward_func(prompts=prompts, completions=completions, **reward_kwargs) # type: ignore
            rewards_per_func[:, i] = torch.tensor(output_reward_func, dtype=torch.float32, device=device)

        rewards_per_func = gather(rewards_per_func) 

        # Apply weights to each reward function's output and sum 
        rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).sum(dim=1) 

       ###############################################################################################
        # Calculate step rewards and outcome rewards separately
        rewards_per_step_func = self._calculate_rewards(
            prompts, completions, self.step_reward_funcs, inputs
        )
        rewards_per_outcome_func = self._calculate_rewards(
            prompts, completions, self.outcome_reward_funcs, inputs
        )
        # Apply weights to each reward function's output and sum
        step_rewards = (rewards_per_step_func * self.step_reward_weights.to(device).unsqueeze(0)).sum(dim=1)
        outcome_rewards = (rewards_per_outcome_func * self.outcome_reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        if self.step_advantage_coef > 0:       
            # Compute normalized advantages
            step_advantages = self._compute_normalized_advantages(step_rewards, len(prompts))
            outcome_advantages = self._compute_normalized_advantages(outcome_rewards, len(prompts))

            # Find the positions of <r> tags in each completion
            result_positions = self._find_result_positions(completion_ids, completion_messages)

            # Apply the combined advantages based on <r> tag positions
            # If there's a <r>, tokens before get step+outcome advantage, after get only outcome
            # If no <r>, all tokens get only outcome advantage
            advantages = self._combine_advantages(
                completion_mask, step_advantages, outcome_advantages, result_positions
            )
        else:
            if self.use_step_rewards:
                advantages = self._compute_normalized_advantages(rewards, len(prompts))
            else:
                advantages = self._compute_normalized_advantages(outcome_rewards, len(prompts))
        ###############################################################################################
        
        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        completion_length = self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item() # type: ignore
        self._metrics[mode]["completion_length"].append(completion_length)

        reward_per_func = rewards_per_func.mean(0) # type: ignore
        for i, reward_func in enumerate(self.reward_funcs):
            reward_func_name = reward_func.__name__ # type: ignore
            self._metrics[mode][f"rewards/{reward_func_name}"].append(reward_per_func[i].item())

        self._metrics[mode]["reward"].append(rewards.mean().item())
        # self._metrics[mode]["reward_std"].append(std_grouped_rewards.mean().item())

        if self.log_completions and self.state.global_step % self.args.logging_steps == 0:
            prompts_to_log = gather_object(prompts)
            completions_to_log = gather_object(completions)
            rewards_to_log = rewards.tolist()

            if self.accelerator.is_main_process:
                if is_rich_available():
                    print_prompt_completions_sample(
                        [str(prompts_to_log[0][-1]["content"])],
                        [completions_to_log[0]],
                        [rewards_to_log[0]],
                        self.state.global_step,
                    )
                if self.args.report_to and "wandb" in self.args.report_to and wandb.run is not None: # type: ignore
                    import pandas as pd

                    # For logging
                    table = {
                        "step": [str(self.state.global_step)] * len(rewards),
                        "prompt": prompts_to_log,
                        "completion": completions_to_log,
                        "reward": rewards.tolist(),
                    }
                    df = pd.DataFrame(table)
                    wandb.log({"completions": wandb.Table(dataframe=df)}) # type: ignore

        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "old_per_token_logps": old_per_token_logps,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

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
        
        return gather(rewards_per_func)
    

    def _compute_normalized_advantages(self, rewards, slice_length=None):
        """Compute normalized advantages from rewards."""
        
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1) # type: ignore

        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore
        advantages = (rewards - mean_grouped_rewards)
        
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1) # type: ignore
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0) # type: ignore

        # Slice to keep only the local part of the data
        process_slice = slice(
            self.accelerator.process_index * slice_length,
            (self.accelerator.process_index + 1) * slice_length,
        )
        return advantages[process_slice]

    # msgrpo specific
    def _find_result_positions(self, completion_ids, completion_messages):
        """
        Find the <result> tag in the environment response and determine the split point.
        
        If the environment response contains a <result> tag, return the start position of the environment response,
        so that all tokens before this position will get step_advantage + outcome_advantage,
        and tokens after this position will only get outcome_advantage.
        
        If no result tag is found, return -1, indicating that the entire sequence only uses outcome_advantage.
        """

        result_positions = []
        
        for i, completion in enumerate(completion_messages):
            ids = completion_ids[i]
            result_pos = -1
            
            # Handle completion content in dialogue history format
            if isinstance(completion, list):
                # Look for the pattern of assistant message followed by user message (env response)
                for j, msg in enumerate(completion):
                    if msg.get('role') == 'assistant':
                        # Check if there is a subsequent environment response
                        if j + 1 < len(completion) and completion[j + 1].get('role') == 'user':
                            user_msg = completion[j + 1].get('content', '')
                            
                            # Check if the environment response contains the <result> tag
                            if '<result>' in user_msg:
                                # Calculate the start token position of the environment response
                                token_pos = 0
                                # Calculate the token length of all previous messages
                                for k in range(j + 1):
                                    token_pos += len(self.processing_class.encode(
                                        str(completion[k].get('content', ''))))
                                
                                # Set the split point to the start position of the environment response
                                result_pos = min(token_pos, len(ids) - 1)
                                break
            
            # Handle completion content in single string format (compatibility retained)
            elif isinstance(completion, str):
                # raise error
                raise ValueError("Completion is a string, which is not supported.")
            
            result_positions.append(result_pos)
            
        return result_positions
    
    # msgrpo specific
    def _combine_advantages(self, completion_mask, step_advantages, outcome_advantages, result_positions):
        """
        Combine step and outcome advantages based on result positions.
        - If result_pos > 0: tokens before get step+outcome, after get only outcome
        - If result_pos = -1: all tokens get only outcome advantage
        
        The step_advantage_coef parameter controls the weight of step advantage.
        """
        
        assert step_advantages.dim() == 1 and outcome_advantages.dim() == 1, \
        "Expected 1D step_advantages and outcome_advantages (shape: [batch_size])"

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
                # Use step_advantage_coef to control the weight of step advantage
                # Expand scalar to sequence length
                outcome_advantage_expanded = outcome_advantages[i].item() * torch.ones_like(completion_mask[i], dtype=torch.float32)
                step_advantage_expanded = step_advantages[i].item() * torch.ones_like(before_result_mask, dtype=torch.float32)
                
                combined_advantages[i] = outcome_advantage_expanded + self.step_advantage_coef * step_advantage_expanded * before_result_mask
            else:
                # No result tag found, use only outcome advantage
                # Expand scalar to sequence length
                outcome_advantage_expanded = outcome_advantages[i].item() * torch.ones_like(completion_mask[i], dtype=torch.float32)
                
                combined_advantages[i] = outcome_advantage_expanded
                
        return combined_advantages
    
    # adopted from GRPOTrainer
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")
        # Compute the per-token log probabilities for the model

        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)  # we only need to compute the logits for the completion tokens

        per_token_logps = self._get_per_token_logps(model, input_ids, attention_mask, logits_to_keep)

        # Compute the KL divergence between the model and the reference model
        if self.beta != 0.0:
            ref_per_token_logps = inputs["ref_per_token_logps"]
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1
            )

        # Compute the loss
        advantages = inputs["advantages"]
        # When using num_iterations == 1, old_per_token_logps == per_token_logps, so we can skip it's computation (see
        # _generate_and_score_completions) and use per_token_logps.detach() instead.
        old_per_token_logps = inputs["old_per_token_logps"] if self.num_iterations > 1 else per_token_logps.detach()
        coef_1 = torch.exp(per_token_logps - old_per_token_logps)
        coef_2 = torch.clamp(coef_1, 1 - self.epsilon, 1 + self.epsilon)
        ############################################################################################################
        # If the advantages are 1D, we need to unsqueeze it to match the shape of the per-token loss
        if advantages.dim() == 1:
            advantages = advantages.unsqueeze(1)
        ############################################################################################################
        per_token_loss1 = coef_1 * advantages.unsqueeze(1)
        per_token_loss2 = coef_2 * advantages.unsqueeze(1)
        per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
        if self.beta != 0.0:
            per_token_loss = per_token_loss + self.beta * per_token_kl
        loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        mode = "eval" if self.control.should_evaluate else "train"

        if self.beta != 0.0:
            mean_kl = (per_token_kl * completion_mask).sum() / completion_mask.sum()
            self._metrics[mode]["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        is_clipped = (per_token_loss1 < per_token_loss2).float()
        clip_ratio = (is_clipped * completion_mask).sum() / completion_mask.sum()
        self._metrics[mode]["clip_ratio"].append(self.accelerator.gather_for_metrics(clip_ratio).mean().item())
        return loss
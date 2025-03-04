from typing import List, Dict, Optional
import os
import re
from openai import OpenAI

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

os.environ["LLM_API_KEY"] = "token-abc123"

class LLMRubric(Rubric):
    """
    Rubric that uses an LLM to evaluate the correctness of answers.
    """
    def __init__(self, 
                 parser: XMLParser = XMLParser(fields=["reasoning", "answer"]),
                 llm_model_name: str = "Qwen/Qwen2.5-72B-Instruct"):
        """
        Initialize the LLM-based evaluation rubric.
        
        Args:
            parser: XMLParser to use for extracting answers
            llm_model_name: Name of the LLM model to use for evaluation
        """
        super().__init__()
        self.parser = parser
        self.llm_model_name = llm_model_name
        self.llm_client = None
        
        
        # Set up reward functions
        self.reward_funcs = [
            #self.exact_answer_reward_func,  # Exact string matching (traditional)
            self.llm_verify_judge_reward_func,    # LLM-based semantic evaluation
            self.int_answer_reward_func,
            self.parser.get_format_reward_func(),  # Check format compliance
            self.parser.get_xml_reward_func()      # Check XML structure
        ]
        
        # # Optional: set weights for the reward functions
        # self.reward_weights = [0.4, 0.4, 0.1, 0.1]  # Prioritize accuracy

    def _get_llm_client(self):
        """Lazy initialization of the LLM client."""
        if self.llm_client is None:
            def sanitize_name(name):
                """Convert model name to valid k8s name"""
                # Remove organization prefix if present
                name = name.split('/')[-1]
                # Replace invalid chars with dash
                return re.sub(r'[^a-z0-9-]', '-', name.lower())
            
            def get_model_url(model_name: str) -> str:
                sanitized_name = sanitize_name(model_name)
                url = f"http://{sanitized_name}.default.svc.cluster.local/v1/"
                return url
            
            url = get_model_url(self.llm_model_name)
            self.llm_client = OpenAI(api_key=os.environ["LLM_API_KEY"], base_url=url)
        
        return self.llm_client

    def llm_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that uses an LLM to evaluate if the answer is correct.
        
        Args:
            completions: List of completion trajectories
            answer: List of ground truth answers
            
        Returns:
            List of reward scores between 0.0 and 1.0
        """
        
        
        
        """Reward function that checks if the final answer matches the expected answer."""
        # responses = [self.get_last_answer(c) for c in completions]
        # return [1.0 if str(r) == str(a) else 0.0 for r, a in zip(responses, answer)]
        
        
        
        client = self._get_llm_client()
        responses = [self.get_last_answer(c) for c in completions]
        rewards = []
        
        for resp, ans in zip(responses, answer):
            if resp is None:
                rewards.append(0.0)
                continue
                
            # Create prompt for the LLM to evaluate the answer
            prompt = [
                {"role": "system", "content": (
                    "You are an expert evaluator. Your task is to determine if the provided answer "
                    "is correct given the ground truth. Only output 'yes' or 'no'."
                )},
                {"role": "user", "content": (
                    f"The ground truth answer is: {ans}\n\n"
                    f"The given answer is: {resp}\n\n"
                    f"Is the given answer correct? Answer with just 'yes' or 'no'."
                )}
            ]
            
            result = client.chat.completions.create(
                model=self.llm_model_name,
                messages=prompt,
                # temperature=0,
                # max_tokens=5
            )
            
            # Parse the result
            eval_response = result.choices[0].message.content.strip().lower()
            reward = 1.0 if "yes" in eval_response else 0.0
            rewards.append(reward)
            self.logger.info(f"LLM evaluation: '{resp}' vs '{ans}' -> {eval_response} (reward: {reward})")
                
            # except Exception as e:
            #     self.logger.error(f"Error in LLM evaluation: {e}")
            #     rewards.append(0.0)
                
        return rewards
    
    def llm_verify_reward_func(self, completions, answer, prompts, **kwargs) -> List[float]:
        """Reward function that uses an LLM to evaluate if the answer is correct based on the original prompt.
        
        Args:
            completions: List of completion trajectories
            answer: List of ground truth answers (not used in this implementation)
            prompts: List of original prompts containing the questions
            
        Returns:
            List of reward scores between 0.0 and 1.0
        """
        client = self._get_llm_client()
        responses = [self.get_last_answer(c) for c in completions]
        rewards = []
        
        
        for resp, prompt, ans in zip(responses, prompts, answer):
            if resp is None:
                rewards.append(0.0)
                continue
            
            user_message = str(prompt)
            
            eval_prompt = [
                {"role": "system", "content": (
                    "You are an expert evaluator. Your task is to determine if the provided answer "
                    "correctly addresses the given question or task. Evaluate the answer solely based "
                    "on its correctness and appropriateness for the question. "
                    "Answer with just 'yes' if the answer is correct, or 'no' if it's incorrect."
                )},
                {"role": "user", "content": (
                    f"Question/Task: {user_message}\n\n"
                    f"Provided Answer: {resp}\n\n"
                    f"Is this answer correct and appropriate for the question/task? Answer with just 'yes' or 'no'."
                )}
            ]
            
            result = client.chat.completions.create(
                model=self.llm_model_name,
                messages=eval_prompt,
                # temperature=0,
                # max_tokens=5
            )
            
            eval_response = result.choices[0].message.content.strip().lower()
            reward = 1.0 if "yes" in eval_response else 0.0
            rewards.append(reward)
            
            # please compare the eval_response with the ground truth answer
            self.logger.info(f"LLM evaluation: '{resp}' vs '{ans}' -> {eval_response} (reward: {reward})")
            
            
        return rewards
    
    def llm_verify_judge_reward_func(self, completions, answer, prompts, **kwargs) -> List[float]:
        """Reward function that uses an LLM to verify answers by calculating the answer itself.
        
        Args:
            completions: List of completion trajectories
            answer: List of ground truth answers (not used in this implementation)
            prompts: List of original prompts containing the questions
            
        Returns:
            List of reward scores between 0.0 and 1.0
        """
        client = self._get_llm_client()
        responses = [self.get_last_answer(c) for c in completions]
        rewards = []
        
        for resp, prompt, ans in zip(responses, prompts, answer):
            if resp is None:
                rewards.append(0.0)
                continue
            
            user_message = str(prompt)
            

            solve_prompt = [
                {"role": "system", "content": (
                    "You are an expert problem solver. Solve the given problem accurately and "
                    "provide just the final numerical answer without any explanation or working."
                )},
                {"role": "user", "content": user_message}
            ]
            
            solve_result = client.chat.completions.create(
                model=self.llm_model_name,
                messages=solve_prompt,
                #temperature=0
            )
            
            llm_answer = solve_result.choices[0].message.content.strip()
            
            # 步骤 2: 让 LLM 比较两个答案是否等价
            compare_prompt = [
                {"role": "system", "content": (
                    "You are an expert evaluator. Compare the following two answers and determine if they are "
                    "mathematically equivalent. Consider different formats of the same value as equivalent "
                    "(e.g., '5', '5.0', 'five' are all equivalent). "
                    "Answer with just 'yes' if they are equivalent, or 'no' if they are not."
                )},
                {"role": "user", "content": (
                    f"The correct answer calculated is: {llm_answer}\n\n"
                    f"The provided answer is: {resp}\n\n"
                    f"Are these answers equivalent? Answer with just 'yes' or 'no'."
                )}
            ]
            
            compare_result = client.chat.completions.create(
                model=self.llm_model_name,
                messages=compare_prompt,
                # temperature=0,
                # max_tokens=5
            )
            
            # 解析结果
            eval_response = compare_result.choices[0].message.content.strip().lower()
            reward = 1.0 if "yes" in eval_response else 0.0
            rewards.append(reward)
            
            # 记录结果用于调试
            self.logger.info(f"LLM verification: Ground truth answer: '{ans}', LLM answer: '{llm_answer}', User answer: '{resp}' -> {eval_response} (reward: {reward})")
                
        return rewards
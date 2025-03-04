from typing import List, Dict, Optional
import os
import re
from openai import OpenAI

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric

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
            self.llm_answer_reward_func,    # LLM-based semantic evaluation
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
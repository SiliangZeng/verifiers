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
            self.exact_answer_reward_func,    # LLM-based semantic evaluation
            self.llm_reasoning_reward_func,
            self.int_answer_reward_func,
            self.parser.get_format_reward_func(),  # Check format compliance
            self.parser.get_xml_reward_func()      # Check XML structure
        ]
        
        # # Optional: set weights for the reward functions
        #self.reward_weights = [0.4, 0.4, 0.1, 0.1]  # Prioritize accuracy
        #self.reward_weights = [0.4, 0.4, 0.1, 0.1, 0.1]

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
    
    def get_reasoning(self, trajectory: List[Dict[str, str]]) -> str | None:
        """Extract the reasoning steps from a trajectory."""
        for msg in reversed(trajectory):
            if msg['role'] == 'assistant':
                if self.parser is None:
                    raise ValueError("Parser is not set")
                parsed = self.parser.parse(msg['content'])
                if hasattr(parsed, 'reasoning') and parsed.reasoning is not None:
                    return parsed.reasoning
        return None

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
        
        user_message = str(prompts[0])
            
        solve_prompt = [
            {"role": "system", "content": (
                "You are an expert problem solver. Solve the given problem carefully and accurately. "
                "First, show your reasoning process to work through the problem step by step. "
                "Then, provide your final answer as an INTEGER NUMBER (no decimals, no text, just digits). "
                "Format your response as follows:\n\n"
                "<reasoning>\n[Your detailed reasoning process]\n</reasoning>\n\n"
                "<answer>\n[Your final answer as an integer, e.g., 42, 7, 100]\n</answer>\n\n"
                "Remember: The answer must be ONLY digits, no text, no decimals, no special characters."
            )},
            {"role": "user", "content": user_message}
        ]
        
        solve_result = client.chat.completions.create(
            model=self.llm_model_name,
            messages=solve_prompt,
            #temperature=0
        )
        
        llm_content = solve_result.choices[0].message.content.strip()
            
        llm_trajectory = [{"role": "assistant", "content": llm_content}]
        
        extracted_answer = self.get_last_answer(llm_trajectory)
        
        for user_resp, ans in zip(responses, answer):
            if user_resp is None:
                rewards.append(0.0)
                continue
            

            reward = 1.0 if str(user_resp).strip() == str(extracted_answer).strip() else 0.0
            rewards.append(reward)
            
            self.logger.info(f"LLM verification: Ground truth: '{ans}', LLM answer: '{extracted_answer}', User answer: '{user_resp}' -> Matched: {reward == 1.0}")
                
        return rewards
    
    
    def llm_reasoning_reward_func(self, completions, prompts, **kwargs) -> List[float]:
        """Reward function that uses an LLM to evaluate the reasoning process.
        
        Args:
            completions: List of completion trajectories
            prompts: List of original prompts containing the questions
            
        Returns:
            List of reward scores between 0.0 and 1.0
        """
        client = self._get_llm_client()
        reasonings = [self.get_reasoning(c) for c in completions]
        rewards = []
        
        for reasoning, prompt in zip(reasonings, prompts):
            if reasoning is None:
                rewards.append(0.0)
                continue
                
            user_message = str(prompt)
            
            # create the prompt for the LLM to evaluate the reasoning process
            eval_prompt = [
                {"role": "system", "content": (
                    "You are an expert evaluator of mathematical reasoning. "
                    "Your task is to evaluate the quality of a reasoning process for solving a given problem. "
                    "Consider the following criteria:\n"
                    "1. Logical Flow: Steps follow logically from one to the next\n"
                    "2. Correctness: Mathematical operations and concepts are used correctly\n"
                    "3. Completeness: All necessary steps are included\n"
                    "4. Clarity: The reasoning is clear and well-explained\n\n"
                    "Provide your evaluation in the following format:\n\n"
                    "<reasoning>\n"
                    "[Your detailed analysis of the reasoning process, discussing strengths and weaknesses "
                    "based on the above criteria]\n"
                    "</reasoning>\n\n"
                    "<answer>\n"
                    "[A single number between 0 and 0.5, where:\n"
                    "0.5: Perfect reasoning with clear steps and correct logic\n"
                    "0.35-0.45: Good reasoning with minor issues\n"
                    "0.2-0.3: Acceptable but with significant gaps or errors\n"
                    "0.05-0.15: Poor reasoning with major logical flaws\n"
                    "0.0: No valid reasoning or completely incorrect]\n"
                    "</answer>\n\n"
                    "Important: The <answer> section must contain ONLY the numerical score (e.g., 0.4), "
                    "no additional text or explanation."
                )},
                {"role": "user", "content": (
                    f"Problem:\n{user_message}\n\n"
                    f"Student's Reasoning Process:\n{reasoning}\n\n"
                    f"Evaluate this reasoning process:"
                )}
            ]
            
            eval_result = client.chat.completions.create(
                model=self.llm_model_name,
                messages=eval_prompt,
                # temperature=0
            )
            
            # create a fake trajectory containing the LLM evaluation
            eval_content = eval_result.choices[0].message.content.strip()
            eval_trajectory = [{"role": "assistant", "content": eval_content}]
            
            # extract the score from the LLM evaluation
            score_str = self.get_last_answer(eval_trajectory)
            
            # if score_str is not None and is a number between 0 and 1
            if score_str is not None and 0 <= float(score_str.strip()) <= 0.5:
                score = float(score_str.strip())
                # ensure the score is between 0 and 1
                score = max(0.0, min(0.5, score))
                rewards.append(score)
                
                
                
                eval_reasoning = self.get_reasoning(eval_trajectory)
                if eval_reasoning:
                    self.logger.info(
                        f"Reasoning evaluation:\nScore: {score:.2f}\n"
                        f"Evaluation:\n{eval_reasoning}..."
                    )
            else:
                self.logger.error("No score found in LLM evaluation")
                self.logger.error(f"LLM evaluation: {score_str}")
                rewards.append(0.0)
                    
        
        return rewards
    
from typing import List, Dict

from verifiers.parsers import XMLParser
from verifiers.rubrics import Rubric
from verifiers.rubrics import ToolRubric

class TrivialQAToolRubric(ToolRubric):
    def __init__(self,
                 parser: XMLParser = XMLParser(fields=["reasoning", ("tool", "answer")]),
                 env_parser: XMLParser = XMLParser(fields=["result"])):
        self.parser = parser
        self.env_parser = env_parser
        self.reward_funcs = [
            self.exist_answer_reward_func,
            self.tool_execution_reward_func,
            self.parser.get_format_reward_func(),
            self.parser.get_xml_reward_func(),
        ]
    
    def exist_answer_reward_func(self, completions, answer, **kwargs) -> List[float]:
        """Reward function that checks if the expected answer exists in the completions."""
        
        '''
        
        def exact_match_reward(responses, answers=None):
            """Reward if generated response contains correct answer."""
            rewards = []
            for response, answer in zip(responses, answers):
                reward = 0.0
                for a in answer:
                    if a.lower() in response.lower():
                        reward += 1.0
                        break
                rewards.append(torch.tensor(reward))
            return rewards
                
        '''
        
        responses = [self.get_last_answer(c) for c in completions]
        rewards = []
        
        for r, a_list in zip(responses, answer):
            if r is None:
                rewards.append(0.0)
                continue
                
            r_lower = str(r).lower()
            reward = 0.0
            
            # Check if any of the accepted answers is contained in the response
            for a in a_list:
                if str(a).lower() in r_lower:
                    reward = 1.0
                    break
            
            rewards.append(reward)
            
        return rewards
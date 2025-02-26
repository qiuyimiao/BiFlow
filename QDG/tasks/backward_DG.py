import os
import re
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.backward_DG import *
from tot.models import gpt
import random


class backwardDGTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='fairytale_test_separated.json'):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        self.data_name_json = file 
        self.data_name = file.rstrip(".json") 
        self.file = os.path.join(DATA_PATH, 'backward_DG', file)
        self.data_initial = json.load(open(self.file, encoding='utf-8'))
        self.data_without_ans = [
            {
                "question_id": item["question_id"],
                "context": item["context"],
                "question": item["question"],
                "choices": random.sample(item["answers"] + item["distractors"], k=len(item["answers"] + item["distractors"]))
            }
            for item in self.data_initial
        ]
        self.data_with_ans = [
            {
                "question_id": item["question_id"],
                "answers": item["answers"]
            }
            for item in self.data_initial
        ]
        self.data_to_save = [#id,context,question,answers,distractors
            {
                "question_id": item["question_id"],
                "context": item["context"],
                "question": item["question"],
                "answers": item["answers"],
                "distractors": item["distractors"]
            }
            for item in self.data_initial
        ]
        self.steps = 1
        self.stops = ['\nQuestion:\n', None]

    def __len__(self) -> int:
        return len(self.data_with_ans)
    
    def get_input_without_ans(self, idx: int) -> str:
        print(len(self.data_without_ans))
        print(idx)
        print(self.data_without_ans[idx])
        return self.data_without_ans[idx]
    
    def get_input_with_ans(self, idx: int) -> str: 
        #print(len(self.data_with_ans))
        #print(idx)
        return self.data_with_ans[idx]
    
    def get_input_with_all(self, idx: int) -> str: 
        #print(len(self.data_to_save))
        #print(idx)
        return self.data_to_save[idx]
    
    '''
    def test_output(self, idx: int, output: str):
        output = output.split('Question:\n')[-1]
        prompt = score_prompt.format(input=self.data[idx]) + output  #format中进行score
        score_outputs = gpt(prompt, n=5, model='gpt-4o')
        scores = []
        for score_output in score_outputs:
            # print(score_output)
            pattern = r".*matching score is (\d+).*"
            match = re.match(pattern, score_output, re.DOTALL)
            if match:
                score = int(match.groups()[0])
                scores.append(score)
            else:
                print(f'------------------score no match: {[score_output]}')
        print(scores)
        # print('------------')
        info = {'rs': scores, 'r': sum(scores) / len(scores) if scores else 0}
        return info
    '''
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='') -> str:  
        return standard_prompt.format(input=x) + y     

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:       
        return cot_prompt.format(input=x) + y           

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:      
        prompts = []
        #print(x)  #?
        for y in ys:
            prompt = vote_prompt.format(input=x)
            prompt += f'Answer set:\n{y}\n'
            prompts.append(prompt)
        return prompts 
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list: 
        #vote_results = [0] * 2 
        right_num = 0
        wrong_num = 0
        for vote_output in vote_outputs:
            pattern = r".*The student's answer is .*(\d+).*"
            match = re.match(pattern, vote_output[0], re.DOTALL)
            if match:
                vote = int(match.groups()[0]) 
                if vote == 0:
                    wrong_num+=1
                elif vote == 1:
                    right_num+=1
            else:
                print(f'vote no match: {vote_output}')
        accuracy_rate = round(float(right_num) / float(right_num + wrong_num), 2)
        return accuracy_rate 

    @staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:                     
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Passage:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
        return prompt.format(input=x)
    
    @staticmethod
    def compare_output_unwrap(compare_output: str):                        
        if 'more coherent passage is 1' in compare_output:
            return 0
        elif 'more coherent passage is 2' in compare_output:
            return 1
        elif 'two passages are similarly coherent' in compare_output:
            return 0.5
        else:
            print(f'-----------------compare no match: {[compare_output]}')
            return -1
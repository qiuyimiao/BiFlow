import os
import re
import json
from tot.tasks.base import Task, DATA_PATH
from tot.prompts.forward_DG import *
from tot.models import gpt
import random


class forwardDGTask(Task):
    """
    Input (x)   : a text instruction
    Output (y)  : a text generation
    Reward (r)  : # TODO
    Input Example: 
    Output Example: 
    """
    def __init__(self, file='fairytale_test.json'):
        """
        file: a text file, each line is some sentences
        """
        super().__init__()
        self.data_name_json = file 
        self.data_name = file.rstrip(".json") 
        self.file = os.path.join(DATA_PATH, 'forward_DG', file)
        self.QGfile = os.path.join(DATA_PATH, 'forward_QG', file)
        self.data_initial = json.load(open(self.file, encoding='utf-8'))          
        self.QG_data = json.load(open(self.QGfile, encoding='utf-8')) 
        self.data = [
            {
                "question_id": item["question_id"],
                "context": item["context"],
                "question": item["generated_question"],
                "answers": item["answers"]
            }
            for item in self.data_initial
        ]
        self.data_with_distractors = [     
            {
                "context": item["context"],
                "question": item["question"],
                "answers": item["answers"],
                "generated_distractors": item["distractors"]
            }
            for item in self.QG_data
        ]   
        self.data_to_save = [
            {
                "question_id": item["question_id"],
                "context": item["context"],
                "question": item["generated_question"],
                "gold_question": item["gold_question"],
                "answers": item["answers"]
            }
            for item in self.data_initial
        ]
        self.steps = 2
        self.stops = ['\nDistractors\n', None]

    def __len__(self) -> int:
        return len(self.data)
    
    def get_input(self, idx: int) -> str:
        print(len(self.data))
        print(idx)
        return self.data[idx]
    
    def get_example_1(self) -> str:
        return random.choice(self.data_with_distractors)

    def get_example_3(self) -> str:
        return random.sample(self.data_with_distractors, 3)

    '''
    def get_input_with_all(self, idx: int) -> str:
        print(len(self.data_to_save))
        print(idx)
        return self.data_to_save[idx]
    '''
    
    def get_data_name(self) -> str:
        return self.data_name
    
    '''def test_output(self, idx: int, output: str):  
        output = output.split('Distractors:\n')[-1]
        prompt = score_prompt.format(input=self.data[idx]) + output 
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
        return info'''
    
    @staticmethod
    def standard_prompt_wrap(x: str, y:str='', ex =[]) -> str:  
        if ex==[]:
            return standard_prompt.format(input=x) + y      
        else:
            return standard_prompt_shot.format(example=ex, input=x) + y

    @staticmethod
    def cot_prompt_wrap(x: str, y:str='') -> str:       
        return cot_prompt.format(input=x) + y           

    @staticmethod
    def vote_prompt_wrap(x: str, ys: list) -> str:      
        prompt = vote_prompt
        print(x)  #?
        for i, y in enumerate(ys, 1):
            # y = y.replace('Plan:\n', '')
            # TODO: truncate the plan part?
            prompt += f'Choice {i}:\n{y}\n'
        return prompt.format(input=x)
    
    @staticmethod
    def vote_outputs_unwrap(vote_outputs: list, n_candidates: int) -> list: 
        vote_results = [0] * n_candidates
        for vote_output in vote_outputs:
            pattern = r".*best distractor set is (\d+).*"
            #pattern = r".*best distractors are \{([\d,]+)\}.*"
            match = re.match(pattern, vote_output, re.DOTALL)
            if match:
                vote = int(match.groups()[0]) - 1
                print("Vote num:")
                if vote in range(n_candidates):
                    print(vote)
                    vote_results[vote] += 1
            else:
                print(f'vote no match: {[vote_output]}')
        return vote_results

    '''@staticmethod
    def compare_prompt_wrap(x: str, ys: list) -> str:                      
        assert len(ys) == 2, 'compare prompt only supports 2 candidates'
        ys = [y.split('Passage:\n')[-1] for y in ys]
        prompt = compare_prompt + f'Passage 1:\n{ys[0]}\n\nPassage 2:\n{ys[1]}\n'
        return prompt.format(input=x)
    '''
    
    '''
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
    '''
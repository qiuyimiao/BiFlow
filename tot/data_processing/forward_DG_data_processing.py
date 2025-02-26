
import json
import os
import re

dataset_name = "fairytale_test"
drop_json = "fairytale_test.json"
drop_separated_json = "fairytale_test_separated.json"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data/')
LOG_PATH = os.path.join(os.path.dirname(__file__),'../..','logs/')

def json_load(file_path):
    """Load a JSON file with UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def json_dump(data, file_path):
    """Dump JSON data into a file with UTF-8 encoding."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_forward_DG_data(forward_file, output_file):
    forward_QG_data = json_load(forward_file)
    #dataset_data = json_load(dataset)
    '''processed_data = [
        {
            "question_id": item["question_id"],
            "context": item["context"],
            "gold_question": item["question"], #已有的题目
            "gold_answers": item["answers"] #已有的正确答案
        }
            for item in dataset_data
    ]'''
    #id_question_map = {item["question_id"]:item["question"] for item in dataset_data}

    extracted_data = []
    for data in forward_QG_data:
        for step in data["steps"]:
            if step["step"] == 1:
                x = step["x"]
                id = x["question_id"]
                context = x["context"]
                question = x["question"]
                answers = x["answers"]
                select_new_ys = step["select_new_ys"]
                #generated_questions = [ys.split("Question:")[-1].strip() for ys in select_new_ys]

                distractors = []
                for text in select_new_ys:
                    text = re.sub(r'[\*\n]', '', text)
                    pattern = r"Distractor 1: (.*?)\s*Distractor 2: (.*?)\s*Distractor 3: (.*?)\s*$"
                    matches = re.findall(pattern, text)
                    if matches:
                        print(f"成功匹配id:{id}")
                        print(matches[0])  
                        distractors.append(matches[0])
                    else:
                        print(f"问题ID {id}：没有找到匹配的干扰项。文本内容：{text}")
                
                extracted_data.append({
                    "question_id": id,
                    "context": context,
                    "question": question,
                    "answers": answers,
                    "distractors": distractors
                })
    json_dump(extracted_data,output_file)
    
def reverse_match(text, pattern):
    reversed_text = text[::-1]
    reversed_pattern = r"3 rotcartsiD: (.*?)\s*2 rotcartsiD: (.*?)\s*1 rotcartsiD: (.*?)(?=\s*3 rotcartsiD|\s*$)"
    matches = re.findall(reversed_pattern, reversed_text)
    if matches:
        last_match = matches[0]
        return tuple(item[::-1] for item in last_match[::-1])
    return None


def generate_forward_DG_data_separated(forward_file, output_file):
    forward_QG_data = json_load(forward_file)
    extracted_data = []
    for data in forward_QG_data:
        for step in data["steps"]:
            if step["step"] == 1:
                x = step["x"]
                id = x["question_id"]
                context = x["context"]
                question = x["question"]
                answers = x["answers"]
                select_new_ys = step["select_new_ys"]
                #generated_questions = [ys.split("Question:")[-1].strip() for ys in select_new_ys]
                pattern = r"Distractor 1: (.*?)\s*Distractor 2: (.*?)\s*Distractor 3: (.*?)\s*$"
                distractors = []
                for text in select_new_ys:
                    text = re.sub(r'[\*\n]', '', text)
                    #text = text.replace('.', '')
                    text = re.sub(r".*?(?=\s*Plan)", "", text)
                    #print(text)
                    matches = re.findall(pattern, text)
                    #match = reverse_match(text, pattern)
                    print(matches)
                    #distractors.append([list(match) for match in matches])
                    distractors.append(matches[0])
                for distractor in distractors:
                    extracted_data.append({
                        "question_id": id,
                        "context": context,
                        "question": question,
                        "answers": answers,
                        "distractors": distractor
                    })
    json_dump(extracted_data,output_file)

if __name__ == '__main__':
    forward_log_json = "gpt-4o_1.0_sample6_vote10_greedy3_start0_end595.json"
    input_forward_dataset_file = os.path.join(LOG_PATH, 'forward_DG', dataset_name)
    input_forward_file = os.path.join(input_forward_dataset_file,forward_log_json)
    output_backward_file = os.path.join(DATA_PATH, 'backward_DG', drop_json)
    output_backward_file_separated = os.path.join(DATA_PATH, 'backward_DG', drop_separated_json)
    #generate_forward_DG_data(input_forward_file,output_backward_file)
    generate_forward_DG_data_separated(input_forward_file,output_backward_file_separated)
            


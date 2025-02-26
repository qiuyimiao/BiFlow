import json
import os
#from tot.tasks.base import Task, DATA_PATH,LOG_PATH
dataset = "fairytale_test"
drop_json = "fairytale_test.json"
drop_separated_json = "fairytale_test_separated.json"
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data')
LOG_PATH = os.path.join(os.path.dirname(__file__),'../..','logs')

def json_load(file_path):
    """Load a JSON file with UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def json_dump(data, file_path):
    """Dump JSON data into a file with UTF-8 encoding."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def generate_forward_QG_data(forward_file, dataset,output_file):
    forward_QG_data = json_load(forward_file)
    dataset_data = json_load(dataset)
    '''processed_data = [
        {
            "question_id": item["question_id"],
            "context": item["context"],
            "gold_question": item["question"], #已有的题目
            "gold_answers": item["answers"] #已有的正确答案
        }
            for item in dataset_data
    ]'''
    id_question_map = {item["question_id"]:item["question"] for item in dataset_data}
    id_ans_type_map = {item["question_id"]:item["ans_type"] for item in dataset_data}
    extracted_data = []
    for data in forward_QG_data:
        for step in data["steps"]:
            if step["step"] == 1:
                x = step["x"]
                id = x["question_id"]
                context = x["context"]
                answers = [answer["text"] for answer in x["answers"]]
                select_new_ys = step["select_new_ys"]
                generated_questions = [ys.split("Question:")[-1].strip() for ys in select_new_ys]
                gold_question = id_question_map.get(id,None)
                ans_type = id_ans_type_map.get(id,None)

                extracted_data.append({
                    "question_id": id,
                    "context": context,
                    "gold_question": gold_question,
                    "generated_questions": generated_questions,
                    "gold_answers": answers,
                    "ans_type": ans_type
                })
    json_dump(extracted_data,output_file)
    #return extracted_data

def generate_forward_QG_data_separated(forward_file, dataset,output_file):
    forward_QG_data = json_load(forward_file)
    dataset_data = json_load(dataset)
    '''processed_data = [
        {
            "question_id": item["question_id"],
            "context": item["context"],
            "gold_question": item["question"], #已有的题目
            "gold_answers": item["answers"] #已有的正确答案
        }
            for item in dataset_data
    ]'''
    id_question_map = {item["question_id"]:item["question"] for item in dataset_data}
    id_ans_type_map = {item["question_id"]:item["ans_type"] for item in dataset_data}

    extracted_data = []
    for data in forward_QG_data:
        for step in data["steps"]:
            if step["step"] == 1:
                x = step["x"]
                id = x["question_id"]
                context = x["context"]
                answers = [answer["text"] for answer in x["answers"]]
                select_new_ys = step["select_new_ys"]
                generated_questions = [ys.split("Question:")[-1].strip() for ys in select_new_ys]
                for generated_question in generated_questions:
                    gold_question = id_question_map.get(id,None)
                    ans_type = id_ans_type_map.get(id,None)
                    extracted_data.append({
                        "question_id": id,
                        "context": context,
                        "gold_question": gold_question,
                        "generated_question": generated_question,
                        "gold_answers": answers,
                        "ans_type": ans_type
                    })
    json_dump(extracted_data,output_file)
    #return extracted_data

if __name__ == '__main__':
    forward_log_json = "gpt-4o_1.0_sample6_vote10_greedy3_start0_end1007.json"#"gpt-4o_1.0_sample6_vote10_greedy3_start0_end1007.json"
    input_forward_file = os.path.join(LOG_PATH, 'forward_QG', dataset, forward_log_json)
    input_dataset_file = os.path.join(DATA_PATH, 'forward_QG', drop_json)
    output_backward_file = os.path.join(DATA_PATH, 'backward_QG', drop_json)
    output_backward_file_separated = os.path.join(DATA_PATH, 'backward_QG', drop_separated_json)
    #generate_forward_QG_data(input_forward_file,input_dataset_file,output_backward_file)
    generate_forward_QG_data_separated(input_forward_file,input_dataset_file,output_backward_file_separated)
            


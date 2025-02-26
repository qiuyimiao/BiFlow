import json
import os
from collections import defaultdict
from math import exp, log
#from tot.tasks.base import Task, DATA_PATH,LOG_PATH
drop_json = "fairytale_test.json"
dataset = "fairytale_test"
dataset_separated = "fairytale_test_separated"
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


def generated_same_question_id_backward(backward_QG_log_file):
    backward_QG_data = json_load(backward_QG_log_file)

    aggregated_data = defaultdict(lambda: {
        "context": "",
        "gold_question": "",
        "generated_questions": [],
        "correct_accuracy": [],
        "answers": [],
        "ans_typr": ""
    })
    for data in backward_QG_data:
        for step in data["steps"]:
            x = step["x"]
            question_id = x['question_id']
            
            aggregated_data[question_id]['context'] = x['context']
            aggregated_data[question_id]['gold_question'] = x['gold_question']
            aggregated_data[question_id]['answers'] = x['answers']
           
            aggregated_data[question_id]['generated_questions'].append(x['generated_question'])
            aggregated_data[question_id]['correct_accuracy'].append(step['correct_accuracy'])
    result = [
        {
            "question_id": qid,
            "context": value['context'],
            "gold_question": value['gold_question'],
            "generated_questions": value['generated_questions'],
            "correct_accuracy": value['correct_accuracy'],
            "answers": value['answers']
        }
        for qid, value in aggregated_data.items()
    ]
    
    return result 

def generated_same_question_id_forward(forward_QG_log_file):
    forward_QG_data = json_load(forward_QG_log_file)
    aggregated_data = defaultdict(lambda: {
        "context": "",
        "votes": []
    })
    for data in forward_QG_data:
        for step in data["steps"]:
            if step["step"] == 1:
                x = step["x"]
                question_id = x['question_id']
                aggregated_data[question_id]['context'] = x['context']
                aggregated_data[question_id]['votes'] = step['select_new_ys_votes']
        result = [
            {
                "question_id": qid,
                "context": value['context'],
                "votes": value['votes']
            }
            for qid, value in aggregated_data.items()
    ]
    return result 

def get_combined_probability(forward_result, backward_result):
    combined_results = []
    alpha = 0.35
    beta = 1 - alpha
    forward_dict = {item["question_id"]: item for item in forward_result}
    
    for backward_item in backward_result:
        question_id = backward_item["question_id"]
        context = backward_item["context"]
        gold_question = backward_item["gold_question"]
        generated_questions = backward_item["generated_questions"]
        correct_accuracy = backward_item["correct_accuracy"]
        answers = backward_item["answers"]
        
        if question_id in forward_dict:
            forward_votes = forward_dict[question_id]["votes"]
            print(forward_votes)#test
            print(correct_accuracy)#test
            if sum(forward_votes) != 0:
                forward_weights = [value/sum(forward_votes) for value in forward_votes]
            else:
                forward_weights = [0 for value in forward_votes]
            if sum(correct_accuracy) !=0:
                backward_weights = [value/sum(correct_accuracy) for value in correct_accuracy]
            else:
                backward_weight = [0 for value in correct_accuracy]
            #print(forward_weights)
            #print(backward_weights)
            combined_scores = []
            
            for i in range(len(generated_questions)):
                if i < len(forward_votes) and i < len(correct_accuracy):
                    forward_weight = forward_weights[i]
                    backward_weight = backward_weights[i]
                    combined_score = forward_weight ** alpha * backward_weight ** beta
                    combined_scores.append(combined_score)
            
            combined_results.append({
                "question_id": question_id,
                "context": context,
                "gold_question": gold_question,
                "generated_questions": generated_questions,
                "combined_scores": combined_scores,
                "answers": answers
            })
    
    #print(combined_results)  # test
    return combined_results

def generate_backward_QG_data(combined_results,dataset_file,output_file):
    
    dataset = json_load(dataset_file)
    filtered_data = []
    
    id_to_difficulty = {item['question_id']: item['difficulty'] for item in dataset}
    for entry in combined_results:
        max_combined_results_index = entry['combined_scores'].index(max(entry['combined_scores']))
        
        best_question = entry['generated_questions'][max_combined_results_index]
        
        filtered_data.append({
            "question_id": entry['question_id'],
            "context": entry['context'],
            "difficulty": id_to_difficulty.get(entry['question_id'], None),
            "gold_question": entry['gold_question'],
            "generated_question": [best_question],  
            "combined_scores": [entry['combined_scores'][max_combined_results_index]], 
            "answers": entry['answers']
        })
    
    json_dump(filtered_data,output_file)

if __name__ == '__main__':
    forward_log_json = "gpt-4o_1.0_sample6_vote10_greedy3_start0_end1007.json"#"gpt-4o_1.0_sample6_vote10_greedy3_start0_end1007.json"
    backward_log_json = "gpt-4o_1.0_sample6_vote1_greedy1_start0_end3021.json"#"gpt-4o_1.0_sample6_vote1_greedy1_start0_end3021.json"
    input_backward_file = os.path.join(LOG_PATH, 'backward_QG', dataset_separated, backward_log_json)
    input_forward_file = os.path.join(LOG_PATH, 'forward_QG', dataset, forward_log_json)
    input_dataset_file = os.path.join(DATA_PATH, 'forward_QG', drop_json)
    output_forward_DG_file = os.path.join(DATA_PATH, 'forward_DG', drop_json)
    backward_result = generated_same_question_id_backward(input_backward_file)
    forward_result = generated_same_question_id_forward(input_forward_file)
    combine_result = get_combined_probability(forward_result, backward_result)
    generate_backward_QG_data(combine_result,input_dataset_file,output_forward_DG_file)

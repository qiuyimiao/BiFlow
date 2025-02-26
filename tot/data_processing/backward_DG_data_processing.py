import json
import os
from collections import defaultdict

dataset = "fairytale_test"
dataset_separated = "fairytale_test_separated"
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

def generated_same_question_id_backward(backward_DG_log_file):
    backward_QG_data = json_load(backward_DG_log_file)
    aggregated_data = defaultdict(lambda: {
        "context": "",
        #"gold_question": "",
        "generated_question": "",
        "answers": [],
        "generated_distractors": [],
        "correct_accuracy": []
    })
    for data in backward_QG_data:
        for step in data["steps"]:
            x = step["x"]
            question_id = x['question_id']
            aggregated_data[question_id]['context'] = x['context']
            aggregated_data[question_id]['generated_question'] = x['question']
            aggregated_data[question_id]['answers'] = x['answers']
            aggregated_data[question_id]['correct_accuracy'].append(step['correct_accuracy'])
            aggregated_data[question_id]['generated_distractors'].append(x['distractors'])
    result = [
        {
            "question_id": qid,
            "context": value['context'],
            "answers": value['answers'],
            "generated_question": value['generated_question'],
            "generated_distractors": value['generated_distractors'],
            "correct_accuracy": value['correct_accuracy']
        }
        for qid, value in aggregated_data.items()
    ]
    print(result)#test
    return result 

'''
def generate_backward_DG_data(backward_DG_log_data):
    
    filtered_data = []
    
    for entry in backward_DG_log_data:
        min_accuracy_index = entry['correct_accuracy'].index(min(entry['correct_accuracy']))
        
        best_distractors = entry['generated_distractors'][min_accuracy_index]
        
        filtered_data.append({
            "question_id": entry['question_id'],
            "context": entry['context'],
            "generated_question": entry['generated_question'],
            "generated_distractors": [best_distractors],  
            "correct_accuracy": [entry['correct_accuracy'][min_accuracy_index]],  
            "answers": entry['answers']
        })
    #json_dump(filtered_data,output_file)
    print(filtered_data)
    return filtered_data 
'''
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
                "votes": value['votes']
            }
            for qid, value in aggregated_data.items()
    ]
    #print(result)#test
    return result 

def get_combined_probability(forward_result, backward_result):
    combined_results = []
    alpha = 0.8
    beta = 1 - alpha
    forward_dict = {item["question_id"]: item for item in forward_result}
    
    for backward_item in backward_result:
        question_id = backward_item["question_id"]
        context = backward_item["context"]
        #gold_question = backward_item["gold_question"]
        generated_question = backward_item["generated_question"]
        generated_distractors = backward_item["generated_distractors"]
        answers = backward_item["answers"]
        correct_accuracy = backward_item["correct_accuracy"]
        
        if question_id in forward_dict:
            forward_votes = forward_dict[question_id]["votes"]
            print(forward_votes)#test
            print(correct_accuracy)#test
            if sum(forward_votes) != 0:
                forward_weights = [value/sum(forward_votes) for value in forward_votes]
            else:
                forward_weights = [0 for value in forward_votes]
            correct_accuracy = [value if value != 0 else 0.001 for value in correct_accuracy]
            reciprocal_sum = sum(1.0 / value for value in correct_accuracy if value != 0)
            backward_weights = [(1.0 / value) / reciprocal_sum for value in correct_accuracy]
            #backward_weights = [value/sum(correct_accuracy) for value in correct_accuracy]
            #backward_weights = [0 for value in correct_accuracy]
            #print(forward_weights)
            #print(backward_weights)
            combined_scores = []
            
            for i in range(len(generated_question)):
                if i < len(forward_votes) and i < len(correct_accuracy):
                    forward_weight = forward_weights[i]
                    backward_weight = backward_weights[i]
                    combined_score = forward_weight ** alpha * backward_weight ** beta
                    combined_scores.append(combined_score)
            
            combined_results.append({
                "question_id": question_id,
                "context": context,
                "generated_question": generated_question,
                "answers": answers,
                "generated_distractors": generated_distractors,
                "combined_scores": combined_scores
            })
    
    #print(combined_results)  # test
    return combined_results

def generate_backward_DG_data(combined_results,initial_dataset_file,output_file):
    
    initial_data = json_load(initial_dataset_file)
    filtered_data = []
    
    id_to_difficulty = {item['question_id']: item['difficulty'] for item in initial_data}
    id_gold_question_map = {item["question_id"]:item["question"] for item in initial_data}
    id_gold_distractor_map = {item["question_id"]:item["distractors"] for item in initial_data}
    for entry in combined_results:
        id = entry['question_id']
        max_combined_results_index = entry['combined_scores'].index(max(entry['combined_scores']))
        
        best_distractor = entry['generated_distractors'][max_combined_results_index]
        distractors = id_gold_distractor_map.get(id,None)
        filtered_data.append({
            "question_id": id,
            "context": entry['context'],
            "difficulty": id_to_difficulty.get(entry['question_id'], None),
            "gold_question": id_gold_question_map.get(id,None),
            "generated_question": entry['generated_question'], 
            "combined_scores": [entry['combined_scores'][max_combined_results_index]],  
            "answers": entry['answers'],
            "generated_distractors": best_distractor,
            "gold_distractors":[distractor['text'] for distractor in distractors] 
        })
    
    json_dump(filtered_data,output_file)

'''
def generated_with_gold_text(combined_results,initial_dataset_file, output_file):
    generated_data = []
    initial_data = json_load(initial_dataset_file)
    id_gold_question_map = {item["question_id"]:item["question"] for item in initial_data}
    id_gold_distractor_map = {item["question_id"]:item["distractors"] for item in initial_data}
    for item in combined_results:
        id = item['question_id']
        distractors = id_gold_distractor_map.get(id,None)
        new_entry = {
            "question_id": id,
            "context": item['context'],
            "gold_question": id_gold_question_map.get(id,None),
            "generated_question": item['generated_question'],
            "answers": item['answers'],
            "generated_distractors": item['generated_distractors'],  
            "gold_distractors":[distractor['text'] for distractor in distractors]
        }
        generated_data.append(new_entry)
    json_dump(generated_data,output_file)#
'''

if __name__ == '__main__':
    backward_log_json = "gpt-4o_1.0_sample6_vote1_greedy1_start0_end1785.json"
    forward_log_json = "gpt-4o_1.0_sample6_vote10_greedy3_start0_end595.json"
    input_backward_log_file = os.path.join(LOG_PATH, 'backward_DG', dataset_separated, backward_log_json)
    input_forward_log_file = os.path.join(LOG_PATH,'forward_DG', dataset, forward_log_json)
    input_initial_dataset = os.path.join(DATA_PATH,'forward_QG',drop_json)

    output_final_file = os.path.join(DATA_PATH, 'evaluation', drop_json)
    backward_result = generated_same_question_id_backward(input_backward_log_file)
    forward_result = generated_same_question_id_forward(input_forward_log_file)
    combine_result = get_combined_probability(forward_result, backward_result)
    #generated_with_gold_text(combine_result, input_initial_dataset, output_final_file)
    generate_backward_DG_data(combine_result,input_initial_dataset,output_final_file)

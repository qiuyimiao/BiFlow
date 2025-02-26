import json
import os
from tqdm import tqdm
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score  
from rouge import Rouge
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize  


dataset_name = "fairytale_test"  
drop_json = "fairytale_test_0_35.json"
drop_separated_json = "fairytale_test_separated.json"
DATA_PATH = os.path.join(os.path.dirname(__file__), '../..', 'data/')
LOG_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'logs/')
METRIC_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'metrics')


print(f'DATA_PATH: {DATA_PATH}')
print(f'LOG_PATH: {LOG_PATH}')
print(f'METRIC_PATH: {METRIC_PATH}')

rouge = Rouge()

def json_load(file_path):
    """Load a JSON file with UTF-8 encoding."""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def json_dump(data, file_path):
    """Dump JSON data into a file with UTF-8 encoding."""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def normalize(text):
    import re
    import string

    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(text))))

def get_batch(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def ceildiv(a, b):
    return -(a // -b)


def grade_score_with_batching(refs, hyps, bleurt, batch_size=64):
    scores = []
    num_batches = ceildiv(len(refs), batch_size)
    for ref_batch, hyp_batch in tqdm(zip(get_batch(refs, batch_size), get_batch(hyps, batch_size)), total=num_batches, desc="BLEURT Batch"):
        batch_scores = bleurt.compute(predictions=hyp_batch, references=ref_batch)
        scores.extend(batch_scores["scores"])
    return scores

def evaluate_metrics(data):
    """BLEU-1、BLEU-2、BLEU-4、ROUGE-L、METEOR、BERTScore 和 BLEURT"""
    refs = [normalize(item['gold_question']) for item in data]
    hyps = [normalize(item['generated_question'][0]) for item in data] 
    question_ids = [item.get('question_id', 'N/A') for item in data]  

    print("begin")

    # BLEU-1, BLEU-2, BLEU-4
    bleu1_scores = []
    bleu2_scores = []
    bleu4_scores = []
    for ref, hyp in tqdm(zip(refs, hyps), total=len(refs), desc="Calculating BLEU"):
        ref_tokens = [ref.split()]
        hyp_tokens = hyp.split()
        bleu1_score = sentence_bleu(ref_tokens, hyp_tokens, weights=(1, 0, 0, 0))  # BLEU-1
        bleu2_score = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.5, 0.5, 0, 0))  # BLEU-2
        bleu4_score = sentence_bleu(ref_tokens, hyp_tokens, weights=(0.25, 0.25, 0.25, 0.25))  # BLEU-4
        bleu1_scores.append(bleu1_score)
        bleu2_scores.append(bleu2_score)
        bleu4_scores.append(bleu4_score)

    # ROUGE-L
    rouge_scores = []
    for ref, hyp in tqdm(zip(refs, hyps), total=len(refs), desc="Calculating ROUGE-L"):
        scores = rouge.get_scores(hyp, ref)
        rouge_scores.append(scores[0]['rouge-l']['f'])

    # METEOR
    '''
    meteor_scores = []
    for ref, hyp in tqdm(zip(refs, hyps), total=len(refs), desc="Calculating METEOR"):
        ref_tokens = word_tokenize(ref)
        hyp_tokens = word_tokenize(hyp)
        meteor_score_value = meteor_score([ref_tokens], hyp_tokens)
        meteor_scores.append(meteor_score_value)
    '''
    # BERTScore
    P, R, F1 = bert_score(hyps, refs, lang="en")
    bert_precision = P.mean().item()
    bert_recall = R.mean().item()
    bert_f1 = F1.mean().item()

    # BLEURT
    # bleurt_scores = grade_score_with_batching(refs, hyps, bleurt)

    individual_results = [
        {
            'question_id': qid,  
            'gold_question': ref,
            'generated_question': hyp,
            'BLEU-1': bleu1_score,
            'BLEU-2': bleu2_score,
            'BLEU-4': bleu4_score,
            'ROUGE-L': rouge_score,
            'BERTScore Precision': bert_precision,
            'BERTScore Recall': bert_recall,
            'BERTScore F1': bert_f1
        }
        for qid, ref, hyp, bleu1_score, bleu2_score, bleu4_score, rouge_score in zip(
            question_ids, refs, hyps, bleu1_scores, bleu2_scores, bleu4_scores, rouge_scores
        )
    ]

    avg_bleu1 = sum(bleu1_scores) / len(bleu1_scores)
    avg_bleu2 = sum(bleu2_scores) / len(bleu2_scores)
    avg_bleu4 = sum(bleu4_scores) / len(bleu4_scores)
    avg_rouge = sum(rouge_scores) / len(rouge_scores)
    #avg_meteor = sum(meteor_scores) / len(meteor_scores)
    # avg_bleurt = sum(bleurt_scores) / len(bleurt_scores)

    return {
        'average_metrics': {
            'BLEU-1': avg_bleu1,
            'BLEU-2': avg_bleu2,
            'BLEU-4': avg_bleu4,
            'ROUGE-L': avg_rouge,
            'BERTScore Precision': bert_precision,
            'BERTScore Recall': bert_recall,
            'BERTScore F1': bert_f1
        },
        'individual_results': individual_results
    }

def main():
    input_forward_dataset_file = os.path.join(DATA_PATH, 'forward_DG', drop_json)

    data = json_load(input_forward_dataset_file)

    explicit_data = [item for item in data if item['difficulty'] == 'explicit']
    implicit_data = [item for item in data if item['difficulty'] == 'implicit']

    print("Evaluating overall performance...")
    overall_metrics = evaluate_metrics(data)
    print("Overall Metrics:", overall_metrics['average_metrics'])

    print("\nEvaluating explicit questions...")
    explicit_metrics = evaluate_metrics(explicit_data)
    print("Explicit Metrics:", explicit_metrics['average_metrics'])

    print("\nEvaluating implicit questions...")
    implicit_metrics = evaluate_metrics(implicit_data)
    print("Implicit Metrics:", implicit_metrics['average_metrics'])

    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_data = len(data)  
    log_filename = f"QG_a=0.35_{dataset_name}_{num_data}_{now}.log"
    log_file_path = os.path.join(LOG_PATH, 'evaluation', log_filename)
    os.makedirs(os.path.join(LOG_PATH, 'evaluation'), exist_ok=True)
    with open(log_file_path, 'w', encoding='utf-8') as log_file:
        log_file.write("Overall Metrics:\n")
        log_file.write(json.dumps(overall_metrics['average_metrics'], indent=4))
        log_file.write("\n\nExplicit Metrics:\n")
        log_file.write(json.dumps(explicit_metrics['average_metrics'], indent=4))
        log_file.write("\n\nImplicit Metrics:\n")
        log_file.write(json.dumps(implicit_metrics['average_metrics'], indent=4))

        log_file.write("\n\nIndividual Results:\n")
        for result in overall_metrics['individual_results']:
            log_file.write(json.dumps(result, indent=4))
            log_file.write("\n")

if __name__ == '__main__':
    main()
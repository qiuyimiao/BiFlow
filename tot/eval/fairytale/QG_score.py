import json
import os
from tqdm import tqdm
from datetime import datetime
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.meteor_score import meteor_score  # 新增 METEOR
from rouge import Rouge
from bert_score import score as bert_score
from nltk.tokenize import word_tokenize  # 新增用于 METEOR 的分词

# 数据集名称和路径
dataset_name = "fairytale_test"  # 后续：修改为args输入的全局变量
drop_json = "fairytale_test_0_35.json"
drop_separated_json = "fairytale_test_separated.json"
DATA_PATH = os.path.join(os.path.dirname(__file__), '../..', 'data/')
LOG_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'logs/')
METRIC_PATH = os.path.join(os.path.dirname(__file__), '../../..', 'metrics')

# 打印路径以确保正确
print(f'DATA_PATH: {DATA_PATH}')
print(f'LOG_PATH: {LOG_PATH}')
print(f'METRIC_PATH: {METRIC_PATH}')

# 加载评估指标
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
    """对文本进行归一化处理：小写化、去标点、去冠词、去多余空格"""
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
    """将数据分批处理"""
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]

def ceildiv(a, b):
    """实现向上取整的除法"""
    return -(a // -b)






def grade_score_with_batching(refs, hyps, bleurt, batch_size=64):
    """批量计算 BLEURT 分数"""
    scores = []
    num_batches = ceildiv(len(refs), batch_size)
    for ref_batch, hyp_batch in tqdm(zip(get_batch(refs, batch_size), get_batch(hyps, batch_size)), total=num_batches, desc="BLEURT Batch"):
        batch_scores = bleurt.compute(predictions=hyp_batch, references=ref_batch)
        scores.extend(batch_scores["scores"])
    return scores

def evaluate_metrics(data):
    """计算 BLEU-1、BLEU-2、BLEU-4、ROUGE-L、METEOR、BERTScore 和 BLEURT 分数"""
    refs = [normalize(item['gold_question']) for item in data]
    hyps = [normalize(item['generated_question'][0]) for item in data]  # 取第一个生成的问题
    question_ids = [item.get('question_id', 'N/A') for item in data]  # 获取 question_id，如果不存在则默认为 'N/A'

    print("开始计算")

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
    P, R, F1 = bert_score(hyps, refs, lang="en")  # 英文使用 "en"
    bert_precision = P.mean().item()
    bert_recall = R.mean().item()
    bert_f1 = F1.mean().item()

    # BLEURT（批量处理）
    # bleurt_scores = grade_score_with_batching(refs, hyps, bleurt)

    # 每条数据的评估结果
    individual_results = [
        {
            'question_id': qid,  # 添加 question_id
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

    # 计算平均分
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

    # 加载 JSON 数据
    data = json_load(input_forward_dataset_file)

    # 按照 difficulty 分类数据
    explicit_data = [item for item in data if item['difficulty'] == 'explicit']
    implicit_data = [item for item in data if item['difficulty'] == 'implicit']

    # 计算整体评估指标
    print("Evaluating overall performance...")
    overall_metrics = evaluate_metrics(data)
    print("Overall Metrics:", overall_metrics['average_metrics'])

    # 计算 explicit 的评估指标
    print("\nEvaluating explicit questions...")
    explicit_metrics = evaluate_metrics(explicit_data)
    print("Explicit Metrics:", explicit_metrics['average_metrics'])

    # 计算 implicit 的评估指标
    print("\nEvaluating implicit questions...")
    implicit_metrics = evaluate_metrics(implicit_data)
    print("Implicit Metrics:", implicit_metrics['average_metrics'])

    # log写入:
    # 获取当前日期时间，格式化为文件名的一部分
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_data = len(data)  # 数据条数
    log_filename = f"QG_a=0.35_{dataset_name}_{num_data}_{now}.log"
    log_file_path = os.path.join(LOG_PATH, 'evaluation', log_filename)
    # 确保日志文件夹存在
    os.makedirs(os.path.join(LOG_PATH, 'evaluation'), exist_ok=True)
    # 打开日志文件并打印评估结果
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
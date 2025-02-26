import itertools
import numpy as np
from functools import partial
from tot.models import gpt

def get_value(task, x, y, n_evaluate_sample, cache_value=True): #no use
    value_prompt = task.value_prompt_wrap(x, y)
    if cache_value and value_prompt in task.value_cache:
        return task.value_cache[value_prompt]
    value_outputs = gpt(value_prompt, n=n_evaluate_sample, stop=None)
    value = task.value_outputs_unwrap(x, y, value_outputs)
    if cache_value:
        task.value_cache[value_prompt] = value
    return value

def get_values(task, x, ys, n_evaluate_sample, cache_value=True): #no use
    values = []
    local_value_cache = {}
    for y in ys:  # each partial output
        if y in local_value_cache:  # avoid duplicate candidates
            value = 0
        else:    
            value = get_value(task, x, y, n_evaluate_sample, cache_value=cache_value)
            local_value_cache[y] = value
        values.append(value)
    return values

def get_votes(task, x, ys, n_evaluate_sample):                          
    vote_prompts = task.vote_prompt_wrap(x, ys)
    vote_outputs =[]
    for vote_prompt in vote_prompts:
        vote_outputs.append(gpt(vote_prompt, n=n_evaluate_sample, stop=None))
    values = task.vote_outputs_unwrap(vote_outputs, len(ys))
    return values

def get_proposals(task, x, y): #no use
    propose_prompt = task.propose_prompt_wrap(x, y)
    proposals = gpt(propose_prompt, n=1, stop=None)[0].split('\n')
    return [y + _ + '\n' for _ in proposals]

def get_samples(task, x, y, n_generate_sample, prompt_sample, stop):    
    if prompt_sample == 'standard': 
        prompt = task.standard_prompt_wrap(x, y)
    elif prompt_sample == 'cot':
        prompt = task.cot_prompt_wrap(x, y)
    else:
        raise ValueError(f'prompt_sample {prompt_sample} not recognized')
    samples = gpt(prompt, n=n_generate_sample, stop=stop) 
    return [y + _ for _ in samples]

def solve(args, task, idx, to_print=True):
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x_with_ans = task.get_input_with_ans(idx)  # input: context+question
    x_without_ans = task.get_input_without_ans(idx)
    x_with_all = task.get_input_with_all(idx)
    ys = ['']  # current output candidates 
    infos = []
    for step in range(task.steps):              
        # generation 
        if args.method_generate == 'sample':    
            new_ys = [get_samples(task, x_without_ans, y, args.n_generate_sample, prompt_sample=args.prompt_sample, stop=task.stops[step]) for y in ys]
        elif args.method_generate == 'propose':
            new_ys = [get_proposals(task, x_without_ans, y) for y in ys]
        new_ys = list(itertools.chain(*new_ys))
        ids = list(range(len(new_ys)))

        # evaluation
        if args.method_evaluate == 'vote':      
            values = get_votes(task, x_with_ans, new_ys, args.n_evaluate_sample)
        elif args.method_evaluate == 'value':
            values = get_values(task, x_with_ans, new_ys, args.n_evaluate_sample)

        # selection 
        '''if args.method_select == 'sample':
            ps = np.array(values) / sum(values)
            select_ids = np.random.choice(ids, size=args.n_select_sample, p=ps).tolist()
        elif args.method_select == 'greedy':    #选这个 贪婪
            select_ids = sorted(ids, key=lambda x: values[x], reverse=True)[:args.n_select_sample]#从排序后,取出前 args.n_select_sample个索引
        select_new_ys = [new_ys[select_id] for select_id in select_ids]'''

        # log
        #if to_print: 
            #sorted_new_ys, sorted_values = zip(*sorted(zip(new_ys, values), key=lambda x: x[1], reverse=True))
            #print(f'-- new_ys --: {sorted_new_ys}\n-- sol values --: {sorted_values}')
        
        infos.append({'step': step, 'x': x_with_all, 'ys': ys, 'new_ys': new_ys, 'correct_accuracy': values})
        #, 'select_new_ys': select_new_ys})
        #ys = select_new_ys
    
    '''if to_print: 
        print(ys)'''
    #return ys, {'steps': infos}
    return {'steps': infos}

def naive_solve(args, task, idx, to_print=True):   
    global gpt
    gpt = partial(gpt, model=args.backend, temperature=args.temperature)
    print(gpt)
    x = task.get_input_without_ans(idx)  # input
    ys = get_samples(task, x, '', args.n_generate_sample, args.prompt_sample, stop=None)
    return ys, {}
import os
import openai
from openai import OpenAI
import backoff 
import time

completion_tokens = prompt_tokens = 0

#api_key = os.getenv("OPENAI_API_KEY", "")

client = OpenAI(
    base_url='',#your url here
    api_key=''#your key here
    )

@backoff.on_exception(backoff.expo, openai.OpenAIError)
def completions_with_backoff(**kwargs):
    return client.chat.completions.create(**kwargs)

def gpt(prompt, model="gpt-4o", temperature=0.7, max_tokens=3000, n=1, stop=None) -> list:             
    messages = [{"role": "user", "content": prompt}]
    collected_outputs = []  
    while True:
        try:
            outputs = chatgpt(messages, model=model, temperature=temperature, max_tokens=max_tokens, n=n, stop=stop)
            if outputs and all(outputs): 
                return outputs
            elif len(collected_outputs) >= n: 
                return collected_outputs[:n]
            else:
                collected_outputs.extend([output for output in outputs if output is not None])
                print("There are empty values in the outputs. Storing the non-empty parts and retrying...")
                print(f"Currently collected content: {collected_outputs}")
                time.sleep(3)
        except Exception as e:
            print(f"An error occurred during the call: {e}. Retrying...")
    
def chatgpt(messages, model="gpt-4o", temperature=0.7, max_tokens=3000, n=1, stop=None) -> list:
    global completion_tokens, prompt_tokens
    outputs = []
    while n > 0:
        cnt = min(n, 20)
        n -= cnt
        res = completions_with_backoff(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens, n=cnt, stop=stop)
        outputs.extend([choice.message.content for choice in res.choices])
    return outputs
    
def gpt_usage(backend="gpt-4o"):
    global completion_tokens, prompt_tokens
    if backend == "gpt-4o":
        cost = completion_tokens / 1000 * 0.06 + prompt_tokens / 1000 * 0.03
    elif backend == "gpt-3.5-turbo":
        cost = completion_tokens / 1000 * 0.002 + prompt_tokens / 1000 * 0.0015
    return {"completion_tokens": completion_tokens, "prompt_tokens": prompt_tokens, "cost": cost}

standard_prompt = '''
Based on the given context and answer set, propose a question for the given answer set that can form a complete question and answer with the given answer in the context.
The result of answering the question will not contain any information beyond the given correct answer.
Try to use expressions that are the same or similar to those in the context.
 The context and answer group content are as follows: {input}
'''
standard_prompt_shot = '''
Based on the given context and answer set, propose a question for the given answer set that can form a complete question and answer with the given answer in the context.
The result of answering the question will not contain any information beyond the given correct answer.
Try to use expressions that are the same or similar to those in the context.
For example: {example}
 The context and answer group content are as follows: {input}
'''



cot_prompt_0 = '''
Based on the given context and answer set, propose a question for the given answer set that can form a complete question and answer with the given answer in the context. 
The result of answering the question will not contain any information beyond the given correct answer.
The context and answer group content are as follows: {input}

Based on the similarity of related events in the context of the answer entity, make a plan then write the question.
Simply mention the general content of the problem in the plan
Your output should be of the following format:

Plan:
Your plan here.(less than 50 words)

Question:
Your question here.(Try to use expressions that are the same or similar to those in the context)
'''

cot_prompt = '''
Based on the given context and answer set, propose a question for the given answer set that can form a complete question and answer with the given answer in the context. 
The result of answering the question will not contain any information beyond the given correct answer.
The context and answer group content are as follows: {input}

Based on the similarity of related events in the context of the answer entity, analyze the answer type to identify the corresponding question words, make a plan then write the question.
Simply mention the general content of the problem in the plan.
Your output should be of the following format:

Plan:
Your plan here.(less than 50 words)

Question:
Your question here.(Try to use expressions that are the same or similar to those in the context)
'''

vote_prompt = '''
The context and answer group content are as follows: {input}
Given a context and an answer set (including multiple answers), given several questions, 
determine which question is most appropriate to ask in the context of the given answer set. 
Analyze the context, answers, and each question in detail(less than 100 words), and then draw a conclusion on the last line "best question is s", where s the integer id of the choice.
Strictly follow the capitalization and content format of 'best question is s'
choices are as follows:
'''

compare_prompt = '''
The context and answer group content are as follows: {input}
Briefly analyze the degree of matching between the question and {context, answer set}.
Conclude in the last line, "The question that matches more closely is 1", "The question that matches the context and answer more closely is 2", or "The appropriateness of these two questions is similar".
'''


score_prompt = '''
The context, question and answer group content are as follows: {input}
Analyze the following context, question, and answer group, and then draw the conclusion in the last line 
that "Therefore, the score for the matching score between the question and 'context, answer group' is s", where s is an integer from 1 to 10
'''

standard_prompt = '''
You are a third grade non English major who answers questions based on the given context and reading comprehension.
Answer questions with the vocabulary and thinking skills that children should possess, and make appropriate mistakes based on the difficulty level.
The context and problem group content are as follows:  {input}
Your output should be in the following format:
Your answer here(Only the core part of the answer, no complete sentence is required)
'''

cot_prompt = '''
You are a third grade non English major who answers questions based on the given context and reading comprehension.
Answer questions with the vocabulary and thinking skills that children should possess, and make appropriate mistakes based on the difficulty level.
The context and problem group content are as follows: {input}
Your output should be in the following format:
answer:
Your answer here(Only the core part of the answer, no complete sentence is required)
'''
#plan:
#The logic of children answering questions, if the question is answered incorrectly, describe the possible reasons for the mistake.(less than 30 words)

vote_prompt = '''
The context and correct answer to a certain question are as follows: {input}
Given a student's answer to a question, determine the correctness of their answer.
Judging by the standard of Q&A questions, 
the answer can be judged as correctthe if the similarity between student's answer and the correct answer exceeds 80%.
Analyze the comparison between student's answer and correct answer(less than 20 words), 
then draw the conclusion in the last line that 'the student's answer is s', 
where s is an integer of 0 or 1, 0 represents error, and 1 represents correctness.
Strictly follow the capitalization and content format of 'the student's answer is s'.
The answers given by the students are as follows:
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
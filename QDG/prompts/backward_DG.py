standard_prompt = '''
Select all correct options based on the given context, multiple answer questions, and multiple options (including multiple correct answers and multiple distractors).
Your reply only includes the answer set of the question, with correct answers separated by commas and no need to explain any reason.
You play the role of a non-native English speaking middle school student, 
answering reading comprehension questions with limited English language knowledge, and there is a probability of making mistakes.
The context and problem group content are as follows: {input}
Your output should be in the following format:
answer:
Your answer here.
'''

cot_prompt = '''
You are a third grade non English major who answers questions based on the given context and reading comprehension.
Choose the answers with the vocabulary and thinking skills that children should possess, and make appropriate mistakes based on the difficulty level.
Your reply only includes the answer set of the question, with correct answers separated by commas and no need to explain any reason.
Note: You play the role of an exam student, and you will make mistakes when dealing with strong interference items.
The context and problem group content are as follows: {input}
Your output should be in the following format:
answer:
Your answer here.(for example: "united states, germans, english, irish")
'''



vote_prompt = '''
The correct answers and incorrect distractors of a multiple-answer question are as follows: {input}
Given a student's answer to a question, determine the correctness of their answer. 

The principle of judging correctness: 
Only when the answer given by student is completely same to the correct answer, 
with only a difference in answer order, can it be judged as correct. 
If student's answer doesn't fully cover all correct answers or contains any interfering items, it should be judged as incorrect.

Compare students' answers with the correct answers,then draw the conclusion in the last line that 'The student's answer is s',
where s is an integer of 0 or 1, 0 represents error, and 1 represents correctness. No need for further explanation of reasons.
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

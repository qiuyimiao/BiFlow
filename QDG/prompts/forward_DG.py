standard_prompt = '''
Design 3 distractors for the given answer set based on the given context, question, and answer set. 
The designed distractors can be combined with the given contextual paragraph, question, and answer to form a multi answer reading comprehension question.
The distractors should appear reasonable but not be included in the correct answer range. Prioritize searching from the given context.
The context, question, and answer groups are as follows:   {input}
here is example 1:
"context": "Old Dragonbeard must have been a master swordsman standing midway between those of the first and of the second order. Molo, however, of whom this story tells, was a sword hero.\n\nAt that time there lived a young man named Tsui, whose father was a high official and the friend of the prince. And the father once sent his son to visit his princely friend, who was ill. The son was young, handsome and gifted. He went to carry out his father's instructions. When he entered the prince's palace, there stood three beautiful slave girls, who piled rosy peaches into a golden bowl, poured sugar over them and presented them to him. After he had eaten he took his leave, and his princely host ordered one of the slave girls, Rose-Red by name, to escort him to the gate. As they went along the young man kept looking back at her. And she smiled at him and made signs with her fingers. First she would stretch out three fingers, then she would turn her hand around three times, and finally she would point to a little mirror which she wore on her breast. When they parted she whispered to him: \"Do not forget me!\"",
    "question": "Who was a master swordsman standing midway between those of the first and of the second order?",
    "answers": "Old Dragonbeard"
    "generated_distractors": [ "Molo","Tsui","the prince"]
example 2:
    "context": "Many, many years ago there lived a good old man who had a wen like a\ntennis-ball growing out of his right cheek. This lump was a great\ndisfigurement to the old man, and so annoyed him that for many years he\nspent all his time and money in trying to get rid of it. He tried\neverything he could think of. He consulted many doctors far and near,\nand took all kinds of medicines both internally and externally. But it\nwas all of no use. The lump only grew bigger and bigger till it was\nnearly as big as his face, and in despair he gave up all hopes of ever\nlosing it, and resigned himself to the thought of having to carry the\nlump on his face all his life.",
    "question": "How did the man feel about his wen?",
    "difficulty": "explicit",
    "answers": "annoyed",
    "distractors": ["frustrated", "disheartened","helpless"]
Your output: 
Distractor 1: xxx(only the distractor part, no reason or other things).
Distractor 2: xxx
Distractor 3: xxx
'''


standard_prompt_shot = '''
Based on the given context, question and answer set, Design 3 distractors.
The result of answering the question will not contain any information beyond the given correct answer.
Try to use expressions that are the same or similar to answers.
For example: {example}
 The context and answer group content are as follows: {input}
Your output: 
Distractor 1: xxx(only the distractor part, no reason or other things).
Distractor 2: xxx
Distractor 3: xxx
'''

cot_prompt = '''
Design 3 best distractors for the given answer based on the given context, question, and answer. 
The designed distractors can be combined with the given information to form a multi answer reading comprehension question.
The distractors should appear reasonable but not be included in the correct answer range. Prioritize searching from the given context.
The context, question, and answer groups are as follows:  {input}

Based on all the aspects that distractors should involve, develop an distractor generation plan and then write the distractors.
Your output should be in the following format:

Plan : Your plan for 3 distractors here. (less than 50 words, reasoning from the type and structure of the answer)

Distractors:(Maintain strict consistency)
Distractor 1: a distractor(only the distractor part, no reason or other thing or characters like '*').
Distractor 2: a distractor.
Distractor 3: a distractor.
'''


vote_prompt = '''
The context, question and answer group content are as follows: {input}
Given a context, question and an answer set (including multiple answers), given several distractors, 
determine which three distractors are the most reasonable as well as incorrect for the current given data, and have the best interference.
Analyze the context, question and answers in detail, and then draw a conclusion on the last line. You MUST end up with "best distractor set is s", where s the integer id of the choice, begin from 1.
choices are as follows:
'''

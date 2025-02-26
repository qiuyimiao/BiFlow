# BiFlow: Teacher-Student Bidirectional Reasoning Enhances MCQ Generation and Distractor Quality
## Project Overview
BiFlow is a novel framework designed to enhance the generation of high-quality Multiple Choice Questions (MCQs) and distractors by integrating bidirectional reasoning perspectives from both teachers and students. The framework leverages a combination of teacher reasoning, which generates contextually relevant questions and plausible distractors, and student reasoning, which evaluates the clarity of questions and the misleading nature of distractors. Additionally, BiFlow introduces PathFinder, a mechanism that employs breadth-first search and Chain-of-Thought (CoT) strategies to explore diverse reasoning paths, improving both the quality and diversity of generated questions and distractors.

The project also extends the FairytaleQA dataset to FairytaleMCQ by adding high-quality distractors, providing a robust benchmark for evaluating MCQ generation models. Experimental results demonstrate that BiFlow outperforms existing methods, particularly in generating complex questions and high-quality distractors for long-text scenarios.
## Key Features
### Bidirectional Reasoning: 
Combines teacher and student reasoning to ensure pedagogical soundness and cognitive challenge.
### PathFinder: 
Utilizes breadth-first search and Chain-of-Thought (CoT) strategies to explore diverse reasoning paths.
### FairytaleMCQ Dataset: 
Extends the FairytaleQA dataset with high-quality distractors, enabling robust evaluation of MCQ generation models.
### Human-Agent Collaboration: 
Incorporates human verification to ensure the quality of generated distractors.
## Method
### 1. Question Generation (QG)
Teacher Reasoning: Generates candidate questions using PathFinder, which explores multiple reasoning paths via CoT and breadth-first search.
Student Reasoning: Evaluates the clarity of generated questions by simulating a student's problem-solving process using an LLM.
Teacher-Student Combination: Combines teacher and student evaluations using a weighted geometric mean to select the final question.
### 2. Distractor Generation (DG)
Teacher Reasoning: Generates plausible distractors using PathFinder, ensuring alignment with the context and answer type.
Student Reasoning: Evaluates the plausibility and misleading nature of distractors by simulating a student's problem-solving process.
Teacher-Student Combination: Combines teacher and student evaluations to select the final set of distractors.
### 3. PathFinder
Reasoning Path Generation: Uses CoT prompting to generate multiple reasoning paths for question and distractor generation.
Path Evaluation: Evaluates the quality of reasoning paths using a zero-shot evaluation prompt.
Question/Distractor Generation: Generates candidate questions or distractors along the best reasoning paths.

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
## Dataset
The FairytaleMCQ dataset is an extension of the FairytaleQA dataset, which consists of 10,580 explicit and implicit questions derived from 278 children-friendly stories. The dataset is enriched with high-quality distractors generated through a human-agent collaborative annotation pipeline.
Training Set: 8,548 QA pairs
Validation Set: 1,025 QA pairs
Test Set: 1,007 QA pairs (with distractors)
## Running the Code
### Prerequisites
Python 3.8 or higher
PyTorch
Transformers library
OpenAI GPT-4 API (for distractor generation and evaluation)
### Running the Model
This section provides detailed instructions on how to run the code for **question generation (QG)** and **distractor generation (DG)** using the BiFlow framework. The code is divided into several steps, each corresponding to a specific part of the pipeline
#### 1. **Question Generation (QG)**

##### 1.1 **Forward Question Generation (Forward_QG)**

The forward question generation step generates multiple candidate questions using the **PathFinder** mechanism, which combines breadth-first search and Chain-of-Thought (CoT) strategies. The top-3 questions are selected based on evaluation scores.

##### Command:
```bash
python run.py --task forward_QG --task_start_index 0 --task_end_index 1007 --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample 6 --n_evaluate_sample 10 --n_select_sample 3 --prompt_sample cot --temperature 1.0


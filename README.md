# BiFlow: Teacher-Student Bidirectional Reasoning Enhances MCQ Generation and Distractor Quality
## Project Overview
![image](https://github.com/user-attachments/assets/da5cb17f-9714-4d09-813a-4852b95ec661)
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
The forward question generation step generates multiple candidate questions using the **PathFinder** mechanism, which combines breadth-first search and Chain-of-Thought (CoT) strategies. The top-b questions are selected based on evaluation scores.
Command:
```bash
python run.py --task forward_QG --task_start_index xx --task_end_index xx --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample xx --n_evaluate_sample xx --n_select_sample b --prompt_sample xx --temperature 1.0
```

##### 1.2 Forward Data Processing
After generating the questions, the results need to be processed to extract the top-3 questions and store them in a structured format.
Command:
Run the following script directly:
```bash
python data_processing/forwardQG_data_processing.py
```
##### 1.3 Backward Question Generation (Backward_QG)
The backward question generation step evaluates the generated questions by simulating a student's problem-solving process. Each question is answered k times, and the accuracy of the answers is calculated to determine the quality of the question.
Command:
```bash
python run_backward_QG.py --task backward_QG --task_start_index xx --task_end_index xxx --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample xx --n_evaluate_sample xx --n_select_sample xx --prompt_sample cot --temperature 1.0
```
##### 1.4 Backward Data Processing
Command:
Run the following script directly:
```bash
python data_processing/backward_QG_data_processing.py
```

#### 2. Distractor Generation (DG)
##### 2.1 Forward Distractor Generation (Forward_DG)
The forward distractor generation step generates multiple candidate distractors using the PathFinder mechanism. The top-b distractors are selected based on evaluation scores.
Command:
```bash
python run.py --task forward_DG --task_start_index xx --task_end_index xx --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample x --n_evaluate_sample x --n_select_sample b --prompt_sample cot --temperature 1.0
```
#### 2.2 Forward DG Data Processing
After generating the distractors, the results need to be processed to extract the top-3 distractors and store them in a structured format.
Command:
Run the following script directly:
```bash
python data_processing/forward_DG_data_processing.py
```
#### 2.3 Backward Distractor Generation (Backward_DG)
The backward distractor generation step evaluates the generated distractors by simulating a student's problem-solving process. The order of answers and distractors is shuffled, and each question is answered 6 times to calculate the accuracy.
Command:
```bash
python run_backward_DG.py --task backward_DG --task_start_index xx --task_end_index xx --method_generate sample --method_evaluate vote --method_select greedy --n_generate_sample xx --n_evaluate_sample x --n_select_sample x --prompt_sample cot --temperature 1.0
```

#### 2.4 Backward DG Data Processing
```bash
data_processing/backward_DG_data_processing.py
```

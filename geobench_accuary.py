import json
from collections import defaultdict

# File paths
#qa_file = "/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Single/qa.json"
#pred_file = "/home/wenhan/Projects/sesame/pred_genbench/lisat_0223_v1/pred_geobench_single.json"
qa_file = "/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Temporal/qa.json"
pred_file = "/home/wenhan/Projects/sesame/pred_geobench_temporal.json"
# Load the QA dataset
with open(qa_file, "r") as f:
    qa_data = json.load(f)

# Load the prediction dataset
with open(pred_file, "r") as f:
    pred_data = [json.loads(line) for line in f]

# Map question_id to task categories
task_mapping = {item["question_id"]: item["task"] for item in qa_data}

# Calculate accuracy per task
task_accuracy = defaultdict(lambda: {"correct": 0, "total": 0})

for pred in pred_data:
    question_id = pred["question_id"]
    task = task_mapping.get(question_id, "Unknown Task")
    
    task_accuracy[task]["total"] += 1
    if pred["is_correct"]:
        task_accuracy[task]["correct"] += 1

# Compute and display accuracy per task
accuracy_results = {
    task: round((data["correct"] / data["total"]) * 100, 2) if data["total"] > 0 else 0
    for task, data in task_accuracy.items()
}

# Print results
for task, accuracy in accuracy_results.items():
    print(f"Task: {task} - Accuracy: {accuracy}%")

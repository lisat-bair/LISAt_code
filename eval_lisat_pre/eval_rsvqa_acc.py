import json
from collections import defaultdict

# ==== File Paths ====
PRED_FILE = "./captioning_dir/lisat/lisat_RSVQA_LR_answer.jsonl"
GT_FILE = "./vqa_caption_ans/RSVQA_LR.jsonl"

# ==== Load predictions ====
pred_dict = {}
with open(PRED_FILE, 'r') as f:
    for line in f:
        obj = json.loads(line)
        pred_dict[obj['question_id']] = obj['text'].strip().lower()

# ==== Load ground truth answers + categories ====
correct_dict = {}
category_counts = defaultdict(lambda: {'correct': 0, 'total': 0})

with open(GT_FILE, 'r') as f:
    for line in f:
        obj = json.loads(line)
        qid = obj['question_id']
        answer = obj['answer'].strip().lower()
        category = obj['category']
        correct_dict[qid] = {'answer': answer, 'category': category}

# ==== Compute category-wise accuracy ====
for qid, gt in correct_dict.items():
    category = gt['category']
    correct_answer = gt['answer']
    pred = pred_dict.get(qid, "").strip().lower()
    
    if pred == correct_answer:
        category_counts[category]['correct'] += 1
    category_counts[category]['total'] += 1

# ==== Report accuracy per category ====
print("Category-wise Accuracy on RSVQA_LR:")
print("-------------------------------------")
for category in sorted(category_counts):
    counts = category_counts[category]
    accuracy = (counts['correct'] / counts['total']) * 100 if counts['total'] > 0 else 0
    print(f"{category:20s}: {accuracy:.2f}% ({counts['correct']}/{counts['total']})")

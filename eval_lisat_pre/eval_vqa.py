import json

pred_dict = {}
correct_dict = {}
category_counts = {}

with open('/home/wenhan/Projects/sesame/captioning_dir/lisat_0221_v1/lisat_0221_v1_RSVQA_LR_answer.jsonl', 'r') as pred_file:
    for line in pred_file:
        pred = json.loads(line)
        question_id = pred['question_id']
        pred_text = pred['text']
        pred_dict[question_id] = pred_text

with open('/home/wenhan/lab_projects/LISAT_PRE/vqa_caption_ans/RSVQA_LR.jsonl', 'r') as correct_file:
    for line in correct_file:
        correct = json.loads(line)
        question_id = correct['question_id']
        answer = correct['answer']
        category = correct['category']
        correct_dict[question_id] = {'answer': answer, 'category': category}
        if category not in category_counts:
            category_counts[category] = {'correct': 0, 'total': 0}

for question_id, pred_text in pred_dict.items():
    if question_id in correct_dict:
        correct_info = correct_dict[question_id]
        correct_answer = correct_info['answer']
        category = correct_info['category']
        pred_text_normalized = pred_text.strip().lower()
        correct_answer_normalized = correct_answer.strip().lower()
        if pred_text_normalized == correct_answer_normalized:
            category_counts[category]['correct'] += 1
        category_counts[category]['total'] += 1

for category, counts in category_counts.items():
    correct = counts['correct']
    total = counts['total']
    accuracy = (correct / total) * 100 if total > 0 else 0
    print(f"{category}: {accuracy:.2f}%")
import json
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pycocoevalcap.cider.cider_scorer import CiderScorer
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate CIDEr score for image captioning.')
    parser.add_argument('--prediction_file', type=str, required=True, help='Path to the prediction JSONL file.')
    parser.add_argument('--answer_file', type=str, required=True, help='Path to the answer JSONL file.')
    args = parser.parse_args()

    predictions = []
    references_dict = {}

    with open(args.prediction_file, 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            data = json.loads(line.strip())
            question_id = data['question_id']
            text = data['text']
            predictions.append({'image_id': question_id, 'caption': text})

    with open(args.answer_file, 'r', encoding='utf-8') as ref_file:
        for line in ref_file:
            data = json.loads(line.strip())
            question_id = data['question_id']
            answers = data['answer']
            if isinstance(answers, list):
                references_dict[question_id] = answers
            else:
                references_dict[question_id] = [answers]

    cider_scorer = CiderScorer(n=4, sigma=6.0)

    for item in predictions:
        image_id = item['image_id']
        hypo = item['caption']
        ref = references_dict.get(image_id, [])
        if not ref:
            print(f"No references found for image_id {image_id}")
            continue
        cider_scorer += (hypo, ref)

    cider_score, _ = cider_scorer.compute_score()

    print("CIDEr Score:", cider_score)

if __name__ == "__main__":
    main()

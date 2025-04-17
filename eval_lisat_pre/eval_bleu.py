import json
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pycocoevalcap.bleu.bleu import Bleu

def main():
    parser = argparse.ArgumentParser(description='Evaluate BLEU score for image captioning.')
    parser.add_argument('--prediction_file', type=str, required=True, help='Path to the prediction JSONL file.')
    parser.add_argument('--answer_file', type=str, required=True, help='Path to the answer JSONL file.')
    args = parser.parse_args()

    predictions_dict = {}
    references_dict = {}

    with open(args.prediction_file, 'r', encoding='utf-8') as pred_file:
        for line in pred_file:
            data = json.loads(line.strip())
            question_id = data['question_id']
            text = data['text']
            predictions_dict[question_id] = [text]

    with open(args.answer_file, 'r', encoding='utf-8') as ref_file:
        for line in ref_file:
            data = json.loads(line.strip())
            question_id = data['question_id']
            answers = data['answer']
            if isinstance(answers, list):
                references_dict[question_id] = answers
            else:
                references_dict[question_id] = [answers]

    bleu_scorer = Bleu(n=4)  

    score, scores = bleu_scorer.compute_score(references_dict, predictions_dict)

    for i, s in enumerate(score):
        print(f"Bleu-{i+1}: {s:.6f}")

if __name__ == "__main__":
    main()

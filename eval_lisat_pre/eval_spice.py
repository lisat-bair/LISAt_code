import json
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pycocoevalcap.spice.spice import Spice

def main():
    parser = argparse.ArgumentParser(description='Evaluate SPICE scores for image captioning.')
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

    res = {}
    gts = {}

    for question_id in predictions_dict.keys():
        if question_id in references_dict:
            res[question_id] = predictions_dict[question_id]  
            gts[question_id] = references_dict[question_id]   
        else:
            print(f"No reference found for question_id {question_id}")

    scorer = Spice()
    score, scores = scorer.compute_score(gts, res)
    print("SPICE Score:", score)

if __name__ == "__main__":
    main()

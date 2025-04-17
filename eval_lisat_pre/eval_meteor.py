import json
import argparse
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from pycocoevalcap.meteor.meteor import Meteor

def main():
    parser = argparse.ArgumentParser(description='Evaluate METEOR score for image captioning.')
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

    res = predictions_dict 
    gts = references_dict 

    meteor_scorer = Meteor()

    score, scores = meteor_scorer.compute_score(gts, res)
    print("METEOR Score:", score)

if __name__ == "__main__":
    main()

import json
import evaluate
import argparse

def main():
    parser = argparse.ArgumentParser(description='Evaluate ROUGE scores for image captioning.')
    parser.add_argument('--prediction_file', type=str, required=True, help='Path to the prediction JSONL file.')
    parser.add_argument('--answer_file', type=str, required=True, help='Path to the answer JSONL file.')
    args = parser.parse_args()

    rouge = evaluate.load('rouge')

    predictions_dict = {}
    references_dict = {}

    with open(args.prediction_file, 'r') as pred_file:
        for line in pred_file:
            data = json.loads(line)
            question_id = data['question_id']
            text = data['text']
            predictions_dict[question_id] = text

    with open(args.answer_file, 'r') as ref_file:
        for line in ref_file:
            data = json.loads(line)
            question_id = data['question_id']
            answers = data['answer']
            references_dict[question_id] = answers

    predictions = []
    references = []

    for question_id in sorted(predictions_dict.keys()):
        if question_id in references_dict:
            predictions.append(predictions_dict[question_id])
            references.append(references_dict[question_id])  
        else:
            print(f"No reference found for question_id {question_id}")

    results = rouge.compute(predictions=predictions, references=references, use_stemmer=True)

    print("ROUGE Scores:", results)

if __name__ == "__main__":
    main()

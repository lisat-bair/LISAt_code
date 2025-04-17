import re
import pandas as pd
import os

log_file = "./eval_lisat_pre/eval_results.log"
output_csv = "./eval_lisat_pre/eval_results_sorted.csv"

dataset_pattern = re.compile(r"Evaluating for (.+)\.\.\.")
bleu_pattern = re.compile(r"Bleu-(\d): ([\d.]+)")
meteor_pattern = re.compile(r"METEOR Score: ([\d.]+)")
rouge_pattern = re.compile(r"ROUGE Scores: .*?'rouge1': ([\d.]+), 'rouge2': ([\d.]+), 'rougeL': ([\d.]+), 'rougeLsum': ([\d.]+)")
cider_pattern = re.compile(r"CIDEr Score: ([\d.]+)")
spice_pattern = re.compile(r"SPICE Score: ([\d.]+)")

with open(log_file, "r") as f:
    lines = f.readlines()

results = []
current_dataset = None
bleu_scores = {}

for line in lines:
    dataset_match = dataset_pattern.search(line)
    if dataset_match:
        if current_dataset:
            results.append([
                current_dataset, bleu_scores.get(1, 0), bleu_scores.get(2, 0),
                bleu_scores.get(3, 0), bleu_scores.get(4, 0), meteor, rouge1, rouge2, rougeL, rougeLsum, cider, spice
            ])
        current_dataset = dataset_match.group(1)
        bleu_scores = {}

    bleu_match = bleu_pattern.search(line)
    if bleu_match:
        bleu_scores[int(bleu_match.group(1))] = float(bleu_match.group(2))

    meteor_match = meteor_pattern.search(line)
    if meteor_match:
        meteor = float(meteor_match.group(1))

    rouge_match = rouge_pattern.search(line)
    if rouge_match:
        rouge1, rouge2, rougeL, rougeLsum = map(float, rouge_match.groups())

    cider_match = cider_pattern.search(line)
    if cider_match:
        cider = float(cider_match.group(1))

    spice_match = spice_pattern.search(line)
    if spice_match:
        spice = float(spice_match.group(1))

if current_dataset:
    results.append([
        current_dataset, bleu_scores.get(1, 0), bleu_scores.get(2, 0),
        bleu_scores.get(3, 0), bleu_scores.get(4, 0), meteor, rouge1, rouge2, rougeL, rougeLsum, cider, spice
    ])

columns = ["Dataset", "Bleu-1", "Bleu-2", "Bleu-3", "Bleu-4", "METEOR",
           "ROUGE-1", "ROUGE-2", "ROUGE-L", "ROUGE-Lsum", "CIDEr", "SPICE"]

df = pd.DataFrame(results, columns=columns)

df_sorted = df.sort_values(by="CIDEr", ascending=False)

output_dir = os.path.dirname(output_csv)
if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)

df_sorted.to_csv(output_csv, index=False)

print(f"Evaluation results saved to {output_csv}")

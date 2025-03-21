#!/bin/bash

PREDICTION_BASE='/home/wenhan/Projects/sesame/captioning_dir/lisat_0223_v1'
ANSWER_BASE='/home/wenhan/Projects/sesame/vqa_caption_ans'
LOG_FILE="/home/wenhan/Projects/sesame/eval_lisat_pre/eval_results.log"
CSV_FILE="/home/wenhan/Projects/sesame/eval_lisat_pre/eval_results_sorted.csv"

> "$LOG_FILE"

DATASETS=("NWPU-Captions" "RSICD" "Sydney-Captions" "UCM-Captions")
# DATASETS=("UCM-Captions" "Sydney-Captions")
# DATASETS=("RSICD" "Sydney-Captions" "UCM-Captions")

for DATASET in "${DATASETS[@]}"
do
  echo "Evaluating for $DATASET..." | tee -a "$LOG_FILE"

  PREDICTION_FILE="$PREDICTION_BASE/lisat_0223_v1_${DATASET}_answer.jsonl"
  ANSWER_FILE="$ANSWER_BASE/${DATASET}.jsonl"

  echo "Evaluating BLEU Score..." | tee -a "$LOG_FILE"
  python /home/wenhan/Projects/sesame/eval_lisat_pre/eval_bleu.py --prediction_file "$PREDICTION_FILE" --answer_file "$ANSWER_FILE" | tee -a "$LOG_FILE"

  echo "Evaluating METEOR Score..." | tee -a "$LOG_FILE"
  python /home/wenhan/Projects/sesame/eval_lisat_pre/eval_meteor.py --prediction_file "$PREDICTION_FILE" --answer_file "$ANSWER_FILE" | tee -a "$LOG_FILE"

  echo "Evaluating ROUGE Scores..." | tee -a "$LOG_FILE"
  python /home/wenhan/Projects/sesame/eval_lisat_pre/eval_rouge.py --prediction_file "$PREDICTION_FILE" --answer_file "$ANSWER_FILE" | tee -a "$LOG_FILE"

  echo "Evaluating CIDEr Score..." | tee -a "$LOG_FILE"
  python /home/wenhan/Projects/sesame/eval_lisat_pre/eval_cider.py --prediction_file "$PREDICTION_FILE" --answer_file "$ANSWER_FILE" | tee -a "$LOG_FILE"

  echo "Evaluating SPICE Score..." | tee -a "$LOG_FILE"
  python /home/wenhan/Projects/sesame/eval_lisat_pre/eval_spice.py --prediction_file "$PREDICTION_FILE" --answer_file "$ANSWER_FILE" | tee -a "$LOG_FILE"

  echo "Completed evaluation for $DATASET." | tee -a "$LOG_FILE"
done

python /home/wenhan/Projects/sesame/eval_lisat_pre/eval_results_csv.py
echo "Evaluation results saved to $CSV_FILE"

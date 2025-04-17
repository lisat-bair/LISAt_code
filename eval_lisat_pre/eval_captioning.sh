#!/bin/bash

# Base directories for predictions and answers
PREDICTION_BASE="./captioning_dir/lisat"
ANSWER_BASE="./dataset/vqa_caption_ans"
EVAL_SCRIPT_DIR="./eval_scripts"

# Output log and CSV summary
LOG_FILE=".eval_results.log"
CSV_FILE=".eval_results_sorted.csv"

# Create output directory if it doesn't exist
mkdir -p "$(dirname "$LOG_FILE")"

# Clear previous log
> "$LOG_FILE"

# Datasets to evaluate
DATASETS=("NWPU-Captions" "RSICD" "Sydney-Captions" "UCM-Captions")

# Loop through datasets and evaluate
for DATASET in "${DATASETS[@]}"; do
  echo "====================" | tee -a "$LOG_FILE"
  echo "Evaluating: $DATASET" | tee -a "$LOG_FILE"

  PREDICTION_FILE="${PREDICTION_BASE}/lisat_${DATASET}_answer.jsonl"
  ANSWER_FILE="${ANSWER_BASE}/${DATASET}.jsonl"

  # Evaluate using multiple metrics
  for METRIC in bleu meteor rouge cider spice; do
    METRIC_SCRIPT="${EVAL_SCRIPT_DIR}/eval_${METRIC}.py"
    if [ -f "$METRIC_SCRIPT" ]; then
      echo "Running $METRIC evaluation..." | tee -a "$LOG_FILE"
      python "$METRIC_SCRIPT" \
        --prediction_file "$PREDICTION_FILE" \
        --answer_file "$ANSWER_FILE" | tee -a "$LOG_FILE"
    else
      echo "⚠️  Missing script: $METRIC_SCRIPT" | tee -a "$LOG_FILE"
    fi
  done

  echo "Finished evaluation for $DATASET" | tee -a "$LOG_FILE"
done

# Create summary CSV
python "${EVAL_SCRIPT_DIR}/eval_results_csv.py"
echo "📊 Evaluation summary saved to $CSV_FILE"

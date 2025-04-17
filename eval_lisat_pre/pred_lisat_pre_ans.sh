#!/bin/bash

# =======================
# Batch VQA Inference Script
# =======================

# List of model checkpoints to evaluate
model_paths=(
  # "./checkpoints/model_A"
  # "./checkpoints/model_B"
)

# Map of dataset keys to JSONL filenames
declare -A datasets
datasets=(
  ["Sydney-Captions"]="Sydney-Captions.jsonl"
  ["UCM-Captions"]="UCM-Captions.jsonl"
  ["NWPU-Captions"]="NWPU-Captions.jsonl"
  ["RSICD"]="RSICD.jsonl"
  ["RSVQA_LR"]="RSVQA_LR.jsonl"
)

# Base paths
QUESTION_DIR="./dataset/vqa_caption_ans"
IMAGE_FOLDER="./dataset"
OUTPUT_BASE="./pred_dir"

# Run inference for each model
for model_path in "${model_paths[@]}"; do
  model_name=$(basename "$model_path")
  suffix="$model_name"
  echo "🔍 Evaluating model: $model_name"

  answers_folder="${OUTPUT_BASE}/pred_${suffix}"
  mkdir -p "$answers_folder"

  # Run inference on each dataset
  for dataset in "${!datasets[@]}"; do
    question_file="${QUESTION_DIR}/${datasets[$dataset]}"
    answers_file="${answers_folder}/${dataset}_prediction_${suffix}.jsonl"

    echo "📄 Running VQA: Dataset=${dataset}, Output=${answers_file}"
    
    python ./llava/eval/model_vqa.py \
      --model-path "$model_path" \
      --question-file "$question_file" \
      --image-folder "$IMAGE_FOLDER" \
      --answers-file "$answers_file" &
  done

  # Wait for all background jobs to finish for this model
  wait
  echo "✅ Completed predictions for: $model_name"
done

echo "🎉 All model predictions finished."

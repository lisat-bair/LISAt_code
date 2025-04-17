#!/bin/bash

# Set your experiment output directory
EXP_DIRECTORY="./runs/lisat"

# Set CUDA device to use
export CUDA_VISIBLE_DEVICES="0"

# Set base model path and target save path
LLAVA_PATH="./LISAt-7b"
HF_CKPT_PATH="${EXP_DIRECTORY}/hg_model"

# Save current directory
ORIGINAL_DIR=$(pwd)

# Step 1: Create a temporary binary file to hold the full FP32 model
TMP_FILE="$(realpath "${EXP_DIRECTORY}/tmp_model_$(date +%s).bin")"
cd "${EXP_DIRECTORY}/ckpt_model" || { echo "Directory not found!"; exit 1; }

# Convert Zero Redundancy Optimizer (ZeRO) checkpoints to single FP32 file
echo "Running zero_to_fp32.py..."
python zero_to_fp32.py . "$TMP_FILE"
if [ $? -ne 0 ]; then
  echo "❌ Error: zero_to_fp32.py failed."
  exit 1
fi

# Step 2: Merge LoRA weights into base model and save in HF format
cd "$ORIGINAL_DIR"
echo "Merging LoRA weights into base model..."
python3 merge_lora_weights_and_save_hf_model.py \
  --version="${LLAVA_PATH}" \
  --weight="$TMP_FILE" \
  --save_path="${HF_CKPT_PATH}"

if [ $? -ne 0 ]; then
  echo "❌ Error: merge_lora_weights_and_save_hf_model.py failed."
  exit 1
fi

# Cleanup temporary file
echo "Cleaning up temporary files..."
rm "$TMP_FILE"

echo "✅ LoRA merge completed successfully. Model saved to: ${HF_CKPT_PATH}"

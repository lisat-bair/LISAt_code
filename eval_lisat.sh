#!/bin/bash

# Function to run inference or evaluation stage
function run_inference() {
    CUDA_DEVICE="${1}"
    PROCESS_NUM="${2}"
    WORLD_SIZE="${3}"
    INFERENCE_CMD="${4:-inference}"

    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python eval_lisat.py \
        --cmd="${INFERENCE_CMD}" \
        --local_rank=0 \
        --process_num="${PROCESS_NUM}" \
        --world_size="${WORLD_SIZE}" \
        --dataset_dir="./dataset" \
        --pretrained_model_path="./runs/lisat/hg_model" \
        --val_dataset="GeoReasonSeg" \
        --vis_save_path="./inference_dir/lisat"
}

# Set number of processes (for DDP or multi-GPU setups)
WORLD_SIZE=1

# Step 1: Run inference
run_inference 0 0 ${WORLD_SIZE}

echo "Waiting for background inference processes to finish..."
wait

# Step 2: Run regular evaluation metrics
echo "Running evaluation metrics..."
run_inference 0 0 ${WORLD_SIZE} "metrics"

# Step 3: Run bootstrap evaluation metrics
echo "Running bootstrap metrics..."
run_inference 0 0 ${WORLD_SIZE} "bootstrap_metrics"

echo "DONE"

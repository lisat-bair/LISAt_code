#!/bin/bash
function run_inference() {
    CUDA_DEVICE="${1}"
    PROCESS_NUM="${2}"
    WORLD_SIZE="${3}"
    INFERENCE_CMD="${4:-inference}"
    CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" python eval_bootstraping.py \
        --cmd="${INFERENCE_CMD}" \
        --local_rank=0 \
        --process_num="${PROCESS_NUM}" \
        --world_size="${WORLD_SIZE}" \
        --dataset_dir /home/wenhan/Projects/sesame/dataset \
        --pretrained_model_path="/home/wenhan/Projects/sesame/runs/lisat_0223_v1/hg_model" \
        --val_dataset="GeoReasonSeg" \
        --vis_save_path="/home/wenhan/Projects/sesame/inference_dir/lisat_0223_v1_large"
}

WORLD_SIZE=1
run_inference 0 0 ${WORLD_SIZE} 
echo "Waiting for background inference processes to finish..."
wait
echo "Background processes finished. Running metrics..."
run_inference 0 0 ${WORLD_SIZE} "metrics"

# Now run bootstrap metrics:
run_inference 0 0 ${WORLD_SIZE} "bootstrap_metrics"

echo "DONE"

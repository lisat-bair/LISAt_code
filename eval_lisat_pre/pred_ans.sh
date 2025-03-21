#!/bin/bash

model_paths=(
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/merged-latest/merged-checkpoint7.5kclip-lora-llava/"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/merged-latest/merged-checkpoint7.5kclip336-lora-llava/"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/merged-latest/merged-checkpoint7.5kgeoclip-lora-llava/"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/merged-latest/merged-checkpoint7.5klremoteclip-lora-llava/"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/merged-latest/merged-checkpoint7.5ksatclip-lora-llava/"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/merged-latest/merged-checkpoint7.5kvremoteclip-lora-llava/"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/merged-latest/merged-checkpoint15.5kvremoteclip-lora-llava"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/latest/outputs/llava-v1.5-7"
#"/home/jquenum/Projects/Geo-LLaVA/working_dir/models_to_be_tested/latest/outputs/llava-v1.6-vicuna-7b"
)

declare -A datasets
datasets=(
  ["Sydney-Captions"]="Sydney-Captions.jsonl"
  ["UCM-Captions"]="UCM-Captions.jsonl"
  ["NWPU-Captions"]="NWPU-Captions.jsonl"
  ["RSICD"]="RSICD.jsonl"
  ["RSVQA_LR"]="RSVQA_LR.jsonl"
)


for model_path in "${model_paths[@]}"; do
    model_name=$(basename "$model_path")
    
    suffix=""
    
    echo "Using suffix: $suffix"
    
    answers_folder="/home/wenhan/lab_projects/LISAT_PRE/eval_lisat_pre/pred_dir/pred_${suffix}"
    
    mkdir -p "$answers_folder"
    
    for dataset in "${!datasets[@]}"; do
        question_file="${datasets[$dataset]}"
        answers_file="${answers_folder}/${dataset}_prediction_${suffix}.jsonl"
        
        echo "Running prediction for ${dataset} using model ${model_name}"
        
        python /home/wenhan/lab_projects/LISAT_PRE/llava/eval/model_vqa.py \
            --model-path "$model_path" \
            --question-file "/home/wenhan/lab_projects/LISAT_PRE/vqa_caption_ans/${question_file}" \
            --image-folder "/home/wenhan/lab_projects/LISAT_PRE/datasets/" \
            --answers-file "$answers_file" &
    done

    wait
done

echo "All predictions completed."

#!/bin/bash

# Assume we use 8 GPUs

# Parameters
VERSION="./LISAt-7b"
DATASET_DIR="./dataset"
EXP_NAME="lisat"
BATCH_SIZE=3
GRAD_ACCUMULATION_STEPS=4
NUM_CLASSES_PER_SAMPLE=3

# User-defined parameters
TRAINING_TYPE="$1"  # 'ReferSeg' or 'ReasonSeg'
GPU_SETTINGS="$2"   # e.g., 'localhost:0,1'
MASTER_PORT="$3"    # e.g., '15990'

# Check if parameters are provided
if [ -z "$TRAINING_TYPE" ] || [ -z "$GPU_SETTINGS" ] || [ -z "$MASTER_PORT" ]; then
    echo "Usage: $0 <Training Type> <GPU Settings> <Master Port>"
    echo "Example: $0 ReferSeg localhost:0,1 15990"
    exit 1
fi

# Vision backbone checkpoint
VISION_PRETRAINED="./sam_vit_h_4b8939.pth"
URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Download if missing
if [ ! -f "$VISION_PRETRAINED" ]; then
    echo "SAM model not found, downloading..."
    wget -O "$VISION_PRETRAINED" "$URL"
else
    echo "SAM model already exists."
fi

# ReferSeg Configuration
DATASET_REFERSEG="refer_seg||correct_refer_seg||vqa||neg_refer_seg"
SAMPLE_RATES_REFERSEG="12,2,2,1"

# ReasonSeg Configuration
DATASET_REASONSEG="sem_seg||refer_seg||correct_refer_seg||vqa||neg_refer_seg||reason_seg||geo_reason_seg"
SAMPLE_RATES_REASONSEG="15,15,2,30,1,1,36"

# Launch training
if [ "$TRAINING_TYPE" == "ReferSeg" ]; then
    echo "Launching ReferSeg training..."
    deepspeed --include "$GPU_SETTINGS" --master_port="$MASTER_PORT" train_lisat.py \
      --version="$VERSION" \
      --dataset_dir="$DATASET_DIR" \
      --vision_pretrained="$VISION_PRETRAINED" \
      --exp_name="$EXP_NAME" \
      --dataset="$DATASET_REFERSEG" \
      --sample_rates="$SAMPLE_RATES_REFERSEG" \
      --batch_size=$BATCH_SIZE \
      --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_classes_per_sample=$NUM_CLASSES_PER_SAMPLE

elif [ "$TRAINING_TYPE" == "ReasonSeg" ]; then
    echo "Launching ReasonSeg training..."
    deepspeed --include "$GPU_SETTINGS" --master_port="$MASTER_PORT" train_lisat.py \
      --version="$VERSION" \
      --dataset_dir="$DATASET_DIR" \
      --vision_pretrained="$VISION_PRETRAINED" \
      --exp_name="$EXP_NAME" \
      --dataset="$DATASET_REASONSEG" \
      --sample_rates="$SAMPLE_RATES_REASONSEG" \
      --reason_seg_data="ReasonSeg|train" \
      --geo_reason_seg_data="GeoReasonSeg|train" \
      --batch_size=$BATCH_SIZE \
      --grad_accumulation_steps $GRAD_ACCUMULATION_STEPS \
      --num_classes_per_sample=$NUM_CLASSES_PER_SAMPLE

else
    echo "Invalid training type. Please specify either 'ReferSeg' or 'ReasonSeg'."
    exit 1
fi

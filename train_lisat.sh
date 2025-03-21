#!/bin/bash

# Assume we use 8 GPUs

# Parameters
VERSION="/home/wenhan/Projects/sesame/llava-v1.5-7b-geollava_remoteclip_merged"
DATASET_DIR="/home/wenhan/Projects/sesame/dataset"
EXP_NAME="lisat_0306_nwpu_v2"
BATCH_SIZE=3
GRAD_ACCUMULATION_STEPS=4
NUM_CLASSES_PER_SAMPLE=3
# TEXT_PROMPT="Segment the target infrastructure in this image."

# User-defined parameters
TRAINING_TYPE="$1"  # Pass 'ReferSeg' or 'ReasonSeg' as the first argument
GPU_SETTINGS="$2"   # Pass GPU settings, e.g., 'localhost:0,1'
MASTER_PORT="$3"    # Pass master port, e.g., '15990'

# Check if parameters are provided
if [ -z "$TRAINING_TYPE" ] || [ -z "$GPU_SETTINGS" ] || [ -z "$MASTER_PORT" ]; then
    echo "Usage: $0 <Training Type> <GPU Settings> <Master Port>"
    echo "Example: $0 ReferSeg localhost:0,1 15990"
    exit 1
fi

# Download the SAM model
VISION_PRETRAINED="sam_vit_h_4b8939.pth"
# VISION_PRETRAINED="sam_decoder_multi_text.pth"
# VISION_PRETRAINED="/home/jquenum/Projects/Geo-LLaVA/geosam_weights/geosam_model.pth"
# URL="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"

# Check if the file does not exist
if [ ! -f "$VISION_PRETRAINED" ]; then
    echo "File does not exist, downloading..."
    wget -O "$VISION_PRETRAINED" "$URL"
else
    echo "File already exists, no need to download."
fi

# Seg-Only Configuration (Training our SESAME- model)
# DATASET_REFERSEG="refer_seg"
# SAMPLE_RATES_REFERSEG="1"

# ReferSeg Configuration (Training our SESAME model)
DATASET_REFERSEG="refer_seg||correct_refer_seg||vqa||neg_refer_seg"
SAMPLE_RATES_REFERSEG="12,2,2,1"

# ReasonSeg Configuration
# DATASET_REASONSEG="sem_seg||refer_seg||correct_refer_seg||vqa||neg_refer_seg||reason_seg||geo_reason_seg"
# SAMPLE_RATES_REASONSEG="10,10,2,3,1,1,15" (LISAT when submitting the paper) vqa:0.071, geo_reason_seg:0.357
# SAMPLE_RATES_REASONSEG="10,10,2,10,1,1,15" (LISAT (0202)) vqa:0.204, geo_reason_seg:0.306
# SAMPLE_RATES_REASONSEG="10,10,2,15,1,1,20" (LISAT (0203)) vqa:0.254, geo_reason_seg:0.339
# SAMPLE_RATES_REASONSEG="10,10,2,30,1,1,30" (LISAT (0204)) vqa:0.357, geo_reason_seg:0.357
# SAMPLE_RATES_REASONSEG="10,10,2,24,1,1,32" # (LISAT (0206)) vqa:0.300, geo_reason_seg:0.400
# DATASET_REASONSEG="vqa||geo_reason_seg"
# SAMPLE_RATES_REASONSEG="1,1" #(LISAT (0207))
# SAMPLE_RATES_REASONSEG="10,10,2,50,1,1,10" # (LISAT (0211_v4))

DATASET_REASONSEG="sem_seg||refer_seg||correct_refer_seg||vqa||neg_refer_seg||reason_seg||geo_reason_seg"
# SAMPLE_RATES_REASONSEG="24,24,5,7,2,2,36" (LISAT when submitting the paper)
# SAMPLE_RATES_REASONSEG="12,12,2,36,1,1,36" (LISAT (0204))
# SAMPLE_RATES_REASONSEG="5,5,2,46,1,1,40" # (LISAT (0212))
# SAMPLE_RATES_REASONSEG="15,15,0,40,0,1,29" # (LISAT (0213))
# SAMPLE_RATES_REASONSEG="12,12,0,60,0,1,15" # (LISAT (0214))
# SAMPLE_RATES_REASONSEG="3,3,0,70,0,1,23" # (LISAT (0218))
# SAMPLE_RATES_REASONSEG="12,12,0,44,0,1,31" # (LISAT (0220))
# SAMPLE_RATES_REASONSEG="15,15,2,30,1,1,36" # (LISAT (0221))
# SAMPLE_RATES_REASONSEG="13,13,0,45,0,1,28" # (LISAT (0223))
# SAMPLE_RATES_REASONSEG="8,8,0,60,0,1,23" # (LISAT (0227))
# SAMPLE_RATES_REASONSEG="13,13,0,43,0,1,30" # (LISAT (0305, 0228)) #giou bad
SAMPLE_RATES_REASONSEG="15,15,2,30,1,1,36" # (LISAT (0306, nwpu))


if [ "$TRAINING_TYPE" == "ReferSeg" ]; then
    # ReferSeg Command
    deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT train_lisat.py \
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
    # ReasonSeg Command
    deepspeed --include $GPU_SETTINGS --master_port=$MASTER_PORT train_lisat.py \
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
fi
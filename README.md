# LISAT: Language-Instructed Segmentation Assistant for Satellite Imagery

This repository provides the PyTorch source code for our paper: [LISAT: Language-Instructed Segmentation Assistant for Satellite Imagery](https://arxiv.org/abs/2312.08366). Check out our project page [here](https://see-say-segment.github.io/)!

**Authors**: [Jerome Quenum*](https://people.eecs.berkeley.edu/~jquenum/), [Wen-Han Hsieh*](https://wen-hanhsieh.github.io/personal_website/), [Tsung-Han Wu](https://tsunghan-wu.github.io/), [Ritwik Gupta](https://ritwikgupta.me/), [Trevor Darrell](https://people.eecs.berkeley.edu/~trevor/), [David M. Chan](https://dchan.cc/), at UC Berkeley. (* equal contribution)

## 🚀 One-Minute Introduction

Reading satellite images isn't just about spotting objects—it's about understanding their context, relationships, and sometimes even the absurdity of what humans ask AI to find. Enter **LISAT**, your AI-powered geospatial detective, trained to not only recognize objects but also **reason** about them. Think of it as the Sherlock Holmes of satellite imagery—minus the deerstalker hat. Whether it's pinpointing urban expansion or figuring out which suspiciously-shaped lake looks like a rubber duck, LISAT ensures that complex queries meet intelligent, nuanced segmentation.

LISAT is trained on two new datasets:
- **GRES (Geospatial Reasoning Segmentation dataset)**: 27,615 segmentation annotations across 9,205 images.
- **PreGRES**: A multimodal geospatial dataset with over 1M QA pairs for pretraining.

LISAT significantly outperforms previous models like RS-GPT4V, achieving +10.04% in BLEU-4 on visual description tasks and +143.36% in gIoU for segmentation tasks. 

## 📌 Status Update
- **02/28/2025:** Released training/evaluation scripts, demo scripts, model checkpoints, and datasets for geospatial segmentation.

## 🛠 Installation Guide

### System Requirements
- **OS**: Linux (Nvidia A100 GPUs recommended due to flash-attn usage)
- **Dependencies**:

```bash
conda create -n lisat python=3.9
pip3 install pybind11==2.11.1
pip3 install -r requirements.txt 
pip install flash-attn --no-build-isolation
```

### Required Files & Datasets
Ensure you have the following directories set up:
```
/home/wenhan/Projects/sesame/dataset
/home/wenhan/Projects/sesame/pycocoevalcap
/home/wenhan/Projects/sesame/eval_lisat_pre/pycocoevalcap
/home/wenhan/Projects/sesame/vqa
/home/wenhan/Projects/sesame/vqa_caption_ans
/home/wenhan/Projects/sesame/llava-v1.5-7b-geollava_remoteclip_merged
```

## 🏋️‍♂️ Training
Train LISAT using:

```bash
bash train_lisat.sh [ReferSeg or ReasonSeg] [Deepspeed GPU Settings] [MASTERPORT]
# Example:
bash train_lisat.sh ReasonSeg localhost:0,1 15990
```

## 🔄 Merge LoRA Weights
After training, merge LoRA weights for inference:

```bash
bash merge_lora_weight.sh
```

## 📊 Evaluation
### Geospatial Segmentation Evaluation (gIoU, cIoU):
```bash
bash eval_bootstrapping.sh
```

### Inference for Image Captioning:
```bash
bash pred_lisat_vqa.py
```

### Image Captioning Evaluation (BLEU, ROUGE-L, etc.)
```bash
cd eval_lisat_pre
bash eval_caption.sh
```

## 🙏 Acknowledgements
LISAT builds upon foundational work from [LISA](https://github.com/dvlab-research/LISA), [LLaVA](https://github.com/haotian-liu/LLaVA), and [SAM](https://github.com/facebookresearch/segment-anything). We acknowledge their contributions and licenses.
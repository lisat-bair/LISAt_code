# 🗂️ Dataset Preparation for LISAT

This README provides links and organization instructions for all datasets required to train and evaluate **LISAT** models across referring segmentation, reasoning segmentation, and captioning tasks.

---

## 📦 1. Referring Segmentation Datasets  
*Used for training and evaluating referring segmentation tasks (e.g., refcoco family).*

- 📄 Annotations:  
  [FP-/R- RefCOCO (+/g)](https://drive.google.com/file/d/1mA3kcY3QiAZz1Zr89MCKYd7e3LBIwUzl/view?usp=sharing)

- 🖼 COCO 2014 Images:  
  [train2014.zip](http://images.cocodataset.org/zips/train2014.zip)

---

## 🧠 2. Visual Question Answering Dataset  
*Used for pretraining referring segmentation models.*

- 🔤 LLaVA-Instruct-150k:  
  [HuggingFace Dataset](https://huggingface.co/datasets/liuhaotian/LLaVA-Instruct-150K/blob/main/llava_instruct_150k.json)

---

## 🧩 3. Semantic Segmentation Datasets  
*Used to train segmentation models for reasoning tasks (multi-class or fine-grained part segmentation).*

- [ADE20K Challenge Dataset](http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip)  
- [COCO-Stuff](http://calvin.inf.ed.ac.uk/wp-content/uploads/data/cocostuffdataset/stuffthingmaps_trainval2017.zip)  
- [PACO-LVIS (annotations only)](https://github.com/facebookresearch/paco/tree/main#dataset-setup)  
- [PASCAL-Part](https://github.com/facebookresearch/VLPart/tree/main/datasets#pascal-part)  
- [COCO Images 2017](http://images.cocodataset.org/zips/train2017.zip)

> ⚠️ Notes:
> - For COCO-Stuff, use `stuffthingmaps_trainval2017.zip` only.
> - Only the LVIS subset of PACO is required.
> - Place COCO images under `dataset/coco/train2017/`.

---

## 🔀 4. Augmented Reasoning Dataset  
*Includes false-premise queries for robust reasoning segmentation, geospatial reasoning segmentation.*

- [FP-Aug ReasonSeg (Google Drive)](https://drive.google.com/file/d/11WNg1KaV2mk7gTdJRa2aahGqfj4luTDw/view?usp=sharing)
- [GeoReasonSeg (Huggingface)](https://huggingface.co/datasets/jquenum/GRES/blob/main/README.md)
---

## 📁 Folder Structure

After downloading all files, organize them as follows:

```
LISAT
├── dataset
│   ├── ade20k
│   │   ├── annotations
│   │   └── images
│   ├── coco
│   │   └── train2017
│   │       ├── 000000000009.jpg
│   │       └── ...
│   ├── cocostuff
│   │   └── train2017
│   │       ├── 000000000009.png
│   │       └── ...
│   ├── llava_dataset
│   │   └── llava_instruct_150k.json
│   ├── reason_seg
│   │   ├── ReasonSeg
│   │   │   ├── train
│   │   │   └── val
│   │   └── GeoReasonSeg
│   │       ├── train
│   │       └── val
│   ├── refer_seg
│   │   ├── images
│   │   │   └── mscoco
│   │   │       └── images
│   │   │           └── train2014
│   │   ├── refclef
│   │   ├── refcoco
│   │   ├── refcoco+
│   │   ├── refcocog
│   │   ├── R-refcoco
│   │   ├── R-refcoco+
│   │   ├── R-refcocog
│   │   ├── fprefcoco
│   │   ├── fprefcoco+
│   │   └── fprefcocog
│   └── vlpart
│       ├── paco
│       │   └── annotations
│       └── pascal_part
│           ├── train.json
│           └── VOCdevkit
```

---

## 🧭 Tips

- Use symbolic links (`ln -s`) if you want to avoid dataset duplication across projects.
- Always verify image paths and annotation alignment before launching training or evaluation scripts.

---

## 📬 Questions?

Feel free to reach out via [issues](https://github.com/lisat-bair/LISAt/issues) if you encounter any trouble setting up the datasets.

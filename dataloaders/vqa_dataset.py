import os
import random
import json
import torch

from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from .base_dataset import BaseDataset

def preprocess_multimodal(source):
    for sentence in source:
        if DEFAULT_IMAGE_TOKEN in sentence["value"]:
            sentence["value"] = (
                sentence["value"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
            )
            sentence["value"] = DEFAULT_IMAGE_TOKEN + "\n" + sentence["value"]
            sentence["value"] = sentence["value"].strip()
            if "mmtag" in conversation_lib.default_conversation.version:
                sentence["value"] = sentence["value"].replace(
                    DEFAULT_IMAGE_TOKEN, "<Image>" + DEFAULT_IMAGE_TOKEN + "</Image>"
                )
    return source

class VQADataset(BaseDataset):
    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 336,
        vqa_data="llava_instruct_150k",
        geo_vqa_data="/home/wenhan/Projects/Geo-LLaVA/RS-GPT4V_EarthGPT.json",
        geo_image_dir="/home/wenhan/Projects/Geo-LLaVA/datasets/",
    ):
        super().__init__(vision_tower, samples_per_epoch, image_size)
        self.base_image_dir = base_image_dir
        self.vqa_image_root = os.path.join(base_image_dir, "coco/train2017")
        self.geo_image_root = geo_image_dir

        # Load the original LLaVA dataset
        DATA_DIR = os.path.join(base_image_dir, "llava_dataset")
        llava_path = os.path.join(DATA_DIR, f"{vqa_data}.json")
        with open(llava_path, "r") as f:
            llava_data = json.load(f)

        # Load the new dataset (RS-GPT4V_EarthGPT)
        with open(geo_vqa_data, "r") as f:
            extra_data = json.load(f)

        self.vqa_data = llava_data + extra_data 
        print(f"LLaVA dataset size: {len(llava_data)}")
        print(f"Geo-LLaVA dataset size: {len(extra_data)}")
        print(f"Total dataset size: {len(self.vqa_data)}")

    def __getitem__(self, idx):
        idx = random.randint(0, len(self.vqa_data) - 1)
        item = self.vqa_data[idx]

        # Ensure image field exists
        if "image" not in item or not item["image"]:
            raise ValueError(f"Missing 'image' key in item: {item}")

        image_name = item["image"]  # Get the relative image path

        # If the image belongs to the Geo-LLaVA dataset, map it correctly
        geo_image_path = os.path.join(self.geo_image_root, image_name)
        
        # If the image belongs to the original LLaVA dataset
        coco_image_path = os.path.join(self.vqa_image_root, image_name)

        # Correctly assign the path based on where the file exists
        if os.path.exists(geo_image_path):
            image_path = geo_image_path
        elif os.path.exists(coco_image_path):
            image_path = coco_image_path
        else:
            raise FileNotFoundError(
                f"Image not found in both datasets:\n"
                f" - Expected Geo-LLaVA Path: {geo_image_path}\n"
                f" - Expected COCO Path: {coco_image_path}"
            )

        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)

        # Process conversation Q/A
        conv = conversation_lib.default_conversation.copy()
        source = preprocess_multimodal(item["conversations"])
        roles = {"human": conv.roles[0], "gpt": conv.roles[1]}
        conversations = []
        
        if roles[source[0]["from"]] != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for sentence in source:
            role = roles[sentence["from"]]
            conv.append_message(role, sentence["value"])

        conversations.append(conv.get_prompt())

        # Empty segmentation maps for VQA datasets
        masks = torch.rand(0, *sam_input_shape)
        exists = [False]
        sam_mask_shape = [sam_input_shape, (masks.shape[1], masks.shape[2])]

        return (
            image_path,  # filename
            image,       # raw image (for SAM)
            image_clip,  # image clip feature (for LMMs)
            conversations,  # QA
            masks,  # segmentation GT
            sam_mask_shape,  # input / output shape for SAM
            exists,  # object existence
            None,
            None
        )

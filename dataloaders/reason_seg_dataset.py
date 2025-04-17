import glob
import json
import os
import random

import cv2
import numpy as np
import torch
from transformers import CLIPImageProcessor

from model.llava import conversation as conversation_lib
from model.segment_anything.utils.transforms import ResizeLongestSide

from .data_processing import get_mask_from_json
from .qa_template import LONG_QUESTION_TEMPLATE, LONG_ANSWER_TEMPLATE, SHORT_QUESTION_TEMPLATE
from .base_dataset import BaseDataset

class ReasonSegDataset(BaseDataset):
    choices = ["True_Premise", "False_Premise_Correction"]
    weights = [0.85, 0.15]

    def __init__(
        self,
        base_image_dir,
        vision_tower,
        samples_per_epoch=500 * 8 * 2 * 10,
        image_size: int = 224,
        num_classes_per_sample: int = 3,
        reason_seg_data="ReasonSeg|train",
        use_fp=True,  # Enable or disable false premise QA
    ):
        super().__init__(vision_tower, samples_per_epoch, image_size)
        self.use_fp = use_fp  # Store the use_fp flag
        self.num_classes_per_sample = num_classes_per_sample
        self.base_image_dir = base_image_dir

        self.short_question_list = SHORT_QUESTION_TEMPLATE
        self.long_question_list = LONG_QUESTION_TEMPLATE
        self.answer_list = LONG_ANSWER_TEMPLATE

        # Load dataset
        reason_seg_data_name, splits = reason_seg_data.split("|")
        splits = splits.split("_")
        data_pairs = []
        for split in splits:
            if reason_seg_data_name == "ReasonSeg":
                images_split = glob.glob(
                    os.path.join(
                        base_image_dir, "reason_seg", reason_seg_data_name, split, "*.jpg"
                    )
                )
                for image_path in images_split:
                    json_path = image_path.replace(".jpg", ".json")
                    if os.path.exists(json_path):
                        with open(json_path, 'r') as f:
                            try:
                                data = json.load(f)
                            except json.JSONDecodeError as e:
                                print(f"Error parsing JSON file {json_path}: {e}")
                                continue  # Skip this file
                        data_pairs.append((image_path, json_path))
                    else:
                        print(f"JSON path does not exist: {json_path}")
            else:
                json_paths = glob.glob(
                    os.path.join(
                        base_image_dir, "reason_seg", reason_seg_data_name, split, "*.json"
                    )
                )
                for json_path in json_paths:
                    with open(json_path, 'r') as f:
                        try:
                            data = json.load(f)
                        except json.JSONDecodeError as e:
                            print(f"Error parsing JSON file {json_path}: {e}")
                            continue  # Skip this file
                    image_name = data.get('shapes', [{}])[0].get('image_name', None)
                    if image_name is None:
                        image_name = os.path.basename(json_path).replace(".json", ".jpg")
                    image_path = os.path.join(base_image_dir, "reason_seg", reason_seg_data_name, split, image_name)
                    if os.path.exists(image_path):
                        data_pairs.append((image_path, json_path))
                    else:
                        print(f"Image path does not exist: {image_path}")
        self.reason_seg_data = data_pairs

        print("number of reason_seg samples: ", len(data_pairs))

    def __len__(self):
        return len(self.reason_seg_data)

    def __getitem__(self, idx):
        data_pairs = self.reason_seg_data
        idx = random.randint(0, len(data_pairs) - 1)

        image_path, json_path = data_pairs[idx]

        # Load images and clip features
        image, image_clip, sam_input_shape = self.load_and_preprocess_image(image_path)
        # Get sentences and segmentation maps
        img = cv2.imread(image_path)[:, :, ::-1]
        mask, sents, fp_qa, is_sentence = get_mask_from_json(json_path, img)

        # Decide the mode for this turn
        if self.use_fp:
            mode_this_turn = random.choices(self.choices, self.weights, k=1)[0]
        else:
            mode_this_turn = "True_Premise"  # Always use True_Premise when use_fp is False

        # Sampling positive examples
        sample_size = min(len(sents), self.num_classes_per_sample)
        sampled_inds = (
            random.sample(range(len(sents)), sample_size)
            if len(sents) >= self.num_classes_per_sample
            else list(range(len(sents)))
        )
        sampled_sents = [sents[idx] for idx in sampled_inds]
        sampled_masks = [
            (mask == 1).astype(np.float32) for _ in range(len(sampled_inds))
        ]

        # Initialize variables for negative samples
        neg_sampled_sents = []
        if self.use_fp and mode_this_turn == "False_Premise_Correction":
            if len(fp_qa) == 0:
                # If fp_qa is empty, default to True_Premise
                mode_this_turn = "True_Premise"
            else:
                # Sampling negative examples
                neg_sample_size = min(len(fp_qa), self.num_classes_per_sample)
                neg_sampled_inds = (
                    random.sample(range(len(fp_qa)), neg_sample_size)
                    if len(fp_qa) >= self.num_classes_per_sample
                    else list(range(len(fp_qa)))
                )
                neg_sampled_sents = [fp_qa[idx] for idx in neg_sampled_inds]

        # Create Q/A Data
        questions = []
        answers = []
        conversations = []

        if mode_this_turn == "True_Premise":
            for idx, text in enumerate(sampled_sents):
                if is_sentence:
                    question_template = random.choice(self.long_question_list)
                    questions.append(question_template.format(sent=text))
                else:
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=text.lower()))
                answers.append(random.choice(self.answer_list))

                conv = conversation_lib.default_conversation.copy()
                conv.append_message(conv.roles[0], questions[-1])
                conv.append_message(conv.roles[1], answers[-1])
                conversations.append(conv.get_prompt())

            exists = [True for _ in range(len(sampled_sents))]
            masks = np.stack(sampled_masks, axis=0)
            masks = torch.from_numpy(masks)
        else:
            for idx, neg_text in enumerate(neg_sampled_sents):
                if neg_text[1] is True:
                    question_template = random.choice(self.long_question_list)
                    questions.append(question_template.format(sent=neg_text[0]))
                else:
                    question_template = random.choice(self.short_question_list)
                    questions.append(question_template.format(class_name=neg_text[0]))
                answers.append(neg_text[2])

                conv = conversation_lib.default_conversation.copy()
                conv.append_message(conv.roles[0], questions[-1])
                conv.append_message(conv.roles[1], answers[-1])
                conversations.append(conv.get_prompt())

            exists = [False for _ in range(len(neg_sampled_sents))]
            # Create empty masks tensor since there are no valid masks
            masks = torch.empty(0, *sam_input_shape, dtype=torch.float32)

        sam_mask_shape = [
            (int(sam_input_shape[0]), int(sam_input_shape[1])),
            (int(masks.shape[1]), int(masks.shape[2])) if masks.shape[0] > 0 else (0, 0),
        ]

        ref_ids = [int(idx)] * len(exists)  # Placeholder ref_ids
        sent_ids = list(range(len(exists)))  # Placeholder sent_ids

        return (
            image_path,      # filename
            image,           # raw image (for SAM)
            image_clip,      # image clip feature (for LMMs)
            conversations,   # QA
            masks,           # segmentation GT
            sam_mask_shape,  # input / output shape for SAM
            exists,          # object existence
            ref_ids,
            sent_ids
        )

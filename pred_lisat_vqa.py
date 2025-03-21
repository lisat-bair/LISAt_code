import argparse
import os
import sys
import cv2
import math
import numpy as np
import pandas as pd
import json
import torch
from tqdm import tqdm
from PIL import Image
from io import BytesIO
from model.SESAME import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad


all_options = ['A', 'B', 'C', 'D']


def parse_args(args):
    parser = argparse.ArgumentParser(description="SESAME demo")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--pretrained_model_path", default="/home/wenhan/Projects/sesame/runs/lisat_0223_v1/hg_model")
    parser.add_argument("--vis_save_dir", default="./demo_directory", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    
    # Multiple datasets
    parser.add_argument("--question_files", nargs='+', type=str, default=[
        "/home/wenhan/Projects/sesame/vqa/UCM-Captions.jsonl",
        "/home/wenhan/Projects/sesame/vqa/Sydney-Captions.jsonl",
        "/home/wenhan/Projects/sesame/vqa/RSICD.jsonl",
        "/home/wenhan/Projects/sesame/vqa/NWPU-Captions.jsonl",
        "/home/wenhan/Projects/sesame/vqa/RSVQA_LR.jsonl",
    ])
    
    parser.add_argument("--answer_files", nargs='+', type=str, default=[
        "captioning_dir/lisat_0223_v1/lisat_0223_v1_UCM-Captions_answer.jsonl",
        "captioning_dir/lisat_0223_v1/lisat_0223_v1_Sydney-Captions_answer.jsonl",
        "captioning_dir/lisat_0223_v1/lisat_0223_v1_RSICD_answer.jsonl",
        "captioning_dir/lisat_0223_v1/lisat_0223_v1_NWPU-Captions_answer.jsonl",
        "captioning_dir/lisat_0223_v1/lisat_0223_v1_RSVQA_LR_answer.jsonl"
    ])
    
    return parser.parse_args(args)


def save_segmentation(pred_mask, input_dict, args):
    pred_mask = pred_mask.detach().cpu().numpy()
    pred_mask = pred_mask > 0

    image_path = input_dict["image_path"]
    image_key = image_path.split("/")[-1].split(".")[0]
    save_dir = os.path.join(args.vis_save_dir, image_key)
    os.makedirs(save_dir, exist_ok=True)
    seg_fname = os.path.join(save_dir, "seg_mask.jpg")
    cv2.imwrite(seg_fname, pred_mask * 100)
    seg_rgb_fname = os.path.join(save_dir, "seg_rgb.jpg")
    image_np = cv2.imread(image_path)
    image_np[pred_mask] = (
        image_np * 0.3
        + pred_mask[:, :, None].astype(np.uint8) * np.array([0, 0, 255]) * 0.7
    )[pred_mask]
    cv2.imwrite(seg_rgb_fname, image_np)
    return save_dir


def is_none(value):
    if value is None:
        return True
    if type(value) is float and math.isnan(value):
        return True
    if type(value) is str and value.lower() == 'nan':
        return True
    if type(value) is str and value.lower() == 'none':
        return True
    return False

def get_options(row, options):
    parsed_options = []
    for option in options:
        option_value = row[option]
        if is_none(option_value):
            break
        parsed_options.append(option_value)
    return parsed_options


@torch.inference_mode()
def run_demo_for_dataset(args, question_file, answer_file):
    """Runs the demo on a specific dataset."""
    
    # Load Model
    tokenizer, segmentation_lmm, vision_tower, context_len = load_pretrained_model_SESAME(
        model_path=args.pretrained_model_path
    )

    # Load bf16 datatype
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")

    tokenizer.padding_side = "left"

    # Load questions
    questions = [json.loads(q) for q in open(os.path.expanduser(question_file), "r")]
    answers_file_path = os.path.expanduser(answer_file)
    
    os.makedirs(os.path.dirname(answers_file_path), exist_ok=True)
    ans_file = open(answers_file_path, "w")

    for index, row in enumerate(tqdm(questions)):
        idx = row['question_id']
        question = row['text']
        cur_prompt = question

        # Format images
        img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
        image_path = os.path.join("/home/wenhan/Projects/Geo-LLaVA/datasets", row['image'])
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_path)

        # Format Question
        conv = conversation_lib.default_conversation.copy()
        question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        conversation_list = [conv.get_prompt()]

        mm_use_im_start_end = getattr(segmentation_lmm.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            conversation_list = replace_image_tokens(conversation_list)
        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')

        # Format input dictionary
        input_dict = {
            "images_clip": torch.stack([image_clip], dim=0),
            "images": torch.stack([image], dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape]
        }

        # Move inputs to GPU in bfloat16
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        # Evaluate the model for text and segmentation
        output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )

        # Decode text response
        real_output_ids = output_ids[:, input_ids.shape[1] :]
        outputs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0]

        # Write answer JSON
        ans_file.write(json.dumps({
            "question_id": idx,
            "round_id": 0,
            "prompt": cur_prompt,
            "text": outputs,
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()


def main(args):
    args = parse_args(args)

    # Validate input length
    if len(args.question_files) != len(args.answer_files):
        raise ValueError("The number of question files and answer files must match!")

    print("Running inference on multiple datasets...")

    # Run the model on each dataset
    for q_file, a_file in zip(args.question_files, args.answer_files):
        print(f"\nProcessing: {q_file} -> {a_file}")
        run_demo_for_dataset(args, q_file, a_file)

    print("\nAll datasets processed successfully!")
if __name__ == "__main__":
    main(sys.argv[1:])
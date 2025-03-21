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

# ========== Import your SESAME, LLaVA, and utility modules as in pred_lisat_vqa.py ==========
from model.SESAME import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad

all_options = ['A', 'B', 'C', 'D']

def parse_args(args):
    parser = argparse.ArgumentParser(description="SESAME GeoBench-VLM Captioning Evaluation")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument("--pretrained_model_path", 
                        default="/home/wenhan/Projects/sesame/runs/lisat_0223_v1/hg_model", 
                        type=str,
                        help="Path to the pretrained SESAME model")
    parser.add_argument("--vis_save_dir", default="./demo_directory", type=str)
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--conv_type",
                        default="llava_v1",
                        type=str,
                        choices=["llava_v1", "llava_llama_2"],
                        help="Type of LLaVA conversation format")

    # Single dataset for GeoBench-VLM
    parser.add_argument("--question_file", 
                        default="/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Captioning/qa.json",
                        type=str,
                        help="GeoBench-VLM Captioning QA file (JSON array)")
    parser.add_argument("--answer_file", 
                        default="pred_geobench_captioning.json", 
                        type=str,
                        help="Output JSON file with model predictions")

    return parser.parse_args(args)

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

@torch.inference_mode()
def run_demo(args):
    """
    Runs the demo on the GeoBench-VLM Captioning dataset,
    and writes out a single JSON file with model predictions.
    """
    # 1. Load the model
    tokenizer, segmentation_lmm, vision_tower, context_len = load_pretrained_model_SESAME(
        model_path=args.pretrained_model_path
    )
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")
    tokenizer.padding_side = "left"

    # 2. Load the QA data (an array of objects) from GeoBench-VLM
    question_file_path = os.path.expanduser(args.question_file)
    with open(question_file_path, "r") as f:
        data = json.load(f)  # data is a list of dicts

    # 3. Prepare output directory and file
    answers_file_path = os.path.expanduser(args.answer_file)
    os.makedirs(os.path.dirname(answers_file_path), exist_ok=True)
    ans_file = open(answers_file_path, "w")

    # 4. Iterate over each record in the QA file
    for entry in tqdm(data, desc="Processing GeoBench-VLM Captioning"):
        question_id = entry["question_id"]
        image_rel_path = entry["image_path"]  # e.g. "Captioning/images/captioning_98.png"
        
        # Use one of the prompts as the question. For example, the first prompt:
        # Or choose any from entry['prompts']. We'll pick the first one for consistency.
        question = entry["prompts"][0] if entry["prompts"] else "Describe the image in detail."

        # 5. Load the image
        #    Typically, you might need to join a root directory with the relative path
        #    Adjust path as necessary based on your directory structure:
        image_full_path = os.path.join("/home/wenhan/Projects/sesame/dataset/GEOBench-VLM", image_rel_path)
        if not os.path.exists(image_full_path):
            print(f"Warning: Image file not found at {image_full_path}")
            continue

        # 6. Image preprocessing
        img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_full_path)

        # 7. Construct conversation
        #    Similar to pred_lisat_vqa.py, we embed the image token + the question prompt
        conv = conversation_lib.default_conversation.copy()
        question_text = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], question_text)
        conv.append_message(conv.roles[1], None)

        conversation_list = [conv.get_prompt()]

        # Check if we need <im_start> and <im_end> tokens
        mm_use_im_start_end = getattr(segmentation_lmm.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            conversation_list = replace_image_tokens(conversation_list)

        # 8. Tokenize
        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')

        # 9. Format the input dict for the model
        input_dict = {
            "images_clip": torch.stack([image_clip], dim=0),
            "images": torch.stack([image], dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape]
        }
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        # 10. Forward pass
        output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )

        # 11. Decode text response
        real_output_ids = output_ids[:, input_ids.shape[1]:]
        outputs = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0]

        # 12. Write the prediction to file
        ans_file.write(json.dumps({
            "question_id": question_id,
            "round_id": 0,
            "prompt": question,
            "text": outputs.strip(),
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()
    print(f"Predictions saved to {answers_file_path}")

def main(args):
    args = parse_args(args)
    run_demo(args)

if __name__ == "__main__":
    main(sys.argv[1:])

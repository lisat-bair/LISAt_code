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

# ========== Import your SESAME, LLaVA, and utility modules as in pred_lisat_vqa or captioning scripts ==========

from model.SESAME_eval import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad


def parse_args(args):
    parser = argparse.ArgumentParser(description="SESAME GeoBench-VLM Single-Choice Inference & Accuracy")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # === Model & Paths ===
    parser.add_argument(
        "--pretrained_model_path",
        default="/home/wenhan/Projects/sesame/runs/lisat_0223_v1/hg_model",
        type=str,
        help="Path to the pretrained SESAME model"
    )
    parser.add_argument(
        "--image_root",
        default="/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Single/images",
        type=str,
        help="Root directory where Single dataset images are stored"
    )
    parser.add_argument(
        "--question_file",
        default="/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Single/qa.json",
        type=str,
        help="GeoBench-VLM Single QA file (JSON array)"
    )
    parser.add_argument(
        "--output_file",
        default="/home/wenhan/Projects/sesame/pred_genbench/lisat_0223_v1/pred_geobench_single.json",
        type=str,
        help="Output JSON file with model predictions"
    )

    # === Model Inference Settings ===
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
        help="Type of LLaVA conversation format"
    )

    return parser.parse_args(args)


def is_none(value):
    """Helper to check if a value is effectively None or NaN."""
    if value is None:
        return True
    if isinstance(value, float) and math.isnan(value):
        return True
    if isinstance(value, str) and value.lower() in ['nan', 'none']:
        return True
    return False


@torch.inference_mode()
def run_inference(args):
    """
    Runs the demo on the GeoBench-VLM Single dataset (multi-choice classification),
    and writes out a single JSON file with model predictions. Also prints accuracy.
    """
    # 1. Load the model
    tokenizer, segmentation_lmm, vision_tower, context_len = load_pretrained_model_SESAME(
        model_path=args.pretrained_model_path
    )
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    # Optional (requires PyTorch 2.0+): compile the model for speed
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")
    tokenizer.padding_side = "left"

    # 2. Load the QA data (an array of objects) from GeoBench-VLM Single
    question_file_path = os.path.expanduser(args.question_file)
    with open(question_file_path, "r") as f:
        data = json.load(f)  # data is a list of dicts

    # 3. Prepare output file
    output_file_path = os.path.expanduser(args.output_file)
    output_dir = os.path.dirname(output_file_path)

    # Only create the directory if output_dir is not empty
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    out_fp = open(output_file_path, "w")

    # For accuracy calculation
    correct_count = 0
    total_count = 0

    # 4. For each entry, build prompt, load image, forward through the model
    for entry in tqdm(data, desc="Processing GeoBench-VLM Single"):
        question_id = entry["question_id"]
        image_rel_path = entry["image_path"]   # e.g. "Single/images/single_347.bmp"
        options_str = entry["options"]         # e.g. "A. ... B. ... C. ... D. ... E. ..."
        ground_truth_option = entry["ground_truth_option"]  # e.g. "E"
        prompts = entry.get("prompts", [])
        # We'll pick the first prompt for a standard approach
        if len(prompts) > 0:
            question_prompt = prompts[0]
        else:
            question_prompt = "What is the correct answer?"

        # Build a single multi-choice question
        # For clarity, we can incorporate the options into the prompt to guide the model:
        full_question_text = (
            f"{question_prompt}\n\n"
            f"Possible answers:\n"
            f"{options_str}\n"
            "Please choose the best option (just give the letter and reasoning if needed)."
        )

        # 5. Load the image
        #    If the `image_path` in JSON is already relative, confirm that `args.image_root`
        #    + the tail of image_rel_path is the correct location.
        #    If it already has "Single/images" in the path, you could do:
        image_full_path = os.path.join(args.image_root, os.path.basename(image_rel_path))
        if not os.path.exists(image_full_path):
            print(f"Warning: Image file not found at {image_full_path}")
            # Skip if not found
            continue

        # 6. Preprocess the image
        img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_full_path)

        # 7. Construct conversation
        conv = conversation_lib.default_conversation.copy()
        question_with_image = DEFAULT_IMAGE_TOKEN + "\n" + full_question_text
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)

        conversation_list = [conv.get_prompt()]

        # Check if we need <im_start> and <im_end> tokens
        mm_use_im_start_end = getattr(segmentation_lmm.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            conversation_list = replace_image_tokens(conversation_list)

        # 8. Tokenize
        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')

        # 9. Prepare input dictionary
        input_dict = {
            "images_clip": torch.stack([image_clip], dim=0),
            "images": torch.stack([image], dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape]
        }
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        # 10. Forward pass to get output
        output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )

        # 11. Decode text response
        real_output_ids = output_ids[:, input_ids.shape[1]:]
        model_output_text = tokenizer.batch_decode(
            real_output_ids, skip_special_tokens=True
        )[0].strip()

        # 12. Heuristically parse the predicted option letter
        #     We expect something like: "I think the answer is E" or "The correct option is A."
        #     We'll look for the first letter in [A-E].
        predicted_option = None
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in model_output_text:
                # We'll do a naive check: if " A " or "A." or "Answer: A" etc. appears
                # We'll break on the first match. Another approach is to parse more carefully.
                # (In practice, you may want a better pattern or prompt the model to give "Answer: X".)
                predicted_option = letter
                break

        # If we never found a letter, the model might have answered in full text. 
        # We can also attempt to match the actual strings in 'options_list' if needed.
        # For now, let's proceed with the letter approach:
        if predicted_option is None:
            predicted_option = "UNDEF"

        # 13. Check correctness
        is_correct = (predicted_option == ground_truth_option)
        if ground_truth_option is not None and ground_truth_option in ["A", "B", "C", "D", "E"]:
            total_count += 1
            if is_correct:
                correct_count += 1

        # 14. Write the prediction (one line per sample) to output JSON
        out_fp.write(json.dumps({
            "question_id": question_id,
            "image_path": image_rel_path,
            "prompt": question_prompt,
            "options": options_str,
            "model_output": model_output_text,
            "predicted_option": predicted_option,
            "ground_truth_option": ground_truth_option,
            "is_correct": is_correct
        }) + "\n")
        out_fp.flush()

    # 15. Close output file
    out_fp.close()

    # 16. Print accuracy
    accuracy = 0.0
    if total_count > 0:
        accuracy = correct_count / total_count
    print(f"Accuracy on GEOBench-VLM Single: {accuracy*100:.2f}% "
          f"({correct_count}/{total_count} correct)")


def main(args):
    args = parse_args(args)
    run_inference(args)


if __name__ == "__main__":
    main(sys.argv[1:])

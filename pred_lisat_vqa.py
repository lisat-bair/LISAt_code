import argparse
import os
import sys
import cv2
import math
import json
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from model.LISAT import load_pretrained_model_LISAT
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad


def parse_args(args):
    parser = argparse.ArgumentParser(description="Batch inference for LISAT VQA/Captioning datasets")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--pretrained_model_path", default="./runs/lisat/hg_model")
    parser.add_argument("--vis_save_dir", default="./demo_output", type=str)
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--conv_type", choices=["llava_v1", "llava_llama_2"], default="llava_v1")

    # Input/output files
    parser.add_argument("--question_files", nargs='+', type=str, default=[
        "./dataset/vqa_caption/UCM-Captions.jsonl",
        "./dataset/vqa_caption/Sydney-Captions.jsonl",
        "./dataset/vqa_caption/RSICD.jsonl",
        "./dataset/vqa_caption/NWPU-Captions.jsonl",
        "./dataset/vqa_caption/RSVQA_LR.jsonl",
    ])
    parser.add_argument("--answer_files", nargs='+', type=str, default=[
        "./captioning_dir/lisat/lisat_ucm.jsonl",
        "./captioning_dir/lisat/lisat_sydney.jsonl",
        "./captioning_dir/lisat/lisat_rsicd.jsonl",
        "./captioning_dir/lisat/lisat_nwpu.jsonl",
        "./captioning_dir/lisat/lisat_rsvqa.jsonl"
    ])
    parser.add_argument("--dataset_base", default="./dataset", type=str, help="Base directory for image paths in JSONL")
    return parser.parse_args(args)


def is_none(value):
    return value is None or (
        isinstance(value, float) and math.isnan(value)
    ) or (
        isinstance(value, str) and value.lower() in {"nan", "none"}
    )


@torch.inference_mode()
def run_demo_for_dataset(args, question_file, answer_file):
    """Run LISAT inference for a specific dataset."""
    tokenizer, model, vision_tower, _ = load_pretrained_model_LISAT(args.pretrained_model_path)

    model = model.to(torch.bfloat16)
    vision_tower = vision_tower.to(torch.bfloat16)
    model = torch.compile(model, mode="reduce-overhead")
    tokenizer.padding_side = "left"

    questions = [json.loads(line) for line in open(question_file, "r")]
    os.makedirs(os.path.dirname(answer_file), exist_ok=True)
    ans_file = open(answer_file, "w")

    for row in tqdm(questions):
        question_id = row["question_id"]
        question = row["text"]
        image_path = os.path.join(args.dataset_base, row["image"])

        # Image processing
        img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_path)

        # Build conversation
        conv = conversation_lib.default_conversation.copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + question)
        conv.append_message(conv.roles[1], None)
        conv_list = [conv.get_prompt()]
        if getattr(model.config, "mm_use_im_start_end", False):
            conv_list = replace_image_tokens(conv_list)

        input_ids, _ = tokenize_and_pad(conv_list, tokenizer, padding="left")
        input_dict = {
            "images_clip": torch.stack([image_clip]),
            "images": torch.stack([image]),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape],
        }
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        # Run model
        output_ids, _, _ = model.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )
        decoded_output = tokenizer.batch_decode(output_ids[:, input_ids.shape[1]:], skip_special_tokens=True)[0]

        # Save response
        ans_file.write(json.dumps({
            "question_id": question_id,
            "round_id": 0,
            "prompt": question,
            "text": decoded_output,
            "metadata": {}
        }) + "\n")
        ans_file.flush()

    ans_file.close()


def main(args):
    args = parse_args(args)

    if len(args.question_files) != len(args.answer_files):
        raise ValueError("Mismatch: Number of question files and answer files must be the same.")

    print("🔍 Starting LISAT inference for all VQA datasets...")
    for q_path, a_path in zip(args.question_files, args.answer_files):
        print(f"➡️  Processing: {q_path} → {a_path}")
        run_demo_for_dataset(args, q_path, a_path)
    print("✅ All datasets processed!")


if __name__ == "__main__":
    main(sys.argv[1:])

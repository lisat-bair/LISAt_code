import argparse
import os
import sys
import cv2
import numpy as np
import torch
import json
from model.SESAME import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad


def parse_args(args):
    parser = argparse.ArgumentParser(description="SESAME demo")
    parser.add_argument("--local_rank", default=6, type=int, help="node rank")
    #parser.add_argument("--pretrained_model_path", default="/home/wenhan/Projects/sesame/runs/geo_v3/hg_model/")
    #parser.add_argument("--pretrained_model_path", default="xinlai/LISA-13B-llama2-v1")
    #parser.add_argument("--pretrained_model_path", default="xinlai/LISA-7B-v1")
    #parser.add_argument("--pretrained_model_path", default="/home/wenhan/Projects/sesame/runs/geo_v4/hg_model/")
    #parser.add_argument("--pretrained_model_path", default="/home/wenhan/Projects/sesame/runs/geo_0120/hg_model/")
    parser.add_argument("--pretrained_model_path", default="xinlai/LISA-13B-llama2-v1")
  
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--conv_type",
        default="llava_v1", 
        type=str,
        choices=["llava_v1", "llava_llama_2", "geo_llava"],  # Add geo_llava 
    )
    # Add input and output directories
    parser.add_argument('--input_dir', default='/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/large', type=str, help='Input directory containing images and JSON files')
    parser.add_argument('--output_dir', default='/home/wenhan/Projects/sesame/LISA-13B-llama2-v1_inference_dir/large_png_inference', type=str, help='Output directory to save segmentation images')
    return parser.parse_args(args)


def save_segmentation(pred_mask, input_dict, args):
    pred_mask = pred_mask.detach().cpu().numpy()
    pred_mask = pred_mask > 0

    image_path = input_dict["image_path"]
    image_key = os.path.basename(image_path).split(".")[0]
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    # Save segmentation mask
    seg_fname = os.path.join(args.output_dir, f"{image_key}_seg_mask.png")
    cv2.imwrite(seg_fname, pred_mask.astype(np.uint8) * 255)
    # Save segmentation overlay
    seg_rgb_fname = os.path.join(args.output_dir, f"{image_key}_seg_rgb.png")
    image_np = cv2.imread(image_path)
    # Overlay the segmentation mask on the original image
    overlay = image_np.copy()
    overlay[pred_mask] = (0, 0, 255)  # Red color for the mask
    alpha = 0.7  # Transparency factor
    cv2.addWeighted(overlay, alpha, image_np, 1 - alpha, 0, image_np)
    cv2.imwrite(seg_rgb_fname, image_np)
    return args.output_dir


@torch.inference_mode()
def demo(args):
    # Initialization
    os.makedirs(args.output_dir, exist_ok=True)

    (
        tokenizer,
        segmentation_lmm,
        vision_tower,
        context_len,
    ) = load_pretrained_model_SESAME(
        model_path=args.pretrained_model_path
    )
    # Load bf16 datatype
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")
    # for eval only
    tokenizer.padding_side = "left"
    
    # Get all image files in the input directory
    image_files = [f for f in os.listdir(args.input_dir) if f.endswith('.jpg')]
    image_files.sort()
    for image_file in image_files:
        image_path = os.path.join(args.input_dir, image_file)
        # Get the corresponding JSON file
        json_file = os.path.join(args.input_dir, image_file.replace('.jpg', '.json'))
        # Check if JSON file exists
        if not os.path.exists(json_file):
            print(f"Warning: JSON file {json_file} does not exist. Skipping image {image_file}.")
            continue
        # Load the JSON file
        with open(json_file, 'r') as f:
            json_data = json.load(f)
        # Extract the first question from 'text' field
        question_list = json_data.get('text', [])
        if not question_list:
            print(f"Warning: No 'text' field in JSON file {json_file}. Skipping image {image_file}.")
            continue
        question = question_list[0]
        print(f"Processing image: {image_file}")
        print(f"Question: {question}")
        # Format images
        img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_path)

        # Format Question
        conv = conversation_lib.conv_templates[args.conv_type].copy()
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
            "image_path": image_path,
            "images_clip": torch.stack([image_clip], dim=0),
            "images": torch.stack([image], dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape]
        }
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)
        output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )
        real_output_ids = output_ids[:, input_ids.shape[1] :]
        generated_outputs = tokenizer.batch_decode(
            real_output_ids, skip_special_tokens=True
        )[0]
        segmentation_dir = save_segmentation(pred_masks[0], input_dict, args)
        print("---------- Model Output ----------")
        print(f"* Object Existence (See): {object_presence[0]}")
        print(f"* Text Response (Say): {generated_outputs}")
        print(f"* Segmentation Paths (Segment): {segmentation_dir}")


def main(args):
    args = parse_args(args)

    print(args)

    demo(args)
if __name__ == "__main__":
    main(sys.argv[1:])

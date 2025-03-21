import argparse
import os
import sys
import json
import math
import numpy as np
import torch
from tqdm import tqdm

from PIL import Image

# ========== Import your SESAME, LLaVA, and utility modules as in eval_bootstraping.py or pred scripts ==========
from model.SESAME_eval import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad
from utils import prepare_input

def parse_args(args):
    parser = argparse.ArgumentParser(
        description="SESAME GeoBench-VLM Ref-Det Inference with Bounding Boxes"
    )
    parser.add_argument(
        "--qa_file",
        default="/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Ref-Det/qa.json",
        type=str,
        help="Path to the GEOBench-VLM Ref-Det QA file."
    )
    parser.add_argument(
        "--pretrained_model_path",
        default="/home/wenhan/Projects/sesame/runs/lisat_0223_v1/hg_model",
        type=str,
        help="Path to the pretrained SESAME model directory."
    )
    parser.add_argument(
        "--image_root",
        default="/home/wenhan/Projects/sesame/dataset/GEOBench-VLM",
        type=str,
        help="Root directory containing the 'images' folder."
    )
    parser.add_argument(
        "--output_json",
        default="/home/wenhan/Projects/sesame/pred_genbench/lisat_0223_v1/pred_geobench_ref_det.json",
        type=str,
        help="Where to save the final JSON predictions."
    )
    parser.add_argument(
        "--seg_save_dir",
        default="/home/wenhan/Projects/sesame/pred_genbench/lisat_0223_v1/seg_npz",
        type=str,
        help="Directory to save predicted segmentation masks as NPZ files."
    )
    parser.add_argument(
        "--image_size",
        default=1024,
        type=int,
        help="Size to which we resize the image for model input."
    )
    parser.add_argument(
        "--model_max_length",
        default=512,
        type=int,
        help="Maximum number of tokens for the text generation."
    )
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
        help="Type of conversation template to use."
    )
    return parser.parse_args(args)

@torch.inference_mode()
def run_inference(args):
    # 1. Prepare output directories
    os.makedirs(os.path.dirname(args.output_json), exist_ok=True)
    os.makedirs(args.seg_save_dir, exist_ok=True)

    # 2. Load model
    tokenizer, segmentation_lmm, vision_tower, context_len = load_pretrained_model_SESAME(
        model_path=args.pretrained_model_path
    )
    # Switch to bf16
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")
    tokenizer.padding_side = "left"

    # 3. Read the QA file
    with open(args.qa_file, "r") as f:
        qa_data = json.load(f)
    print(f"Loaded {len(qa_data)} Ref-Det QA entries from {args.qa_file}")

    # 4. Initialize image processor
    img_processor = ImageProcessor(
        vision_tower.image_processor,
        args.image_size
    )

    # 5. Prepare an output list
    all_predictions = []

    # 6. Run inference
    for idx, entry in enumerate(tqdm(qa_data, desc="Ref-Det Inference")):
        question_id = entry["question_id"]
        image_path_rel = entry["image_path"]   # e.g. "Ref-Det/images/ref_det_192.png"
        prompts = entry.get("prompts", [])
        if not prompts:
            prompt_text = "Which object is described?"
        else:
            prompt_text = prompts[0]

        # Full image path
        image_full_path = os.path.join(args.image_root, image_path_rel)
        if not os.path.exists(image_full_path):
            print(f"WARNING: {image_full_path} not found, skipping.")
            continue

        # 7. Load and preprocess the image
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_full_path)

        # 8. Build the conversation
        conv = conversation_lib.default_conversation.copy()
        user_msg = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
        conv.append_message(conv.roles[0], user_msg)
        conv.append_message(conv.roles[1], None)

        conversation_list = [conv.get_prompt()]

        # Some models need <im_start> / <im_end> tokens
        mm_use_im_start_end = getattr(segmentation_lmm.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            conversation_list = replace_image_tokens(conversation_list)

        # 9. Tokenize
        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')

        # 10. Prepare input dict
        input_dict = {
            "images_clip": image_clip.unsqueeze(0),  # shape (1, 3, H, W)
            "images": image.unsqueeze(0),            # shape (1, 3, H, W)
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape]
        }
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        # 11. Forward pass (inference)
        output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )

        # 12. Decode text
        real_output_ids = output_ids[:, input_ids.shape[1]:]
        decoded_text = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0].strip()

        # 13. The predicted segmentation mask & presence
        pred_mask = pred_masks[0].cpu().numpy()  # shape (H, W), 0/1 or float
        pred_exist_flag = bool(object_presence[0] > 0)  # or store the raw float

        # ---- Compute bounding box from the mask (in 1024×1024 scale) ----
        mask_bool = pred_mask > 0
        ys, xs = np.where(mask_bool)
        if len(ys) == 0:
            # No pixels predicted => no bounding box
            pred_bbox = [-1, -1, -1, -1]
        else:
            y_min, y_max = ys.min(), ys.max()
            x_min, x_max = xs.min(), xs.max()
            pred_bbox = [int(x_min), int(y_min), int(x_max), int(y_max)]
        # Format [xmin, ymin, xmax, ymax] in the same coordinate system as the mask.

        # 14. Optionally save the predicted mask in an NPZ
        mask_npz_name = f"pred_mask_{question_id:06d}.npz"
        mask_npz_path = os.path.join(args.seg_save_dir, mask_npz_name)
        np.savez_compressed(mask_npz_path, mask=pred_mask)

        # 15. Record this item in our JSON output
        result_dict = {
            "question_id": question_id,
            "image_path": image_path_rel,
            "prompt": prompt_text,
            "pred_text": decoded_text,
            "pred_exists": pred_exist_flag,
            "mask_npz": mask_npz_name,
            # The bounding box in [xmin, ymin, xmax, ymax] format:
            "pred_bbox": pred_bbox
        }
        all_predictions.append(result_dict)

    # 16. Write the final JSON
    with open(args.output_json, "w") as outf:
        json.dump(all_predictions, outf, indent=2)
    print(f"Done! Wrote predictions to {args.output_json} with bounding boxes and masks.")

def main(args):
    args = parse_args(args)
    run_inference(args)

if __name__ == "__main__":
    main(sys.argv[1:])

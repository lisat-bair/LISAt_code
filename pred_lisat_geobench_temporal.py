import argparse
import os
import sys
import math
import json
import torch
from tqdm import tqdm
from PIL import Image

# ========== Import your SESAME, LLaVA, and utility modules ==========
from model.SESAME_eval import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from utils import prepare_input
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad


def parse_args(args):
    parser = argparse.ArgumentParser(description="SESAME LISAT on GEOBench-VLM/Temporal Inference & Accuracy")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")

    # === Model & Paths ===
    parser.add_argument(
        "--pretrained_model_path",
        default="/home/wenhan/Projects/sesame/runs/lisat_0223_v1/hg_model",
        type=str,
        help="Path to the pretrained SESAME (LISAT) model"
    )
    parser.add_argument(
        "--image_root",
        default="/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Temporal/images",
        type=str,
        help="Root directory where the Temporal dataset images are stored"
    )
    parser.add_argument(
        "--question_file",
        default="/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Temporal/qa.json",
        type=str,
        help="Path to the Temporal QA JSON file (with multiple images per question)"
    )
    parser.add_argument(
        "--output_file",
        default="/home/wenhan/Projects/sesame/pred_genbench/lisat_0223_v1/pred_geobench_temporal.json",
        type=str,
        help="Output JSON lines file for model predictions"
    )

    # === Model Inference Settings ===
    parser.add_argument("--image_size", default=1024, type=int, help="Image input size")
    parser.add_argument("--model_max_length", default=512, type=int, help="Max tokens for the text generation")
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
        help="Conversation format (LLaVA variants)."
    )

    return parser.parse_args(args)


@torch.inference_mode()
def run_inference(args):
    """
    Performs multi-image classification on the GEOBench-VLM/Temporal dataset.
    Each sample in qa.json has 2 images + a multi-choice question (A-E).
    Prints accuracy and saves predictions.
    """

    # 1. Load the model (SESAME / LISAT)
    tokenizer, segmentation_lmm, vision_tower, context_len = load_pretrained_model_SESAME(
        model_path=args.pretrained_model_path
    )
    # Move to bfloat16 for efficiency
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    # Optional compile speedup (requires PyTorch 2.0+)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")
    tokenizer.padding_side = "left"

    # 2. Load the Temporal QA data
    with open(args.question_file, "r") as f:
        data = json.load(f)  # list of dicts

    # 3. Prepare output
    if args.output_file:
        output_dir = os.path.dirname(args.output_file)

        # Only create the directory if it is not empty
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

    out_fp = open(args.output_file, "w")

    # Counters for accuracy
    correct_count = 0
    total_count = 0

    # 4. Process each QA entry
    for entry in tqdm(data, desc="Processing GEOBench-VLM Temporal"):
        question_id = entry["question_id"]
        image_paths = entry["image_path"]  # e.g. ["Temporal/images/temporal_5642.png", "Temporal/images/temporal_2186.png"]
        options_str = entry["options"]      # e.g. "A. Snow ... B. Precip ... C. Human ..."
        ground_truth_option = entry["ground_truth_option"]  # e.g. "B"
        prompts = entry.get("prompts", [])
        question_prompt = prompts[0] if prompts else "What is the correct answer?"

        # Make sure ground_truth_option is one of [A-E] or None
        # Some data might have an empty or missing ground truth

        # 5. Construct the multi-choice prompt for the model
        full_question_text = (
            f"{question_prompt}\n\n"
            f"Possible answers:\n"
            f"{options_str}\n"
            "Please choose the best option (A, B, C, D, or E)."
        )

        # 6. Load and preprocess both images
        if len(image_paths) < 2:
            # If there's any mismatch or only one image, skip or handle differently
            print(f"Warning: question {question_id} does not have 2 images, skipping.")
            continue

        images_processed = []
        images_clip_processed = []
        sam_mask_shapes = []

        for img_rel_path in image_paths:
            # Join with image_root (and possibly extract basename)
            img_full_path = os.path.join(args.image_root, os.path.basename(img_rel_path))
            if not os.path.exists(img_full_path):
                print(f"Warning: Image not found: {img_full_path}")
                continue

            # Use the ImageProcessor
            img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
            image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(img_full_path)
            images_processed.append(image)
            images_clip_processed.append(image_clip)
            sam_mask_shapes.append(sam_mask_shape)

        if len(images_processed) < 2:
            print(f"Skipping question {question_id} due to missing images.")
            continue

        # 7. Build the conversation (LLaVA style)
        conv = conversation_lib.default_conversation.copy()
        # Insert <Image> token + question
        question_with_image = DEFAULT_IMAGE_TOKEN + "\n" + full_question_text
        conv.append_message(conv.roles[0], question_with_image)
        conv.append_message(conv.roles[1], None)

        conversation_list = [conv.get_prompt()]

        # If the model uses <im_start> and <im_end> tokens, do so
        mm_use_im_start_end = getattr(segmentation_lmm.config, "mm_use_im_start_end", False)
        if mm_use_im_start_end:
            conversation_list = replace_image_tokens(conversation_list)

        # 8. Tokenize text input
        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding='left')

        # 9. Prepare the model input
        #    We'll pass the two images as a batch of size 2
        #    The LISAT model can handle multiple images if designed to do so (check your code).
        #    If your model only processes 1 image at a time, you may need an alternative approach.
        #    We'll assume it can handle them as a batch or merges them internally.
        input_dict = {
            "images_clip": torch.stack(images_clip_processed, dim=0),
            "images": torch.stack(images_processed, dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": sam_mask_shapes
        }
        # Move data to GPU in bfloat16
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        # 10. Forward pass
        output_ids, pred_masks, object_presence = segmentation_lmm.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )

        # 11. Decode model output
        real_output_ids = output_ids[:, input_ids.shape[1]:]
        model_output_text = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0].strip()

        # 12. Extract predicted option (A-E) from text
        predicted_option = None
        for letter in ["A", "B", "C", "D", "E"]:
            if letter in model_output_text:
                predicted_option = letter
                break

        if predicted_option is None:
            predicted_option = "UNDEF"

        # 13. Check correctness
        is_correct = (predicted_option == ground_truth_option)
        if ground_truth_option in ["A", "B", "C", "D", "E"]:
            total_count += 1
            if is_correct:
                correct_count += 1

        # 14. Save prediction
        out_fp.write(json.dumps({
            "question_id": question_id,
            "image_paths": image_paths,
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

    # 16. Report accuracy
    accuracy = 0.0
    if total_count > 0:
        accuracy = correct_count / total_count
    print(f"Accuracy on GEOBench-VLM Temporal: {accuracy*100:.2f}% "
          f"({correct_count}/{total_count} correct)")


def main(args):
    args = parse_args(args)
    run_inference(args)


if __name__ == "__main__":
    main(sys.argv[1:])

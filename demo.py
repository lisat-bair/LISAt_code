# demo.py

import argparse
import os
import sys
import cv2
import numpy as np
import torch

from model.LISAT import load_pretrained_model_LISAT
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN
from dataloaders.base_dataset import ImageProcessor
from dataloaders.utils import replace_image_tokens, tokenize_and_pad
from utils import prepare_input


def parse_args(args):
    parser = argparse.ArgumentParser(description="Run LISAT in interactive demo mode")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--pretrained_model_path", default="./runs/lisat/hg_model")
    parser.add_argument("--vis_save_dir", default="./demo_output", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        choices=["llava_v1", "llava_llama_2"],
        help="Conversation prompt template type"
    )
    return parser.parse_args(args)


def save_segmentation(pred_mask, input_dict, args):
    """
    Save predicted segmentation mask and an RGB overlay visualization.
    """
    pred_mask = (pred_mask.detach().cpu().numpy() > 0).astype(np.uint8)

    image_path = input_dict["image_path"]
    image_key = os.path.splitext(os.path.basename(image_path))[0]
    save_dir = os.path.join(args.vis_save_dir, image_key)
    os.makedirs(save_dir, exist_ok=True)

    # Save mask as grayscale
    seg_mask_path = os.path.join(save_dir, "seg_mask.jpg")
    cv2.imwrite(seg_mask_path, pred_mask * 100)

    # Overlay mask on original image
    rgb_path = os.path.join(save_dir, "seg_rgb.jpg")
    image_np = cv2.imread(image_path)
    overlay = (image_np * 0.3 + pred_mask[:, :, None] * np.array([0, 0, 255]) * 0.7).astype(np.uint8)
    image_np[pred_mask == 1] = overlay[pred_mask == 1]
    cv2.imwrite(rgb_path, image_np)

    return save_dir


@torch.inference_mode()
def demo(args):
    """
    Interactive segmentation + reasoning demo with LISAT.
    """
    os.makedirs(args.vis_save_dir, exist_ok=True)

    tokenizer, model, vision_tower, context_len = load_pretrained_model_LISAT(
        model_path=args.pretrained_model_path
    )
    model = model.to(torch.bfloat16)
    vision_tower = vision_tower.to(torch.bfloat16)
    model = torch.compile(model, mode="reduce-overhead")

    tokenizer.padding_side = "left"

    while True:
        print("\n========== User Input ==========")
        print("Press Ctrl-C to exit.")
        question = input("Prompt: ").strip()
        image_path = input("Image path: ").strip()

        # Preprocess image
        img_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(image_path)

        # Build conversation
        conv = conversation_lib.default_conversation.copy()
        full_question = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], full_question)
        conv.append_message(conv.roles[1], None)
        conversation_list = [conv.get_prompt()]

        if getattr(model.config, "mm_use_im_start_end", False):
            conversation_list = replace_image_tokens(conversation_list)

        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding="left")

        input_dict = {
            "image_path": image_path,
            "images_clip": torch.stack([image_clip]),
            "images": torch.stack([image]),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape],
        }
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        # Run model
        output_ids, pred_masks, object_presence = model.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=args.model_max_length,
        )

        # Decode and save results
        real_output_ids = output_ids[:, input_ids.shape[1]:]
        response = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0]
        seg_dir = save_segmentation(pred_masks[0], input_dict, args)

        print("\n========== Model Output ==========")
        print(f"[See]  Object Exists: {object_presence[0]}")
        print(f"[Say]  Response: {response}")
        print(f"[Segment] Saved at: {seg_dir}")


def main(args):
    args = parse_args(args)
    print(args)
    demo(args)


if __name__ == "__main__":
    main(sys.argv[1:])

import argparse
import os
import sys
import json
import math
import numpy as np
import torch
import torch.nn.functional as F
import cv2

from functools import partial
from torch.utils.data import Dataset, DataLoader, Subset
from tqdm import tqdm
from PIL import Image

# SESAME (LISAT) model loading and conversation logic:
from model.SESAME_eval import load_pretrained_model_SESAME
from model.llava import conversation as conversation_lib
from model.llava.constants import DEFAULT_IMAGE_TOKEN

# Tokenization / utility
from dataloaders.utils import replace_image_tokens, tokenize_and_pad
from utils import prepare_input

# For image preprocessing
from dataloaders.base_dataset import ImageProcessor


###############################################################################
# 1) RefSegDataset that actually loads and preprocesses images in __getitem__
###############################################################################
class RefSegDataset(Dataset):
    """
    Reads a QA JSON (example):
    [
      {
        "image_path": "Ref-Seg/images/ref_seg_79.jp2",
        "ground_truth": "Ref-Seg/masks/mask_ref_seg_452.png",
        "prompts": [...],
        "question_id": 0
      },
      ...
    ]
    Then calls self.image_processor to get (image, image_clip, sam_mask_shape).
    """

    def __init__(self, qa_file, image_root, image_processor, image_size):
        with open(qa_file, "r") as f:
            self.qa_data = json.load(f)
        self.image_root = image_root
        self.image_processor = image_processor
        self.image_size = image_size

    def __len__(self):
        return len(self.qa_data)

    def __getitem__(self, index):
        entry = self.qa_data[index]
        image_rel = entry["image_path"]     # e.g. "Ref-Seg/images/ref_seg_79.jp2"
        gt_mask_rel = entry["ground_truth"] # e.g. "Ref-Seg/masks/mask_ref_seg_452.png"
        prompts = entry["prompts"]
        question_id = entry["question_id"]

        # --- Remove the leading "Ref-Seg/" if present to avoid doubling ---
        image_rel_fixed = image_rel.replace("Ref-Seg/", "")
        gt_mask_rel_fixed = gt_mask_rel.replace("Ref-Seg/", "")

        # Construct the final path for the image
        full_img_path = os.path.join(self.image_root, image_rel_fixed)
        # e.g. "/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Ref-Seg/images/ref_seg_79.jp2"

        # Use self.image_processor.load_and_preprocess_image to get the Tensors
        # We'll handle the possibility that cv2 can't read jp2 by letting the image_processor do fallback or raise an error
        if not os.path.exists(full_img_path):
            print(f"[Warning] Image not found: {full_img_path}")
        image, image_clip, sam_mask_shape = self.image_processor.load_and_preprocess_image(full_img_path)

        # Build the path for the ground truth mask as well
        full_mask_path = os.path.join(self.image_root, gt_mask_rel_fixed)
        if not os.path.exists(full_mask_path):
            # We won't raise an error, just store a dummy
            print(f"[Warning] GT mask not found: {full_mask_path}")
            gt_array = np.zeros(sam_mask_shape, dtype=np.uint8)
        else:
            # Load & binarize
            gt_pil = Image.open(full_mask_path).convert("L")
            gt_pil = gt_pil.resize((sam_mask_shape[1], sam_mask_shape[0]), Image.NEAREST)
            gt_array = np.array(gt_pil)
            gt_array = (gt_array > 128).astype(np.uint8)

        return {
            "image": image,                  # Tensor shape (3, H, W)
            "image_clip": image_clip,        # Possibly the same shape, or preprocessed
            "sam_shape": sam_mask_shape,     # (H, W)
            "gt_mask": gt_array,            # np.uint8
            "prompts": prompts,
            "question_id": question_id
        }


###############################################################################
# 2) The collate function that matches 'eval_bootstraping.py' structure
###############################################################################
def collate_fn_refseg(batch, tokenizer, use_mm_start_end, padding="left"):
    """
    Each item has:
      {
        "image": Tensor (3,H,W),
        "image_clip": Tensor (3,H,W),
        "sam_shape": (H,W),
        "gt_mask": np array shape (H,W),
        "prompts": [...],
        "question_id": ...
      }

    We produce a dictionary with:
      "images": shape (B,3,H,W)
      "images_clip": shape (B,3,H,W)
      "masks_list": [ torch.Size(B, H, W) ]
      "sam_mask_shape_list": list of (H,W) with length B?
      "conversation_list", "input_ids", "exists", ...
    """
    # We'll hold the "batch" dimension for the images,
    # but the prompts can produce multiple queries. We'll flatten them in the inference loop.

    # We must produce a single item for each batch in "eval_bootstraping" style:
    # In that code, we do: "for input_dict in test_loader:", then for each input_dict we might have multiple queries inside.

    # So let's gather everything for the entire batch:
    B = len(batch)

    images_list = []
    images_clip_list = []
    sam_shapes = []
    masks_list = []
    # We'll store *all* queries from *all* samples in one big conversation list, etc.
    # However, "eval_bootstraping.py" typically processes "N" queries at once. Let's replicate that logic:
    conversation_list = []
    input_ids_list = []
    exists_list = []
    ref_ids = []
    sent_ids = []

    # We'll also store a big stacked GT mask for all queries. But we can keep only 1 per prompt or flatten them. 
    # In "eval_bootstraping.py," it uses "len(input_dict['input_ids'])" for the queries, and "gt" is repeated. 
    # We'll do the simpler approach: "B=1" is typical. But let's just store them all once. We can do:
    #   "masks_list" => [ stacked_mask_of_shape (total_queries, H, W) ]
    # But if we are following the EXACT pattern, we store them at the item level. We'll store one for each item in the batch.

    # We'll gather them in a list. Then "eval_bootstraping.py" index = # queries. 
    # Actually, "eval_bootstraping" either uses reason_seg_dataset or refer_seg_dataset, which splits each item into multiple queries. 
    # We'll do that approach. But let's keep it simpler:
    # We'll assume each item => multiple prompts => we store them in the "conversation_list" and "input_ids".

    # For "exists," we might store a list of length #prompts. We'll flatten them.

    for i, sample in enumerate(batch):
        images_list.append(sample["image"])           # shape (3, H, W)
        images_clip_list.append(sample["image_clip"]) # shape (3, H, W)
        sam_shapes.append(sample["sam_shape"])
        masks_list.append(sample["gt_mask"])

        # We'll flatten the prompts into multiple queries
        prompts = sample["prompts"]
        for prompt_text in prompts:
            user_msg = DEFAULT_IMAGE_TOKEN + "\n" + prompt_text
            conv = conversation_lib.default_conversation.copy()
            conv.append_message(conv.roles[0], user_msg)
            conv.append_message(conv.roles[1], None)
            conv_list = [conv.get_prompt()]

            if use_mm_start_end:
                conv_list = replace_image_tokens(conv_list)

            tokens, _ = tokenize_and_pad(conv_list, tokenizer, padding=padding)
            conversation_list.append(conv_list[0])
            input_ids_list.append(tokens)
            exists_list.append(1)  # by default
            ref_ids.append(None)
            sent_ids.append(None)

    # Now, "eval_bootstraping" expects "masks_list" to be a list with 1 element containing a stacked (N, H, W).
    # But we only have B images, each with multiple prompts. The total #prompts is N = sum of all prompts across the batch.
    # We need to replicate the GT mask for each prompt in that item. So let's do that now:

    # We'll replicate each masks_list[i] for the # of prompts in that item.
    # We can track how many prompts belong to each sample, but let's do a simpler approach: We'll just re-scan them.
    # The "final" structure is "masks_list": [ torch.Size( (N, H, W) ) ].

    # We already have B items, each with len(prompts_i). We'll do a second pass:
    total_queries = len(input_ids_list)
    # We'll store them in expanded_gt: shape (total_queries, H, W)
    expanded_gt = []
    idx_prompts = 0
    index_in_input_ids = 0

    # Let's do an approach: for each sample, replicate the mask for len(prompts).
    # But we need the #prompts for each sample. We'll gather them now:
    # This is the same approach used above when we built conversation_list, but let's store the counts.
    counts_per_sample = []
    for sample in batch:
        counts_per_sample.append(len(sample["prompts"]))

    pointer = 0
    for i, sample in enumerate(batch):
        gt_array_i = sample["gt_mask"]   # shape (H, W)
        n_q = len(sample["prompts"])
        for _ in range(n_q):
            t_mask = torch.from_numpy(gt_array_i).long()
            expanded_gt.append(t_mask)

    if len(expanded_gt) > 0:
        stacked_gt = torch.stack(expanded_gt, dim=0)  # shape (N, H, W)
    else:
        stacked_gt = torch.zeros((0,1,1), dtype=torch.long)

    out_dict = {
        # images/images_clip => shape (B, 3, H, W) in float32 or float16
        "images": torch.stack(images_list, dim=0),
        "images_clip": torch.stack(images_clip_list, dim=0),

        "conversation_list": conversation_list,       # length N
        "input_ids": input_ids_list,                  # list of length N
        "exists": [exists_list],                      # shape [1, N]
        "masks_list": [stacked_gt],                   # single item => shape (N, H, W)
        "sam_mask_shape_list": None,                  # We'll fill in a flattened shape list
        "image_paths": [],                            # We'll fill in a flattened or partial path
        "ref_ids": ref_ids,
        "sent_ids": sent_ids
    }

    # "sam_mask_shape_list" is expected to be length N, so we replicate each item for each prompt
    # same for "image_paths". We'll do that:
    expanded_sam_shapes = []
    expanded_image_paths = []

    pointer = 0
    for i, sample in enumerate(batch):
        n_q = len(sample["prompts"])
        for _ in range(n_q):
            expanded_sam_shapes.append(sample["sam_shape"])
            # We can store a placeholder or real path. We don't currently store the absolute path in the item, 
            # but if we want, we can do that. We'll store "image_i" or something:
            expanded_image_paths.append(f"image_{i:03d}")

    out_dict["sam_mask_shape_list"] = expanded_sam_shapes
    out_dict["image_paths"] = expanded_image_paths

    return out_dict


###############################################################################
# 3) The main "inference" function, matching the pattern from eval_bootstraping.py
###############################################################################
def inference(args):
    os.makedirs(args.vis_save_path, exist_ok=True)
    os.makedirs(os.path.join(args.vis_save_path, "segmentation_mask"), exist_ok=True)

    # 1) Load model
    tokenizer, segmentation_lmm, vision_tower, context_len = load_pretrained_model_SESAME(
        model_path=args.pretrained_model_path
    )
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")
    tokenizer.padding_side = "left"

    # 2) Build dataset
    image_processor = ImageProcessor(vision_tower.image_processor, args.image_size)
    dataset = RefSegDataset(
        qa_file=args.qa_file,
        # Typically: "/home/wenhan/Projects/sesame/dataset/GEOBench-VLM/Ref-Seg"
        image_root=os.path.dirname(args.qa_file),
        image_processor=image_processor,
        image_size=args.image_size
    )

    # Possibly slice for multi-GPU
    dataset = get_dataset_slice(dataset, args.process_num, args.world_size, debug=False)

    # 3) Build DataLoader
    loader = DataLoader(
        dataset,
        batch_size=1,  # each item is 1 image with multiple prompts
        shuffle=False,
        drop_last=False,
        num_workers=1,
        pin_memory=False,
        collate_fn=partial(
            collate_fn_refseg,
            tokenizer=tokenizer,
            use_mm_start_end=getattr(segmentation_lmm.config, "mm_use_im_start_end", False),
            padding="left"
        ),
    )

    idx_counter = 0
    output_json = {}

    # 4) Loop
    for input_dict in tqdm(loader, desc="Ref-Seg Inference"):
        idx_counter += 1
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)

        N = len(input_dict["input_ids"])  # total queries across all prompts in this batch item
        batch_size = 5
        num_batch = math.ceil(N / batch_size)

        pred_masks_list = []
        pred_exists_list = []

        gt_masks = input_dict["masks_list"][0].int()  # shape (N, H, W)

        # We'll pick the first path as the key, or do something else
        if len(input_dict["image_paths"]) > 0:
            image_file_key = input_dict["image_paths"][0]
        else:
            image_file_key = f"image_{idx_counter:03d}"

        print(f"Processing {image_file_key} with {N} queries.")

        raw_questions = []
        for conv_text in input_dict["conversation_list"]:
            # Typically "<image>\nPROMPT"
            parts = conv_text.split("\n", 1)
            if len(parts) > 1:
                user_part = parts[1]
                if "ASSISTANT:" in user_part:
                    user_part = user_part.split("ASSISTANT:")[0].strip()
                raw_questions.append(user_part)
            else:
                raw_questions.append(conv_text)

        # Prepare output structure
        output_json[image_file_key] = {
            "conversation_list": raw_questions,
            "pred_sent": [],
            "gt_exists": input_dict["exists"][0],  # shape (N,)
            "ref_ids": input_dict["ref_ids"],
            "sent_ids": input_dict["sent_ids"]
        }

        # 4b) sub-batch loop
        for b_idx in range(num_batch):
            start_idx = b_idx * batch_size
            end_idx = min((b_idx + 1) * batch_size, N)

            input_ids = torch.stack(input_dict["input_ids"][start_idx:end_idx], dim=0)
            real_bs = input_ids.shape[0]

            # replicate images
            images_clip = input_dict["images_clip"].repeat(real_bs, 1, 1, 1)
            images = input_dict["images"].repeat(real_bs, 1, 1, 1)

            sam_mask_shape_list = input_dict["sam_mask_shape_list"][start_idx:end_idx]

            with torch.inference_mode():
                output_ids, pred_masks_batch, object_presence = segmentation_lmm.evaluate(
                    images_clip,
                    images,
                    input_ids,
                    sam_mask_shape_list,
                    max_new_tokens=512,
                )
                pred_exists_list += object_presence
                pred_masks_list += pred_masks_batch

                real_output_ids = output_ids[:, input_ids.shape[1]:]
                generated_outputs = tokenizer.batch_decode(
                    real_output_ids, skip_special_tokens=True
                )
                output_json[image_file_key]["pred_sent"] += generated_outputs

        # 4c) stack predicted masks
        pred_masks_tensor = torch.stack(pred_masks_list, dim=0)
        output_json[image_file_key]["pred_exists"] = pred_exists_list

        seg_fname_rel = os.path.join("segmentation_mask", f"ref_seg_{idx_counter:04d}.npz")
        seg_fname_abs = os.path.join(args.vis_save_path, seg_fname_rel)
        np.savez_compressed(
            seg_fname_abs,
            pred=pred_masks_tensor.cpu().numpy(),
            gt=gt_masks.cpu().numpy()
        )
        output_json[image_file_key]["segmentation_path"] = seg_fname_rel

    # 5) Write partial shards + final JSON
    save_preds("RefSeg", output_json, args.process_num, args.world_size, args.vis_save_path)
    if args.process_num == 0:
        with open(args.output_json, "w") as f:
            json.dump(output_json, f, indent=2)
        print(f"[Ref-Seg] Combined predictions saved to {args.output_json}")


###############################################################################
# The usual helper methods from eval_bootstraping.py
###############################################################################
def parse_args(args):
    parser = argparse.ArgumentParser(description="LISAT Ref-Seg Inference with 'images' fix")
    parser.add_argument("--cmd", default="inference", type=str)
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--process_num", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--pretrained_model_path", type=str, default="/path/to/hg_model")
    parser.add_argument("--vis_save_path", type=str, default="./refseg_save")
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--model_max_length", type=int, default=512)
    parser.add_argument("--val_dataset", type=str, default="RefSeg")
    parser.add_argument("--dataset_dir", type=str, default="./dataset")
    parser.add_argument("--qa_file", type=str, default="./dataset/GEOBench-VLM/Ref-Seg/qa.json")
    parser.add_argument("--output_json", type=str, default="./pred_geobench_ref_seg.json")
    return parser.parse_args(args)


def get_dataset_slice(val_dataset, process_num: int, world_size: int, debug: bool = False):
    indices = np.arange(len(val_dataset))
    splits = np.array_split(indices, world_size)
    subset_indices = splits[process_num]
    if debug:
        subset_indices = subset_indices[:10]
    return Subset(val_dataset, indices=subset_indices)


def save_preds(val_dataset, preds, process_num, world_size, inference_dir):
    filename = f"preds_{val_dataset}_{process_num}_of_{world_size}.json"
    file_path = os.path.join(inference_dir, filename)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, "w") as f:
        json.dump(preds, f)


def load_preds_file(val_dataset, process_num, world_size, inference_dir):
    filename = f"preds_{val_dataset}_{process_num}_of_{world_size}.json"
    file_path = os.path.join(inference_dir, filename)
    with open(file_path, "rb") as file:
        return json.load(file)


def main():
    args = parse_args(sys.argv[1:])
    print("[Ref-Seg] Args:", args)
    if args.cmd == "inference":
        inference(args)
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main()

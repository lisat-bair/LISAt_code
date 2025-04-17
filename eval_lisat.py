import argparse
import os
import sys
import json
from functools import partial
import tqdm
import math
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Subset
from dataloaders.test_dataset import TestReasoningDataset, TestReferDataset, collate_fn_test
from model.LISAT_eval import load_pretrained_model_LISAT
from utils import (
    prepare_input,
    intersectionAndUnionGPU,
    AverageMeter,
    Summary,
)
from scipy.spatial import ConvexHull  # used for geometric IoU metrics


def parse_args(args):
    parser = argparse.ArgumentParser(description="LISAT Evaluation Script")
    parser.add_argument("--cmd", default="inference", type=str, help="inference | metrics | bootstrap_metrics | download")
    parser.add_argument("--local_rank", default=0, type=int)
    parser.add_argument("--process_num", default=0, type=int)
    parser.add_argument("--world_size", default=1, type=int)

    parser.add_argument("--pretrained_model_path", default="./runs/lisat/hg_model")
    parser.add_argument("--vis_save_path", default="./inference_dir/lisat", type=str)
    parser.add_argument("--image_size", default=1024, type=int)
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--val_dataset", default="GeoReasonSeg", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--conv_type", default="llava_v1", choices=["llava_v1", "llava_llama_2"])
    return parser.parse_args(args)


def save_preds(val_dataset, preds, process_num, world_size, inference_dir):
    filename = f"preds_{val_dataset}_{process_num}_of_{world_size}.json"
    file_path = os.path.join(inference_dir, filename)
    with open(file_path, "w") as file:
        json.dump(preds, file)


def load_preds_file(val_dataset, process_num, world_size, inference_dir):
    filename = f"preds_{val_dataset}_{process_num}_of_{world_size}.json"
    file_path = os.path.join(inference_dir, filename)
    with open(file_path, "rb") as file:
        return json.load(file)


def inference(args):
    # Initialization
    os.makedirs(args.vis_save_path, exist_ok=True)
    os.makedirs(os.path.join(args.vis_save_path, "segmentation_mask"), exist_ok=True)

    (
        tokenizer,
        segmentation_lmm,
        vision_tower,
        context_len,
    ) = load_pretrained_model_LISAT(
        model_path=args.pretrained_model_path
    )
    # Load bf16 datatype
    vision_tower = vision_tower.to(torch.bfloat16)
    segmentation_lmm = segmentation_lmm.to(torch.bfloat16)
    segmentation_lmm = torch.compile(segmentation_lmm, mode="reduce-overhead")

    # for eval only
    tokenizer.padding_side = "left"
    reason_seg_dataset = ["GeoReasonSeg"]
    refer_seg_dataset = [
        "fprefcoco", "fprefcoco+", "fprefcocog",
        "refcoco", "refcoco+", "refcocog",
        "R-refcoco", "R-refcoco+", "R-refcocog"
    ]
    if args.val_dataset in reason_seg_dataset:
        test_dataset = TestReasoningDataset(
            args.dataset_dir,
            vision_tower.image_processor,
            args.image_size,
            datasetname=args.val_dataset,
            train_test_split="test" # modify to "large" or "small" if you want to eval on splitted eval dataset
        )
    elif args.val_dataset in refer_seg_dataset:
        test_dataset = TestReferDataset(
            args.dataset_dir,
            vision_tower.image_processor,
            args.image_size,
            datasetname=args.val_dataset,
            train_test_split="val"
        )
    else:
        raise ValueError(f"Unsupported val_dataset: {args.val_dataset}")

    test_dataset = get_dataset_slice(test_dataset, args.process_num, args.world_size, debug=False)
    test_loader = DataLoader(
        test_dataset, batch_size=1, num_workers=1,
        shuffle=False, drop_last=False, pin_memory=False,
        collate_fn=partial(
            collate_fn_test,
            tokenizer=tokenizer,
            use_mm_start_end=getattr(segmentation_lmm.config, "mm_use_im_start_end", False),
            padding="left",
        ),
    )

    idx = 0
    output_json = {}

    # Process each item in the test loader
    for input_dict in tqdm.tqdm(test_loader):
        idx += 1
        input_dict = prepare_input(input_dict, "bf16", is_cuda=True)
        N = len(input_dict["input_ids"])
        assert N == len(input_dict["exists"][0])

        batch_size = 5
        num_batch = math.ceil(N / batch_size)

        pred_masks = []
        pred_exists = []
        gt_masks = input_dict["masks_list"][0].int()

        # Prepare output JSON structure
        image_file_key = input_dict["image_paths"][0]
        output_json[image_file_key] = {}
        print(f"Processing image {image_file_key} with {N} questions.")

        raw_questions = [
            x.split("\n")[1].split("ASSISTANT:")[0].strip()
            for x in input_dict["conversation_list"]
        ]
        output_json[image_file_key] = {
            "conversation_list": raw_questions,
            "pred_sent": [],
            "gt_exists": input_dict["exists"][0],
            "ref_ids": input_dict["ref_ids"],
            "sent_ids": input_dict["sent_ids"]
        }

        # Batch inference to prevent OOM
        for n in range(num_batch):
            start_idx = n * batch_size
            end_idx = min((n + 1) * batch_size, N)
            input_ids = input_dict["input_ids"][start_idx:end_idx]
            real_batch_size = input_ids.shape[0]

            images_clip = input_dict["images_clip"].repeat(real_batch_size, 1, 1, 1)
            images = input_dict["images"].repeat(real_batch_size, 1, 1, 1)
            sam_mask_shape_list = input_dict["sam_mask_shape_list"] * real_batch_size

            with torch.inference_mode():
                output_ids, pred_masks_batch, object_presence = segmentation_lmm.evaluate(
                    images_clip,
                    images,
                    input_ids,
                    sam_mask_shape_list,
                    max_new_tokens=512,
                )
                pred_exists += object_presence
                pred_masks += pred_masks_batch

                real_output_ids = output_ids[:, input_ids.shape[1]:]
                generated_outputs = tokenizer.batch_decode(
                    real_output_ids, skip_special_tokens=True
                )
                output_json[image_file_key]["pred_sent"] += generated_outputs

        # Save results
        pred_masks = torch.stack(pred_masks, dim=0)
        output_json[image_file_key]["pred_exists"] = pred_exists
        output_json[image_file_key]["segmentation_path"] = os.path.join(
            "segmentation_mask", f"{args.process_num}_{idx:04d}.npz"
        )
        output_seg_fname = os.path.join(
            args.vis_save_path,
            output_json[image_file_key]["segmentation_path"],
        )
        np.savez_compressed(
            output_seg_fname,
            pred=pred_masks.cpu().numpy(),
            gt=gt_masks.cpu().numpy(),
        )

    save_preds(args.val_dataset, output_json, args.process_num, args.world_size, args.vis_save_path)


def get_dataset_slice(
    val_dataset, process_num: int, world_size: int, debug: bool = False
) -> Subset:
    """
    Return a torch.utils.data.Subset object that is a subset of val_dataset.
    val_dataset is broken into roughly equal chunks. Number of chunks = world_size.
    The Nth chunk is returned (where N = process_num).
    """
    all_indices = np.array(range(len(val_dataset)))
    print(f"Splitting Total images: {len(all_indices)}, world_size: {world_size}")
    splits = np.array_split(all_indices, world_size)
    print("Total splits generated: ", len(splits))
    print(
        "Split sizes (each size is for one process/process_num): ",
        [len(s) for s in splits],
    )
    subset_indices = splits[process_num]
    if debug:
        # Force the dataset size to be tiny so we can test the script
        subset_indices = subset_indices[:10]
    print("Indices for current process: ", len(subset_indices))
    subset = Subset(val_dataset, indices=subset_indices)
    print("Subset dataset size: ", len(subset))
    return subset


def run_eval(preds, args):
    """
    This evaluates cIoU, gIoU, and See Accuracy over *all* keys in `preds`.
    """
    return run_eval_subset(preds, list(preds.keys()), args)


### Helper to evaluate a *subset* of keys ###
def run_eval_subset(preds, subset_keys, args):
    """
    Evaluate cIoU, gIoU, and See Accuracy, but only for the specified subset of keys.
    This is reused both by the 'run_eval' (all keys) and the bootstrap approach (subset).
    """
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    total_count = 0
    correct_query = 0

    for idx, pred_key in enumerate(subset_keys):
        pred = preds[pred_key]
        # Evaluate SEE (binary classification)
        correct_this_round = np.sum(
            np.array(pred["gt_exists"]) == np.array(pred["pred_exists"])
        )
        total_count += len(pred["gt_exists"])
        correct_query += correct_this_round

        # Evaluate Segment (IoU and GIoU)
        seg_fname = os.path.join(args.vis_save_path, pred["segmentation_path"])
        segmentation = np.load(seg_fname)
        pred_masks = torch.from_numpy(segmentation["pred"]).cuda().int()
        gt_masks = torch.from_numpy(segmentation["gt"]).cuda().int()

        batch_size = pred_masks.shape[0]
        intersection = torch.zeros(2).cuda()
        union = torch.zeros(2).cuda()
        acc_iou = torch.zeros(2).cuda()

        for i in range(batch_size):
            output_i = pred_masks[i].contiguous().clone()
            mask_i = gt_masks[i].contiguous()
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i, mask_i, 2, ignore_index=255
            )
            acc_iou_i = intersection_i / (union_i + 1e-5)
            # Handle no-object target
            acc_iou_i[union_i == 0] += 1.0

            intersection += intersection_i
            union += union_i
            acc_iou += acc_iou_i

        # Update meters
        intersection_meter.update(intersection.cpu().numpy())
        union_meter.update(union.cpu().numpy())
        acc_iou_meter.update((acc_iou / batch_size).cpu().numpy(), n=batch_size)

    # Final result
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]
    detection_acc = correct_query / total_count

    return ciou, giou, detection_acc


### Bootstrap evaluation function ###
def run_bootstrap_eval(preds, args, n_round=20, fraction=0.8):
    """
    Perform N-round bootstrap evaluation, each time sampling 80% (default) of the dataset keys.
    Return arrays or final prints of cIoU/gIoU means and stds.
    """
    all_keys = list(preds.keys())
    num_samples = int(len(all_keys) * fraction)

    ciou_list = []
    giou_list = []
    see_list = []

    for i in range(n_round):
        # Sample keys with replacement
        subset_keys = np.random.choice(all_keys, size=num_samples, replace=True)
        ciou, giou, see_acc = run_eval_subset(preds, subset_keys, args)
        ciou_list.append(ciou)
        giou_list.append(giou)
        see_list.append(see_acc)
        print(f"[Round {i+1}/{n_round}] cIoU={ciou:.4f}, gIoU={giou:.4f}")

    # Compute final stats
    ciou_array = np.array(ciou_list)
    giou_array = np.array(giou_list)
    see_array = np.array(see_list)

    ciou_mean, ciou_std = ciou_array.mean(), ciou_array.std()
    giou_mean, giou_std = giou_array.mean(), giou_array.std()
    see_mean, see_std = see_array.mean(), see_array.std()

    print("\n====== Bootstrap Results ({} rounds, {:.0f}% sampling) ======".format(n_round, fraction*100))
    print(f"cIoU: mean={ciou_mean:.4f}, std={ciou_std:.4f}")
    print(f"gIoU: mean={giou_mean:.4f}, std={giou_std:.4f}")

    # Return if desired
    return ciou_mean, ciou_std, giou_mean, giou_std, see_mean, see_std


def main(args):
    args = parse_args(args)
    print(args)

    if args.cmd == "inference":
        inference(args)

    elif args.cmd == "metrics":
        # Combine preds from all ranks, then evaluate over all
        combined_preds = {}
        for rank in range(args.world_size):
            combined_preds |= load_preds_file(args.val_dataset, rank, args.world_size, args.vis_save_path)
        ciou, giou, see_acc = run_eval(combined_preds, args)
        print(f"[Final Result - {args.val_dataset}] cIoU: {ciou:.4f} | GIoU: {giou:.4f}")

    ### Bootstrap evaluation entry point ###
    elif args.cmd == "bootstrap_metrics":
        # Combine preds from all ranks, then run bootstrap procedure
        combined_preds = {}
        for rank in range(args.world_size):
            combined_preds |= load_preds_file(args.val_dataset, rank, args.world_size, args.vis_save_path)

        run_bootstrap_eval(combined_preds, args, n_round=20, fraction=0.8)

    elif args.cmd == "download":
        print("Your custom download code here.")
    else:
        raise ValueError(f"Unknown cmd: {args.cmd}")


if __name__ == "__main__":
    main(sys.argv[1:])

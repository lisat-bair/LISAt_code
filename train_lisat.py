# Launch flash attention (optional)

# from model.llava.train.llama_flash_attn_monkey_patch import (
# replace_llama_attn_with_flash_attn,
# )

# replace_llama_attn_with_flash_attn()


import argparse
import os
import shutil
import sys
from functools import partial
import deepspeed
import torch
import tqdm
import wandb
import json
from PIL import Image

# Model & Data
from model.LISAT import init_LISAT_model
from model.llava import conversation as conversation_lib
from dataloaders.trainval_dataset import (
    HybridDataset, ReasonSegDataset,
    collate_fn_train, collate_fn_val
)
from dataloaders.utils import replace_image_tokens, tokenize_and_pad
from dataloaders.base_dataset import ImageProcessor
from model.llava.constants import DEFAULT_IMAGE_TOKEN

# Utils
from utils import (
    AverageMeter, ProgressMeter, Summary,
    prepare_input, intersectionAndUnionGPU,
)

# BLEU scoring
from pycocoevalcap.bleu.bleu import Bleu


def parse_args(args):
    """Define and parse training arguments."""
    parser = argparse.ArgumentParser(description="Train LISAT Model")

    # Model paths
    parser.add_argument("--version", default="./LISAt-7b")
    parser.add_argument("--vision-tower", default="./remote_clip_vit_l_14")
    # Precision settings
    parser.add_argument("--precision", choices=["fp32", "bf16", "fp16"], default="bf16")
    parser.add_argument("--load_in_8bit", action="store_true")
    parser.add_argument("--load_in_4bit", action="store_true")

    # Image and input settings
    parser.add_argument("--image_size", type=int, default=1024)
    parser.add_argument("--model_max_length", type=int, default=1024)

    # Dataset and training configuration
    parser.add_argument("--dataset", default="refer_seg||correct_refer_seg||vqa||neg_refer_seg")
    parser.add_argument("--sample_rates", default="9,3,3")
    parser.add_argument("--sem_seg_data", default="ade20k||cocostuff||pascal_part||paco_lvis")
    parser.add_argument("--refer_seg_data", default="refclef||refcoco||refcoco+||refcocog")
    parser.add_argument("--neg_refer_seg_data", default="R-refcocog||R-refcoco||R-refcoco+")
    parser.add_argument("--correct_refer_seg_data", default="fprefcocog||fprefcoco||fprefcoco+")
    parser.add_argument("--vqa_data", default="llava_instruct_150k")
    parser.add_argument("--reason_seg_data", default="ReasonSeg|train")
    parser.add_argument("--geo_reason_seg_data", default="GeoReasonSeg|train")

    # Directories
    parser.add_argument("--dataset_dir", default="./dataset")
    parser.add_argument("--log_base_dir", default="./runs")
    parser.add_argument("--exp_name", default="LISAT")

    # Training hyperparameters
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--steps_per_epoch", type=int, default=500)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--val_batch_size", type=int, default=1)
    parser.add_argument("--grad_accumulation_steps", type=int, default=1)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.0003)
    parser.add_argument("--ce_loss_weight", type=float, default=1.0)
    parser.add_argument("--dice_loss_weight", type=float, default=0.5)
    parser.add_argument("--bce_loss_weight", type=float, default=2.0)
    parser.add_argument("--lora_r", type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=16)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj")
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--num_classes_per_sample", type=int, default=1)

    # Training control
    parser.add_argument("--no_eval", action="store_true")
    parser.add_argument("--eval_only", action="store_true")
    parser.add_argument("--resume", default="")
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--print_freq", type=int, default=3)
    parser.add_argument("--gradient_checkpointing", action="store_true", default=True)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--conv_type", choices=["llava_v1", "llava_llama_2"], default="llava_v1")
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H")
    parser.add_argument("--out_dim", type=int, default=256)

    # Eval VQA captioning files
    parser.add_argument("--vqa_eval_file_nwpu", default="./dataset/vqa_caption/NWPU-Captions.jsonl")
    parser.add_argument("--vqa_eval_file_sydney", default="./dataset/vqa_caption/Sydney-Captions.jsonl")
    parser.add_argument("--vqa_eval_file_ucm", default="./dataset/vqa_caption/UCM-Captions.jsonl")

    return parser.parse_args(args)


def main(args):
    args = parse_args(args)
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)

    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        wandb.init(project="lisat", name=args.exp_name)

    # ---- Init conversation template ----
    conversation_lib.default_conversation = conversation_lib.conv_templates[args.conv_type]

    # ---- Init model ----
    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "vision_pretrained": args.vision_pretrained,
        "use_mm_start_end": args.use_mm_start_end,
    }
    tokenizer, model, vision_tower = init_LISAT_model(args, model_args)
    from IPython import embed; embed()
    # Setup DDP
    world_size = torch.cuda.device_count()
    args.distributed = world_size > 1

    # ---- Build training set ----
    train_dataset = HybridDataset(
        args.dataset_dir,
        vision_tower.image_processor,
        samples_per_epoch=args.batch_size
        * args.grad_accumulation_steps
        * args.steps_per_epoch
        * world_size,
        image_size=args.image_size,
        num_classes_per_sample=args.num_classes_per_sample,
        dataset=args.dataset,
        sample_rate=[float(x) for x in args.sample_rates.split(",")],
        sem_seg_data=args.sem_seg_data,
        refer_seg_data=args.refer_seg_data,
        neg_refer_seg_data=args.neg_refer_seg_data,
        vqa_data=args.vqa_data,
        reason_seg_data=args.reason_seg_data,
        geo_reason_seg_data=args.geo_reason_seg_data,
    )

    # ---- Build validation set ----
    if not args.no_eval:
        val_dataset = ReasonSegDataset(
            args.dataset_dir,
            vision_tower.image_processor,
            samples_per_epoch=200,
            image_size=args.image_size,
            num_classes_per_sample=3,
            reason_seg_data="GeoReasonSeg|val",
            use_fp=False,
        )
        print(f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples.")
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples (no_eval).")

    # ---- Deepspeed Config ----
    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": args.epochs * args.steps_per_epoch,
                "warmup_min_lr": 0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        },
        "fp16": {"enabled": args.precision == "fp16"},
        "bf16": {"enabled": args.precision == "bf16"},
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }

    model_engine, optimizer, train_loader, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        training_data=train_dataset,
        collate_fn=partial(
            collate_fn_train,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
        ),
        config=ds_config,
    )

    # ---- Resume if needed ----
    if args.auto_resume and len(args.resume) == 0:
        maybe_resume = os.path.join(args.log_dir, "ckpt_model")
        if os.path.exists(maybe_resume):
            args.resume = maybe_resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        print(f"Resume training from {args.resume}, start from epoch {args.start_epoch}")

    # ---- Validation DataLoader ----
    if val_dataset is not None:
        assert args.val_batch_size == 1
        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset, shuffle=True, drop_last=True
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.val_batch_size,
            shuffle=False,
            num_workers=args.workers,
            pin_memory=False,
            sampler=val_sampler,
            collate_fn=partial(
                collate_fn_val,
                tokenizer=tokenizer,
                use_mm_start_end=args.use_mm_start_end,
            ),
        )

    train_iter = iter(train_loader)

    # Keep track of the best combined metric
    best_score = 0.0

    # ---- Evaluate-Only Mode ----
    if args.eval_only:
        # Evaluate segmentation if desired
        if val_dataset is not None:
            giou, ciou = validate_seg(val_loader, model_engine, 0, args)
        else:
            giou, ciou = 0.0, 0.0

        # Evaluate all VQA/Caption sets
        nwpu_bleu4 = validate_vqa(
            args.vqa_eval_file_nwpu,
            model_engine,
            tokenizer,
            vision_tower,
            args.precision,
            args.image_size,
            args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            max_new_tokens=args.model_max_length,
        )
        sydney_bleu4 = validate_vqa(
            args.vqa_eval_file_sydney,
            model_engine,
            tokenizer,
            vision_tower,
            args.precision,
            args.image_size,
            args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            max_new_tokens=args.model_max_length,
        )
        ucm_bleu4 = validate_vqa(
            args.vqa_eval_file_ucm,
            model_engine,
            tokenizer,
            vision_tower,
            args.precision,
            args.image_size,
            args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            max_new_tokens=args.model_max_length,
        )

        # Compute new best_score formula
        combined_metric = (
            (nwpu_bleu4 / 65.8)
            + (sydney_bleu4 / 62.23)
            + (ucm_bleu4 / 72.34)
            + (giou / 0.275)
        )

        print(
            f"[Eval-Only] NWPU={nwpu_bleu4:.2f}, Sydney={sydney_bleu4:.2f}, "
            f"UCM={ucm_bleu4:.2f}, gIoU={giou:.4f}, cIoU={ciou:.4f}, best_score={combined_metric:.4f}"
        )
        return

    # ---- Training Loop ----
    for epoch in range(args.start_epoch, args.epochs):
        # Train
        train_iter, global_iters = train_one_epoch(
            train_loader, model_engine, epoch, scheduler, train_iter, args
        )

        if not args.no_eval:
            # Validate segmentation
            giou, ciou = validate_seg(val_loader, model_engine, global_iters, args)

            # Validate NWPU
            nwpu_bleu4 = validate_vqa(
                args.vqa_eval_file_nwpu,
                model_engine,
                tokenizer,
                vision_tower,
                args.precision,
                args.image_size,
                args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                max_new_tokens=args.model_max_length,
            )
            # Validate Sydney
            sydney_bleu4 = validate_vqa(
                args.vqa_eval_file_sydney,
                model_engine,
                tokenizer,
                vision_tower,
                args.precision,
                args.image_size,
                args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                max_new_tokens=args.model_max_length,
            )
            # Validate UCM
            ucm_bleu4 = validate_vqa(
                args.vqa_eval_file_ucm,
                model_engine,
                tokenizer,
                vision_tower,
                args.precision,
                args.image_size,
                args.conv_type,
                use_mm_start_end=args.use_mm_start_end,
                max_new_tokens=args.model_max_length,
            )

            # Log separate BLEU-4 and seg metrics
            if args.local_rank == 0:
                wandb.log(
                    {
                        "giou": giou,
                        "ciou": ciou,
                        "nwpu_bleu4": nwpu_bleu4,
                        "sydney_bleu4": sydney_bleu4,
                        "ucm_bleu4": ucm_bleu4,
                    },
                    step=global_iters,
                )

            # Compute the new combined metric (best_score formula)
            combined_metric = (
                # (nwpu_bleu4 / 65.8)
                # + (sydney_bleu4 / 62.23)
                # + (ucm_bleu4 / 72.34)
                # + (giou / 0.275)
                giou
            )

            # Check and save if best
            is_best = combined_metric > best_score
            best_score = max(best_score, combined_metric)

            if args.local_rank == 0:
                wandb.log({"best_score": combined_metric}, step=global_iters)

            if is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if os.path.exists(save_dir) and args.local_rank == 0:
                    shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir)


def train_one_epoch(train_loader, model, epoch, scheduler, train_iter, args):
    """Main training loop (one epoch)."""
    keys = ["loss", "ce_loss", "mask_bce_loss", "mask_dice_loss", "mask_loss"]
    loss_meters = {k: AverageMeter(k, ":.4f") for k in keys}

    progress = ProgressMeter(
        args.steps_per_epoch,
        list(loss_meters.values()),
        prefix=f"Epoch: [{epoch}]"
    )

    model.train()

    for global_step in range(args.steps_per_epoch):
        for _ in range(args.grad_accumulation_steps):
            try:
                input_dict = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                input_dict = next(train_iter)

            input_dict = prepare_input(input_dict, args.precision, is_cuda=True)
            output_dict = model(**input_dict)

            batch_size = input_dict["images"].size(0)
            for k in keys:
                loss_meters[k].update(output_dict[k].item(), batch_size)

            model.backward(output_dict["loss"])
            model.step()

        # Log + reset
        if global_step % args.print_freq == (args.print_freq - 1):
            if args.distributed:
                for k in keys:
                    loss_meters[k].all_reduce()

            total_steps = global_step + args.steps_per_epoch * epoch
            if args.local_rank == 0:
                progress.display(global_step + 1)
                for k in keys:
                    wandb.log({k: loss_meters[k].avg}, step=total_steps)
                curr_lr = scheduler.get_last_lr()[0]
                wandb.log({"lr": curr_lr}, step=total_steps)

            for k in keys:
                loss_meters[k].reset()

    return train_iter, (epoch + 1) * args.steps_per_epoch


@torch.inference_mode()
def validate_seg(val_loader, model_engine, global_iters, args):
    """Validate on a segmentation dataset (GeoReasonSeg)."""
    intersection_meter = AverageMeter("Intersec", ":6.3f", Summary.SUM)
    union_meter = AverageMeter("Union", ":6.3f", Summary.SUM)
    acc_iou_meter = AverageMeter("gIoU", ":6.3f", Summary.SUM)

    model_engine.eval()

    for input_dict in tqdm.tqdm(val_loader, desc="Val-Seg"):
        torch.cuda.empty_cache()
        input_dict = prepare_input(input_dict, args.precision, is_cuda=True)
        output_dict = model_engine(**input_dict)

        pred_masks = output_dict["pred_masks"]
        masks_list = output_dict["gt_masks"][0].int()
        output_list = (pred_masks[0] > 0).int()

        intersection, union, acc_iou = 0.0, 0.0, 0.0
        for mask_i, output_i in zip(masks_list, output_list):
            inter_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += inter_i
            union += union_i
            acc_iou += inter_i / (union_i + 1e-5)
            # If union_i == 0 => no-object target
            acc_iou[union_i == 0] += 1.0

        intersection, union = intersection.cpu().numpy(), union.cpu().numpy()
        acc_iou = acc_iou.cpu().numpy() / masks_list.shape[0]
        intersection_meter.update(intersection)
        union_meter.update(union)
        acc_iou_meter.update(acc_iou, n=masks_list.shape[0])

    intersection_meter.all_reduce()
    union_meter.all_reduce()
    acc_iou_meter.all_reduce()

    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    ciou = iou_class[1]
    giou = acc_iou_meter.avg[1]

    if args.local_rank == 0:
        print(f"[Val-Seg] gIoU={giou:.4f}, cIoU={ciou:.4f}")

    return giou, ciou


@torch.inference_mode()
def validate_vqa(
    vqa_file,
    model_engine,
    tokenizer,
    vision_tower,
    precision="bf16",
    image_size=1024,
    conv_type="llava_v1",
    use_mm_start_end=True,
    max_new_tokens=256,
):
    """Validate on a vqa dataset (NWPU, Sydney, UCM)."""
    device = next(model_engine.parameters()).device
    model_engine.eval()
    conversation_lib.default_conversation = conversation_lib.conv_templates[conv_type]
    tokenizer.padding_side = "left"
    img_processor = ImageProcessor(vision_tower.image_processor, image_size)

    predictions_dict = {}
    references_dict = {}

    with open(vqa_file, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in tqdm.tqdm(lines, desc=f"Val-VQA ({os.path.basename(vqa_file)})"):
        example = json.loads(line.strip())
        question_id = example["question_id"]
        question = example["text"]
        references = example["answer"]
        if not isinstance(references, list):
            references = [references]
        references_dict[question_id] = references

        base_dir = os.path.dirname(vqa_file) 
        image_path = os.path.join(base_dir, example["image"])
        raw_image = Image.open(image_path).convert("RGB")
        image, image_clip, sam_mask_shape = img_processor.load_and_preprocess_image(
            image_path
        )

        conv = conversation_lib.default_conversation.copy()
        prompt = DEFAULT_IMAGE_TOKEN + "\n" + question
        conv.append_message(conv.roles[0], prompt)
        conv.append_message(conv.roles[1], None)  
        conversation_list = [conv.get_prompt()]

        if use_mm_start_end:
            conversation_list = replace_image_tokens(conversation_list)

        input_ids, _ = tokenize_and_pad(conversation_list, tokenizer, padding="left")

        input_dict = {
            "images_clip": torch.stack([image_clip], dim=0),
            "images": torch.stack([image], dim=0),
            "input_ids": input_ids,
            "sam_mask_shape_list": [sam_mask_shape],
        }
        input_dict = prepare_input(input_dict, precision, is_cuda=True)

        output_ids, pred_masks, object_presence = model_engine.module.evaluate(
            input_dict["images_clip"],
            input_dict["images"],
            input_dict["input_ids"],
            input_dict["sam_mask_shape_list"],
            max_new_tokens=max_new_tokens,
        )
        real_output_ids = output_ids[:, input_ids.shape[1] :]
        pred_text = tokenizer.batch_decode(real_output_ids, skip_special_tokens=True)[0]

        predictions_dict[question_id] = [pred_text]

    # Compute BLEU with pycocoevalcap
    bleu_scorer = Bleu(n=4)
    bleu_score, _ = bleu_scorer.compute_score(references_dict, predictions_dict)
    bleu4 = bleu_score[3] * 100.0

    if torch.distributed.get_rank() == 0:
        print(
            f"[Val-VQA: {os.path.basename(vqa_file)}] "
            f"BLEU1={bleu_score[0]*100:.2f}, BLEU2={bleu_score[1]*100:.2f}, "
            f"BLEU3={bleu_score[2]*100:.2f}, BLEU4={bleu4:.2f}"
        )

    return bleu4

if __name__ == "__main__":
    main(sys.argv[1:])

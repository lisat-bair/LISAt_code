import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict

import pandas as pd

from ..coco import COCO
from .class_dist import CocoClassDistHelper
from .utils import load_json, save_json


# Build some category count info:
def get_cat_counts_dict(classes, coco):
    cat_counts = {}
    for cat, img_count, ann_count in list(
        zip(classes, coco.get_class_img_counts().items(), coco.get_class_ann_counts().items())
    ):
        assert img_count[0] == ann_count[0]
        cat_counts[img_count[0]] = {
            "img_count": int(img_count[1]),
            "ann_count": int(ann_count[1]),
            "name": cat,
            "id": img_count[0],
        }
    return cat_counts


def show_class_img_and_ann_counts(coco: COCO) -> None:
    cats = [c for c in sorted(coco.dataset["categories"], key=lambda cat: cat["id"])]
    classes = [c["name"] for c in cats]
    cat_counts_dict = get_cat_counts_dict(classes, coco)
    for cat in sorted(cat_counts_dict.items(), key=lambda c: c[1]["id"]):
        print(cat)
    print("Total cats: ", len(classes))


def show_img_stats(coco: COCO) -> None:
    """
    Counts annotations images have. Results are grouped into buckets of annotations counts: [-1-0)
    annotations, [0-1) annotations, [1-10), etc.
    """
    imgs = coco.dataset["images"]
    anns = coco.dataset["annotations"]
    img_counts = Counter({img["id"]: 0 for img in imgs})
    img_counts.update([ann["image_id"] for ann in anns])
    img_widths = [img["width"] for img in sorted(imgs, key=lambda i: int(i["id"]))]
    img_heights = [img["height"] for img in sorted(imgs, key=lambda i: int(i["id"]))]
    c = sorted(img_counts.items(), key=lambda c: c[1])
    img_ids, ann_counts = zip(*c)
    print(len(img_widths))
    counts = pd.DataFrame(
        {
            "img_id": img_ids,
            "ann_count": ann_counts,
            "ann_count_bin": pd.cut(
                ann_counts,
                [-1, 0, 1, 10, 20, 50, 100, 500, 1000, 10000, 100000, 1000000],
            ),
            "width": img_widths,
            "width_bin": pd.cut(
                img_widths,
                [-1, 0, 100, 512, 768, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000],
            ),
            "height": img_heights,
            "height_bin": pd.cut(
                img_heights,
                [-1, 0, 100, 512, 768, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 10000],
            ),
        }
    )
    print(counts)
    print(counts.groupby(["ann_count_bin"]).agg(total_imgs=("img_id", "count")))
    print(counts.agg(total_imgs=("img_id", "count"), total_anns=("ann_count", "sum")))
    print(counts.groupby(["width_bin"]).agg(total_imgs=("img_id", "count")))
    print(counts.groupby(["height_bin"]).agg(total_imgs=("img_id", "count")))
    print(counts.describe())


def check_stats(ann_path: Path):
    """
    Display various stats about the coco formatted dataset. Category counts, image counts,
    annotation counts, image dimensions, how many empty images there are, etc.
    """
    assert ann_path.exists() and ann_path.is_file()
    coco = CocoClassDistHelper(ann_path)
    show_class_img_and_ann_counts(coco)
    show_img_stats(coco)


def check_boxes(ann_path: Path):
    """Basic checks on validity of bounding boxes """
    data = load_json(ann_path)
    anns = data["annotations"]
    invalids = []
    for ann in anns:
        x, y, w, h = ann["bbox"]
        if w < 0 or w > 512:
            invalids.append(("invalid width", ann))
        elif h < 0 or h > 512:
            invalids.append(("invalid height", ann))
        elif x < 0 or x > 512:
            invalids.append(("invalid x", ann))
        elif y < 0 or y > 512:
            invalids.append(("invalid y", ann))
    # invalids.append(("invalid y1", ann))
    # invalids.append(("invalid y1", ann))
    # invalids.append(("invalid y2", ann))
    # invalids.append(("invalid y3", ann))
    counts = Counter([i[0] for i in invalids])
    for k in counts.keys():
        for i in invalids:
            if i[0] == k:
                print(i)
                break
    print(f"Found {len(invalids)} invalid boxes")
    print("Counts: ", counts)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--action", type=str, default="stats")
    parser.add_argument(
        "--ann_path",
        type=Path,
        default=Path(
            "/h4d_root/data/fsdet/coco/vanilla/comsat_full_val.json",
            help=(
                "/h4d_root/data/fsdet/coco/vanilla/comsat_full_train.json",
                "/h4d_root/data/fsdet/coco/vanilla/comsat_full_val.json",
            ),
        ),
    )
    args = parser.parse_args()
    if args.action == "stats":
        check_stats(args.ann_path)
    elif args.action == "box_check":
        check_boxes(args.ann_path)

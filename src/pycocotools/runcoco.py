import unittest
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def test_coco_json_loading() -> None:
    ann_path = (
        Path("") / "tests/data/annotations/xview_coco_v2_tiny_val_chipped.json"
    ).resolve()
    coco = COCO(ann_path)

    assert len(coco.imgs) == 64
    assert len(coco.cats) == 35


if __name__ == "__main__":
    test_coco_json_loading()
"""
trainvalsplit.py is a script that splits an MS COCO formatted dataset into train and val partitions.
For sample usage, run from command line:

Example:
    python trainvalsplit.py --help
"""
import random
from pathlib import Path
from typing import Any, List, Tuple

import numpy as np

from .class_dist import CocoClassDistHelper
from .coco_builder import CocoJsonBuilder

# Used to check the results of the split--all classes in both splits
# should have at least this many annotations:
_CLASS_COUNT_THRESHOLD = 0

# Seed value 341589 was chosen via the train-val-split-xviewcoco notebook:
_RANDOM_SEED = 486

# Size of val split. The train split size will be 1 - _TEST_SIZE.
_TEST_SIZE = 0.065


def split(data: List, test_size: float = 0.2, random_state=None) -> Tuple[List[Any], List[Any]]:
    """
    Similar to scikit learn, creates train/test splits of the passed in data.

    Args:
        data: A list or iterable type, of data to split.

        test_size: value in [0, 1.0] indicating the size of the test split. random_state: an int or
        RandomState object to seed the numpy randomness.

    Returns: 2-tuple of lists; (train, test), where each item in data has been placed
        into either the train or test split.
    """
    n = len(data)
    num_test = int(np.ceil(test_size * n))
    # print(F"n:{n}, num_test:{num_test}")
    np.random.seed(random_state)
    test_idx = set(np.random.choice(range(n), num_test))
    data_test, data_train = list(), list()
    for idx, datum in enumerate(data):
        if idx in test_idx:
            data_test.append(data[idx])
        else:
            data_train.append(data[idx])
    return data_train, data_test


def create_split(input_json, output_path, output_json_name):
    """
    Creates train/val split for the coco-formatted dataset defined by input_json.

    Args:

        input_json: full path or Path object to coco-formatted input json file. output_path: full
        path or Path object to directory where outputted json will be saved. output_json_name:
    """
    coco = CocoClassDistHelper(input_json)
    train_img_ids, val_img_ids = split(
        coco.img_ids, test_size=_TEST_SIZE, random_state=_RANDOM_SEED
    )
    train_counts, train_percents = coco.get_class_dist(train_img_ids)
    val_counts, val_percents = coco.get_class_dist(val_img_ids)

    # Generate coco-formatted json's for train and val:
    def generate_coco_json(coco, split_type, img_ids):
        coco_builder = CocoJsonBuilder(
            coco.cats, dest_path=output_path, dest_name=output_json_name.format(split_type)
        )
        for idx, img_id in enumerate(img_ids):
            coco_builder.add_image(coco.imgs[img_id], coco.imgToAnns[img_id])
        coco_builder.save()

    generate_coco_json(coco, "train", train_img_ids)
    generate_coco_json(coco, "val", val_img_ids)
    return coco


def verify_output(original_coco, output_path, output_json_name):
    """
    Verify that the outputted json's for the train/val split can be loaded, and have correct number
    of annotations, and minimum count for each class meets our threshold.
    """

    def verify_split_part(output_json_name, split_part):
        json_path = output_path / output_json_name.format(split_part)
        print(f"Checking if we can load json via coco api:{json_path}...")
        coco = CocoClassDistHelper(json_path)
        counts, _ = coco.get_class_dist()
        assert min(counts.values()) >= _CLASS_COUNT_THRESHOLD, (
            f"min class count ({min(counts.values())}) is "
            + f"lower than threshold of {_CLASS_COUNT_THRESHOLD}"
        )
        print(f"{split_part} class counts: ", counts)
        return coco

    train_coco = verify_split_part(output_json_name, "train")
    val_coco = verify_split_part(output_json_name, "val")
    assert len(original_coco.imgs) == len(train_coco.imgs) + len(
        val_coco.imgs
    ), "Num Images in original data should equal sum of imgs in splits."
    assert len(original_coco.anns) == len(train_coco.anns) + len(
        val_coco.anns
    ), "Num annotations in original data should equal sum of those in splits."


def _main(opt):
    """
    Creates train/val split and verifies output.

    Args:
        opt: command line options (there are none right now)

        output_json_name: format-string of output file names, with a '{}' style placeholder where
        split type will be inserted.
    """
    print(h4dconfig.DATA_DIR)
    datadir: Path = Path("/home/laielli/data")
    output_json_name = "xview_coco_complete_v1_{}.json"
    input_json = datadir / "Xview/coco_complete/{}.json".format("xview_coco_complete_v0")
    output_path = datadir / "Xview/coco_complete"
    original_coco = create_split(input_json, output_path, output_json_name)
    verify_output(original_coco, output_path, output_json_name)


if __name__ == "__main__":
    opt = None
    # parser = argparse.ArgumentParser()
    # opt = parser.parse_args()
    _main(opt)

from copy import deepcopy
from pathlib import Path
from typing import Dict, Union

from ..coco import COCO
from .utils import load_json, save_json

__all__ = ["reindex_coco_json"]


def reindex_coco_json(input_file: Union[str, Path]):
    """
    If the coco categories in the input_file use cat_id=0, AND cat_id=0 is used for something other
    than "background", then this function will reindex the categories (and adjust the annotations'
    category_id references) so they start at cat_id=0 as being used for background_id, and the id's
    for all the other categories are shifted by +1. The results overwrite the existing input_file.
    """
    if isinstance(input_file, str):
        input_file = Path(input_file)
    coco = COCO(input_file)
    is_zero_background_catid = coco_has_zero_as_background_id(coco)
    if not is_zero_background_catid:
        new_cats, new_anns = adjust_cat_ids(coco)
        root_json = {}
        root_json["categories"] = new_cats
        if "info" in coco.dataset:
            root_json["info"] = deepcopy(coco.dataset["info"])
        if "licenses" in coco.dataset:
            root_json["licenses"] = deepcopy(coco.dataset["licenses"])
        root_json["images"] = coco.dataset["images"]
        root_json["annotations"] = new_anns
        save_json(input_file, root_json)
    else:
        print("cat_id 0 is either already background, or unused. Nothing to do.")


def adjust_cat_ids(coco):
    cats = coco.dataset["categories"]
    new_cats = []
    for cat in cats:
        new_cat = {
            "supercategory": cat["supercategory"] if "supercategory" in cat else "",
            "id": int(cat["id"]) + 1,
            "name": cat["name"],
        }
        new_cats.append(new_cat)
    # Adjust annotations:
    anns = coco.dataset["annotations"]
    new_anns = deepcopy(anns)
    for ann in new_anns:
        ann["category_id"] = ann["category_id"] + 1

    return new_cats, new_anns


def coco_has_zero_as_background_id(coco: COCO):
    """
    Return true if category_id=0 is either unused, or used for background class. Else return false.
    """
    cats = coco.dataset["categories"]
    cat_id_zero_nonbackground_exists = False
    for cat in cats:
        if cat["id"] == 0:
            if cat["name"] not in ["background", "__background__"]:
                cat_id_zero_nonbackground_exists = True
                break

    return not cat_id_zero_nonbackground_exists

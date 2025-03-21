from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, OrderedDict, Tuple, Union

import numpy as np
import pandas as pd

from ..coco import COCO

__all__ = ["CocoClassDistHelper", "get_ref_stats"]


class CocoClassDistHelper(COCO):
    """
    A subclass of pycocotools.coco that adds a method(s) to calculate class distribution.
    """

    def __init__(self, annotation_file: Optional[Union[str, Path]] = None, **kwargs):
        super().__init__(annotation_file, **kwargs)

        self.cats_list = self.loadCats(self.getCatIds())
        """list of dictionaries. 3 keys each: (supercategory, id, name)"""
        list.sort(self.cats_list, key=lambda c: c["id"])

        self.cat_name_lookup = {c["id"]: c["name"] for c in self.cats_list}
        """Dictionaries to lookup category and supercategory names from category id"""

        self.img_ids = self.getImgIds()
        """List of integers, image id's"""

        self.ann_ids = self.getAnnIds(imgIds=self.img_ids)
        """List of strings, each is an annotation id"""

        self.anns_list = self.loadAnns(self.ann_ids)
        print(f"num images: {len(self.img_ids)}")
        # print(F"num annotation id's: {len(self.ann_ids)}")
        print(f"num annotations: {len(self.anns)}")

        #  Create self.img_ann_counts, a dictionary keyed off of img_id. For each img_id it stores a
        #  collections.Counter object that has a count of how many annotations for each
        #  category/class there are for that img_id
        self.img_ann_counts = {}
        for img_id in self.imgToAnns.keys():
            imgAnnCounter = Counter({cat["name"]: 0 for cat in self.cats_list})
            anns = self.imgToAnns[img_id]
            for ann in anns:
                imgAnnCounter[self.cat_name_lookup[ann["category_id"]]] += 1
            self.img_ann_counts[img_id] = imgAnnCounter
        self.num_cats = len(self.cats_list)

        self.cat_img_counts: Dict[int, float] = {
            c["id"]: float(len(np.unique(self.catToImgs[c["id"]]))) for c in self.cats_list
        }

        # Annotation Counts
        self.cat_ann_counts: Dict[int, int] = defaultdict(int)
        for cat_id in self.cat_name_lookup.keys():
            self.cat_ann_counts[cat_id] = 0
        for ann in self.anns.values():
            self.cat_ann_counts[ann["category_id"]] += 1

        # Img + Ann counts
        self.cat_counts: Dict[int, Dict[str, Any]] = {
            c["id"]: {
                **c,
                "img_count": float(len(np.unique(self.catToImgs[c["id"]]))),
                "ann_count": float(self.cat_ann_counts[c["id"]]),
            }
            for c in self.cats_list
        }

        self.cat_img_counts = OrderedDict(sorted(self.cat_img_counts.items()))
        self.cat_ann_counts = OrderedDict(sorted(self.cat_ann_counts.items()))

    def get_class_dist(self, img_ids: List[int] = None):
        """
        Args:
            img_ids: List of image id's. If None, distribution is calculated for
                all image id's in the dataset.

        Returns: A dictionary representing the class distribution. Keys are category
            names Values are counts (e.g., how many annotations are there with that category/class
            label) np.array of class percentages. Entries are sorted by category_id (same as
            self.cats_list)
        """
        cat_counter = Counter({cat["name"]: 0 for cat in self.cats_list})
        if img_ids is None:
            img_ids = self.imgToAnns.keys()

        for img_id in img_ids:
            if img_id not in self.imgToAnns:
                continue
            cat_counter += self.img_ann_counts[img_id]

        # Convert to np array where entries correspond to cat_id's sorted asc.:
        total = float(sum(cat_counter.values()))
        cat_names = [c["name"] for c in self.cats_list]
        cat_percents = np.zeros((self.num_cats))
        for idx, cat_name in enumerate(sorted(cat_names)):
            cat_percents[idx] = cat_counter[cat_name] / total

        return cat_counter, cat_percents

    def get_cat_counts(self) -> Dict[int, Dict[str, Any]]:
        """
        Returns dictionary whose keys are class_id's and values are dictionaries with category id,
        name, and img & annotation counts
        """
        return self.cat_counts

    def get_class_img_counts(self):
        """
        Returns dictionary whose keys are class_id's and values are number of images with one or
        more instances of that class
        """
        return self.cat_img_counts

    def get_class_ann_counts(self):
        """
        Returns dictionary whose keys are class_id's and values are number of annotations available
        for that class
        """
        return self.cat_ann_counts


def get_ref_stats(coco: COCO, L: int = 3) -> tuple[pd.DataFrame, pd.DataFrame]:
    assert coco.is_ref_dataset, "COCO object must be a referring expression dataset."
    sentence_counts = Counter()
    counts = []

    for idx, ref in enumerate(coco.refs_data):
        sentences: list[dict] = ref["sentences"]
        count = len(sentences)
        sentence_counts.update({count: 1})
        counts.append(
            {
                "ref_id": ref["ref_id"],
                "ann_id": ref["ann_id"],
                "category_id": ref["category_id"],
                "category": coco.cats[ref["category_id"]]["name"],
                "supercategory": coco.cats[ref["category_id"]]["supercategory"]
                if "supercategory" in coco.cats[ref["category_id"]]
                else str(coco.cats[ref["category_id"]]),
                "sent_count": len(sentences),
                "pos_sent_count": len(
                    [s for s in sentences if ("exist" in s and s["exist"]) or "exist" not in s]
                ),
                "neg_sent_count": len([s for s in sentences if ("exist" in s and not s["exist"])]),
            }
        )

    # print(sentence_counts, len(sentence_counts))
    df = pd.DataFrame(counts)
    # display(df)
    print(
        "pos/neg sentence_counts: ",
        df.pos_sent_count.sum(),
        df.neg_sent_count.sum(),
    )
    df_agg = pd.DataFrame(
        df.groupby(lambda x: True).agg(
            num_refs=("ref_id", "count"),
            sent_count=("sent_count", "sum"),
            total_pos_sents=("pos_sent_count", "sum"),
            total_neg_sents=("neg_sent_count", "sum"),
        )
    )
    # display(df_agg)
    if L >= 1:
        df_agg = pd.DataFrame(
            df.groupby(["pos_sent_count"]).agg(
                num_refs=("ref_id", "count"),
                sent_count=("sent_count", "sum"),
                total_pos_sents=("pos_sent_count", "sum"),
                total_neg_sents=("neg_sent_count", "sum"),
            )
        )
        # display(df_agg)
    if L >= 2:
        df_agg = pd.DataFrame(
            df.groupby(["pos_sent_count", "neg_sent_count"]).agg(
                num_refs=("ref_id", "count"),
                sent_count=("sent_count", "sum"),
                total_pos_sents=("pos_sent_count", "sum"),
                total_neg_sents=("neg_sent_count", "sum"),
            )
        )
        # display(df_agg)

    df_agg["dataset"] = f"{coco.dataset_name}({coco.split_by})"
    df_agg["ann_count"] = len(coco.anns)
    df_agg["img_count"] = len(coco.imgs)
    # Make 'dataset' the first column:
    df_agg.insert(0, "dataset", df_agg.pop("dataset"))
    # display(df_agg.droplevel(axis=0, level=0).reset_index(drop=True))
    return df, df_agg.reset_index()

__author__ = "tylin"
__version__ = "2.0"
# Interface for accessing the Microsoft COCO dataset.

# Microsoft COCO is a large image dataset designed for object detection,
# segmentation, and caption generation. pycocotools is a Python API that
# assists in loading, parsing and visualizing the annotations in COCO.
# Please visit http://mscoco.org/ for more information on COCO, including
# for the data, paper, and tutorials. The exact format of the annotations
# is also described on the COCO website. For example usage of the pycocotools
# please see pycocotools_demo.ipynb. In addition to this API, please download both
# the COCO images and annotations in order to run the demo.

# An alternative to using the API is to load the annotations directly
# into Python dictionary
# Using the API provides additional utility functions. Note that this API
# supports both *instance* and *caption* annotations. In the case of
# captions not all functions are defined (e.g. categories are undefined).

# The following API functions are defined:
#  COCO       - COCO api class that loads COCO annotation file and prepare data structures.
#  decodeMask - Decode binary mask M encoded via run-length encoding.
#  encodeMask - Encode binary mask M using run-length encoding.
#  getAnnIds  - Get ann ids that satisfy given filter conditions.
#  getCatIds  - Get cat ids that satisfy given filter conditions.
#  getImgIds  - Get img ids that satisfy given filter conditions.
#  loadAnns   - Load anns with the specified ids.
#  loadCats   - Load cats with the specified ids.
#  loadImgs   - Load imgs with the specified ids.
#  annToMask  - Convert segmentation in an annotation to binary mask.
#  showAnns   - Display the specified annotations.
#  loadRes    - Load algorithm results and create API for accessing them.
#  download   - Download COCO images from mscoco.org server.
# Throughout the API "ann"=annotation, "cat"=category, and "img"=image.
# Help on each functions can be accessed by: "help COCO>function".

# See also COCO>decodeMask,
# COCO>encodeMask, COCO>getAnnIds, COCO>getCatIds,
# COCO>getImgIds, COCO>loadAnns, COCO>loadCats,
# COCO>loadImgs, COCO>annToMask, COCO>showAnns

# Microsoft COCO Toolbox.      version 2.0
# Data, paper, and tutorials available at:  http://mscoco.org/
# Code written by Piotr Dollar and Tsung-Yi Lin, 2014.
# Licensed under the Simplified BSD License [see bsd.txt]

import copy
import itertools

try:
    import simdjson as json
except:  # noqa: E722
    import json
import os
import pickle
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.request import urlretrieve

import matplotlib.pyplot as plt
import numpy as np
import skimage.io as io
from matplotlib.collections import PatchCollection
from matplotlib.patches import Polygon, Rectangle

from . import mask as maskUtils

__all__ = ["COCO", "Ann", "Cat", "Image", "Ref"]

# Typing aliases
Ann = dict[str, Any]
Cat = dict[str, Any]
Image = dict[str, Any]
Ref = dict[str, Any]


def _isArrayLike(obj):
    return hasattr(obj, "__iter__") and hasattr(obj, "__len__")


class COCO:
    def __init__(
        self,
        annotation_file: Optional[Union[str, Path]] = None,
        is_ref_dataset: bool = False,
        split_by: str = "unc",
        dataset_name: str = "",
    ):
        """
        Constructor of Microsoft COCO helper class for reading and visualizing annotations.

        :param annotation_file (str): location of annotation file. If this is a referring
            segmentation (ref_seg) dataset, this should be the path containing subdirectories, where
            the subdirectories are named after each ref_seg dataset.
        :param image_folder (str): location to the folder that hosts images.
        :param is_ref_dataset: Default is False. Set to True if this is a referring segmentation
            (ref_seg) dataset, such as refcoco, refcoco+, refcocog, or one of the Robust refcoco
            (R-refcoco) variants.
        :param split_by: Only used if is_ref_dataset=True. Indicates which refs `.p` file to load,
            e.g., 'refs(split_by).p`
        :param dataset_name: Only used if is_ref_dataset=True. The name of the subdirectory within
            the `annotation_file` directory. Also used to load the correct images (an example of why
            this is needed is the image folder structure is different for refclef than the others)

        Example:

        >>> coco = COCO(
        >>>    refseg_path / dataset_name / "instances.json",
        >>>    is_ref_dataset=True,
        >>>    dataset_name="refcoco+",
        >>>    split_by="unc",
        >>> )
        """
        # load dataset
        self.dataset, self.anns, self.cats, self.imgs = dict(), dict(), dict(), dict()
        self.imgToAnns, self.catToImgs = defaultdict(list), defaultdict(list)
        self.is_ref_dataset: bool = is_ref_dataset and len(split_by) > 0
        self.split_by: str = split_by
        self.dataset_name: str = dataset_name

        if annotation_file is None:
            return
        else:
            _annotation_file = Path(annotation_file).resolve()

        if self.is_ref_dataset:
            assert (
                self.dataset_name
            ), "You must specify a dataset_name if is_ref_datset==True"
            # You can either pass in the path to the intsances.json, or the folder that
            # contains refcoco/refcocog/refcoco+/refclef folders.
            if _annotation_file.is_dir():
                self.DATA_ROOT = _annotation_file
                self.DATA_DIR = self.DATA_ROOT / self.dataset_name
                _annotation_file = self.DATA_DIR / "instances.json"
            else:
                assert _annotation_file.exists() and _annotation_file.is_file(), str(
                    _annotation_file
                )
                self.DATA_ROOT = _annotation_file.parent.parent
                self.DATA_DIR = _annotation_file.parent
                assert str(self.DATA_DIR).endswith(self.dataset_name), (
                    str(self.DATA_DIR) + " " + self.dataset_name
                )
            if "coco" in self.dataset_name:
                self.IMAGE_DIR = self.DATA_ROOT / "images/mscoco/images/train2014"
            elif "clef" in self.dataset_name:
                self.IMAGE_DIR = self.DATA_ROOT / "images/saiapr_tc-12"
            else:
                raise NotImplementedError(f"Unsupported dataset: {self.dataset_name}")

            # Load refs:
            self.load_refs()

        if annotation_file is not None:
            print("loading annotations into memory...")
            _annotation_file = _annotation_file.resolve()
            tic = time.time()
            with open(_annotation_file, "r") as f:
                dataset = json.load(f)
            assert isinstance(
                dataset, dict
            ), "annotation file format {} not supported".format(type(dataset))
            print("Done (t={:0.2f}s)".format(time.time() - tic))
            self.dataset = dataset
            self.createIndex()

    def createIndex(self):
        # create index
        print("creating index...")
        anns, cats, imgs = {}, {}, {}
        name_to_cat: dict[str, Cat] = {}
        imgToAnns: dict[int, list[Ann]] = defaultdict(list)
        catToImgs: dict[int, list[Image]] = defaultdict(list)

        if "annotations" in self.dataset:
            for ann in self.dataset["annotations"]:
                imgToAnns[ann["image_id"]].append(ann)
                anns[ann["id"]] = ann

        if "images" in self.dataset:
            for img in self.dataset["images"]:
                imgs[img["id"]] = img

        if "categories" in self.dataset:
            for cat in self.dataset["categories"]:
                cats[cat["id"]] = cat
                name_to_cat[cat["name"]] = cat

        if "annotations" in self.dataset and "categories" in self.dataset:
            for ann in self.dataset["annotations"]:
                catToImgs[ann["category_id"]].append(ann["image_id"])

        # create class members
        self.anns: dict[int, Ann] = anns
        self.imgToAnns = imgToAnns
        self.catToImgs = catToImgs
        self.imgs: dict[int, Image] = imgs
        self.cats: dict[int, Cat] = cats
        self.name_to_cat = name_to_cat
        if self.is_ref_dataset:
            self.create_index_refs()
        print("index created!")

    def create_index_refs(self) -> None:
        """
        Similar to create_index(), but handles just the refs data (i.e., refering
        expressions data).
        """
        self.refs: dict[int, Ref] = {}
        self.img_to_refs: dict[int, list[Ref]] = defaultdict(list)
        self.cat_to_refs: dict[int, list[Ref]] = defaultdict(list)
        self.ann_to_ref: dict[int, Ref] = {}
        self.ref_to_ann: dict[int, Ann] = {}
        # Sentences:
        self.sents: dict[int, dict[str, Any]] = {}
        self.sent_to_ref: dict[int, Ref] = {}
        self.sent_to_tokens: dict[int, list[str]] = {}

        for ref in self.refs_data:
            self.refs[ref["ref_id"]] = ref
            self.img_to_refs[ref["image_id"]].append(ref)
            self.cat_to_refs[ref["category_id"]].append(ref)
            self.ann_to_ref[ref["ann_id"]] = ref
            self.ref_to_ann[ref["ref_id"]] = self.anns[ref["ann_id"]]
            # Sentences
            for s in ref["sentences"]:
                self.sents[s["sent_id"]] = s
                self.sent_to_ref[s["sent_id"]] = ref
                self.sent_to_tokens[s["sent_id"]] = s["tokens"]

    def load_refs(self) -> None:
        """
        refs file is a pickled list of dicts. Example dict::
            {'sent_ids': [0, 1, 2], 'file_name': 'COCO_train2014_000000581857_16.jpg',
            'ann_id': 1719310, 'ref_id': 0, 'image_id': 581857, 'split': 'train',
            'sentences': [{'tokens': ['the', 'lady', 'with', 'the', 'blue', 'shirt'],
            'raw': 'THE LADY WITH THE BLUE SHIRT', 'sent_id': 0, 'sent': 'the lady with
            the blue shirt'}, {'tokens': ['lady', 'with', 'back', 'to', 'us'], 'raw':
            'lady w back to us', 'sent_id': 1, 'sent': 'lady with back to us'},
            {'tokens': ['blue', 'shirt'], 'raw': 'blue shirt', 'sent_id': 2, 'sent':
            'blue shirt'}], 'category_id': 1}
        """
        ref_file = self.DATA_DIR / f"refs({self.split_by}).p"
        assert ref_file.exists(), str(ref_file)
        print(f"Loading refs from '{ref_file}'")
        self.refs_data: list[Ref] = pickle.load(open(ref_file, "rb"))
        print(f"Loaded {len(self.refs_data)} refs")

    def info(self):
        """
        Print information about the annotation file.
        :return:
        """
        for key, value in self.dataset["info"].items():
            print("{}: {}".format(key, value))

    def getRefIds(self, image_ids=[], cat_ids=[], ref_ids=[], split="") -> list[int]:
        assert (
            self.is_ref_dataset
        ), "Can only use getRefIds if self.is_ref_dataset==True"
        image_ids = image_ids if type(image_ids) == list else [image_ids]
        cat_ids = cat_ids if type(cat_ids) == list else [cat_ids]
        ref_ids = ref_ids if type(ref_ids) == list else [ref_ids]

        if len(image_ids) == len(cat_ids) == len(ref_ids) == len(split) == 0:
            refs = self.refs_data
        else:
            if not len(image_ids) == 0:
                refs = [
                    img for image_id in image_ids for img in self.img_to_refs[image_id]
                ]
            elif len(cat_ids) > 0:
                refs = [img for cat_id in cat_ids for img in self.cat_to_refs[cat_id]]
            else:
                refs = self.refs_data
            if not len(cat_ids) == 0:
                refs = [ref for ref in refs if ref["category_id"] in cat_ids]
            if not len(ref_ids) == 0:
                refs = [ref for ref in refs if ref["ref_id"] in ref_ids]
            if not len(split) == 0:
                if split in ["testA", "testB", "testC"]:
                    refs = [
                        ref for ref in refs if split[-1] in ref["split"]
                    ]  # we also consider testAB, testBC, ...
                elif split in ["testAB", "testBC", "testAC"]:
                    refs = [
                        ref for ref in refs if ref["split"] == split
                    ]  # rarely used I guess...
                elif split == "test":
                    refs = [ref for ref in refs if "test" in ref["split"]]
                elif split == "train" or split == "val":
                    refs = [ref for ref in refs if ref["split"] == split]
                else:
                    print("No such split [%s]" % split)
                    sys.exit()
        ref_ids = [ref["ref_id"] for ref in refs]
        return ref_ids

    def getAnnIds(
        self, imgIds=[], catIds=[], areaRng=[], iscrowd=None, refIds=[]
    ) -> list[int]:
        """
        Get ann ids that satisfy given filter conditions. default skips that filter
        :param imgIds  (int array)     : get anns for given imgs
               catIds  (int array)     : get anns for given cats
               areaRng (float array)   : get anns for given area range (e.g. [0 inf])
               iscrowd (boolean)       : get anns for given crowd label (False or True)
        :return: ids (int array)       : integer array of ann ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        ref_ids = refIds if _isArrayLike(refIds) else [refIds]
        if ref_ids:
            assert (
                self.is_ref_dataset
            ), "Can only filter by refIds if self.is_ref_dataset==True"

        if len(imgIds) == len(catIds) == len(areaRng) == len(ref_ids) == 0:
            anns = self.dataset["annotations"]
        else:
            if not len(imgIds) == 0:
                lists = [
                    self.imgToAnns[imgId] for imgId in imgIds if imgId in self.imgToAnns
                ]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset["annotations"]
            anns = (
                anns
                if len(catIds) == 0
                else [ann for ann in anns if ann["category_id"] in catIds]
            )
            anns = (
                anns
                if len(areaRng) == 0
                else [
                    ann
                    for ann in anns
                    if ann["area"] > areaRng[0] and ann["area"] < areaRng[1]
                ]
            )

            if len(ref_ids) > 0:
                filtered_anns = {self.ref_to_ann[ref_id]["id"] for ref_id in ref_ids}
                anns = [ann for ann in anns if ann["id"] in filtered_anns]
        if iscrowd is not None:
            ids = [ann["id"] for ann in anns if ann["iscrowd"] == iscrowd]
        else:
            ids = [ann["id"] for ann in anns]
        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]) -> list[int]:
        """
        filtering parameters. default skips that filter.
        :param catNms (str array)  : get cats for given cat names
        :param supNms (str array)  : get cats for given supercategory names
        :param catIds (int array)  : get cats for given cat ids
        :return: ids (int array)   : integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            cats = self.dataset["categories"]
        else:
            cats = self.dataset["categories"]
            cats = (
                cats
                if len(catNms) == 0
                else [cat for cat in cats if cat["name"] in catNms]
            )
            cats = (
                cats
                if len(supNms) == 0
                else [cat for cat in cats if cat["supercategory"] in supNms]
            )
            cats = (
                cats
                if len(catIds) == 0
                else [cat for cat in cats if cat["id"] in catIds]
            )
        ids = [cat["id"] for cat in cats]
        return ids

    def getImgIds(self, imgIds=[], catIds=[], refIds=[]) -> list[int]:
        """
        Get img ids that satisfy given filter conditions.
        :param imgIds (int array) : get imgs for given ids
        :param catIds (int array) : get imgs with all given cats
        :return: ids (int array)  : integer array of img ids
        """
        imgIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]
        refIds = refIds if _isArrayLike(refIds) else [refIds]
        if refIds:
            assert (
                self.is_ref_dataset
            ), "Can only filter by refIds if self.is_ref_dataset==True"

        if len(imgIds) == len(catIds) == len(refIds) == 0:
            ids = self.imgs.keys()
        else:
            ids = set(imgIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToImgs[catId])
                else:
                    ids &= set(self.catToImgs[catId])
            if len(refIds) > 0:
                ref_imgs = {self.refs[ref_id]["image_id"] for ref_id in refIds}
                ids = ids.intersection(ref_imgs) if ids else ref_imgs
        return list(ids)

    def loadAnns(self, ids=[]) -> list[Ann]:
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]
        else:
            raise NotImplementedError()

    def loadCats(self, ids=[]) -> list[Cat]:
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]
        return []

    def loadImgs(self, ids=[]) -> list[Image]:
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.imgs[id] for id in ids]
        elif type(ids) == int:
            return [self.imgs[ids]]
        raise NotImplementedError()

    def loadRefs(self, ref_ids=[]) -> list[Ref]:
        """Returns ref objects (dicts) for the given ref_ids."""
        assert self.is_ref_dataset, "loadRefs() is only valid if is_ref_dataset==True"
        if _isArrayLike(ref_ids):
            return [self.refs[ref_id] for ref_id in ref_ids]
        elif isinstance(ref_ids, int):
            return [self.refs[ref_ids]]
        raise NotImplementedError()

    def getRefBox(self, ref_id) -> list[int]:
        assert self.is_ref_dataset, "getRefBox() is only valid if is_ref_dataset==True"
        # ref = self.refs[ref_id]
        ann = self.ref_to_ann[ref_id]
        return ann["bbox"]  # [x, y, w, h]

    def showAnns(self, anns, draw_bbox=False):
        """
        Display the specified annotations.
        :param anns (array of object): annotations to display
        :return: None
        """
        if len(anns) == 0:
            return 0
        if "segmentation" in anns[0] or "keypoints" in anns[0]:
            datasetType = "instances"
        elif "caption" in anns[0]:
            datasetType = "captions"
        else:
            raise Exception("datasetType not supported")
        if datasetType == "instances":
            ax = plt.gca()
            ax.set_autoscale_on(False)
            polygons = []
            color = []
            for ann in anns:
                c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]
                if "segmentation" in ann:
                    if type(ann["segmentation"]) == list:
                        # polygon
                        for seg in ann["segmentation"]:
                            poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                            polygons.append(Polygon(poly))
                            color.append(c)
                    else:
                        # mask
                        t = self.imgs[ann["image_id"]]
                        if type(ann["segmentation"]["counts"]) == list:
                            rle = maskUtils.frPyObjects(
                                [ann["segmentation"]], t["height"], t["width"]
                            )
                        else:
                            rle = [ann["segmentation"]]
                        m = maskUtils.decode(rle)
                        img = np.ones((m.shape[0], m.shape[1], 3))
                        if ann["iscrowd"] == 1:
                            color_mask = np.array([2.0, 166.0, 101.0]) / 255
                        else:
                            # if ann["iscrowd"] == 0:
                            color_mask = np.random.random((1, 3)).tolist()[0]
                        for i in range(3):
                            img[:, :, i] = color_mask[i]
                        ax.imshow(np.dstack((img, m * 0.5)))
                if "keypoints" in ann and type(ann["keypoints"]) == list:
                    # turn skeleton into zero-based index
                    sks = np.array(self.loadCats(ann["category_id"])[0]["skeleton"]) - 1
                    kp = np.array(ann["keypoints"])
                    x = kp[0::3]
                    y = kp[1::3]
                    v = kp[2::3]
                    for sk in sks:
                        if np.all(v[sk] > 0):
                            plt.plot(x[sk], y[sk], linewidth=3, color=c)
                    plt.plot(
                        x[v > 0],
                        y[v > 0],
                        "o",
                        markersize=8,
                        markerfacecolor=c,
                        markeredgecolor="k",
                        markeredgewidth=2,
                    )
                    plt.plot(
                        x[v > 1],
                        y[v > 1],
                        "o",
                        markersize=8,
                        markerfacecolor=c,
                        markeredgecolor=c,
                        markeredgewidth=2,
                    )

                if draw_bbox:
                    [bbox_x, bbox_y, bbox_w, bbox_h] = ann["bbox"]
                    poly = [
                        [bbox_x, bbox_y],
                        [bbox_x, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y + bbox_h],
                        [bbox_x + bbox_w, bbox_y],
                    ]
                    np_poly = np.array(poly).reshape((4, 2))
                    polygons.append(Polygon(np_poly))
                    color.append(c)

            p = PatchCollection(polygons, facecolor=color, linewidths=0, alpha=0.4)
            ax.add_collection(p)
            p = PatchCollection(
                polygons, facecolor="none", edgecolors=color, linewidths=2
            )
            ax.add_collection(p)
        elif datasetType == "captions":
            for ann in anns:
                print(ann["caption"])

    def showRef(self, ref, seg_box="seg"):
        assert self.is_ref_dataset, "showRef is only valid if is_ref_dataset==True"
        ax = plt.gca()
        # show image
        image = self.imgs[ref["image_id"]]
        I = io.imread(self.IMAGE_DIR / image["file_name"])
        ax.imshow(I)
        # show refer expression
        for sid, sent in enumerate(ref["sentences"]):
            print(f"{sid+1}. {sent['sent']}")
        # show segmentations
        if seg_box == "seg":
            ann_id = ref["ann_id"]
            ann = self.anns[ann_id]
            polygons = []
            color = []
            c = "none"
            if type(ann["segmentation"][0]) == list:
                # polygon used for refcoco*
                for seg in ann["segmentation"]:
                    poly = np.array(seg).reshape((int(len(seg) / 2), 2))
                    polygons.append(Polygon(poly, alpha=0.4))
                    color.append(c)
                p = PatchCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=(1, 1, 0, 0),
                    linewidths=3,
                    alpha=1,
                )
                ax.add_collection(p)  # thick yellow polygon
                p = PatchCollection(
                    polygons,
                    facecolors=color,
                    edgecolors=(1, 0, 0, 0),
                    linewidths=1,
                    alpha=1,
                )
                ax.add_collection(p)  # thin red polygon
            else:
                # mask used for refclef
                rle = ann["segmentation"]
                m = maskUtils.decode(rle)
                img = np.ones((m.shape[0], m.shape[1], 3))
                color_mask = np.array([2.0, 166.0, 101.0]) / 255
                for i in range(3):
                    img[:, :, i] = color_mask[i]
                ax.imshow(np.dstack((img, m * 0.5)))
        # show bounding-box
        elif seg_box == "box":
            ann_id = ref["ann_id"]
            ann = self.anns[ann_id]
            bbox = self.getRefBox(ref["ref_id"])
            box_plot = Rectangle(
                (bbox[0], bbox[1]),
                bbox[2],
                bbox[3],
                fill=False,
                edgecolor="green",
                linewidth=3,
            )
            ax.add_patch(box_plot)

    def loadRes(self, resFile):
        """
        Load result file and return a result api object.
        :param   resFile (str)     : file name of result file
        :return: res (obj)         : result api object
        """
        res = COCO()
        res.dataset["images"] = [img for img in self.dataset["images"]]

        print("Loading and preparing results...")
        tic = time.time()
        if type(resFile) == str:
            anns = json.load(open(resFile))
        elif type(resFile) == np.ndarray:
            anns = self.loadNumpyAnnotations(resFile)
        else:
            anns = resFile
        assert type(anns) == list, "results in not an array of objects"
        annsImgIds = [ann["image_id"] for ann in anns]
        assert set(annsImgIds) == (
            set(annsImgIds) & set(self.getImgIds())
        ), "Results do not correspond to current coco set"
        if "caption" in anns[0]:
            imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
                [ann["image_id"] for ann in anns]
            )
            res.dataset["images"] = [
                img for img in res.dataset["images"] if img["id"] in imgIds
            ]
            for id, ann in enumerate(anns):
                ann["id"] = id + 1
        elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                bb = ann["bbox"]
                x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
                if "segmentation" not in ann:
                    ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
                ann["area"] = bb[2] * bb[3]
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "segmentation" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                # now only support compressed RLE format as segmentation results
                ann["area"] = maskUtils.area(ann["segmentation"])
                if "bbox" not in ann:
                    ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
                ann["id"] = id + 1
                ann["iscrowd"] = 0
        elif "keypoints" in anns[0]:
            res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
            for id, ann in enumerate(anns):
                s = ann["keypoints"]
                x = s[0::3]
                y = s[1::3]
                x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
                ann["area"] = (x1 - x0) * (y1 - y0)
                ann["id"] = id + 1
                ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
        print("DONE (t={:0.2f}s)".format(time.time() - tic))

        res.dataset["annotations"] = anns
        res.createIndex()
        return res

    def download(self, tarDir=None, imgIds=[]):
        """
        Download COCO images from mscoco.org server.

        Args:
            - tarDir (_type_, optional): COCO results directory name. Defaults to None.
            - imgIds (list, optional): images to be downloaded. Defaults to [].

        Returns:
            _type_: -1 if an error occurs, else None
        """
        if tarDir is None:
            print("Please specify target directory")
            return -1
        if len(imgIds) == 0:
            imgs = self.imgs.values()
        else:
            imgs = self.loadImgs(imgIds)
        N = len(imgs)
        if not os.path.exists(tarDir):
            os.makedirs(tarDir)
        for i, img in enumerate(imgs):
            tic = time.time()
            fname = os.path.join(tarDir, img["file_name"])
            if not os.path.exists(fname):
                urlretrieve(img["coco_url"], fname)
            print(
                "downloaded {}/{} images (t={:0.1f}s)".format(i, N, time.time() - tic)
            )

    def loadNumpyAnnotations(self, data):
        """
        Convert result data from a numpy array [Nx7] where each row contains {imageID,x1,y1,w,h,score,class}
        :param  data (numpy.ndarray)
        :return: annotations (python nested list)
        """
        print("Converting ndarray to lists...")
        assert type(data) == np.ndarray
        print(data.shape)
        assert data.shape[1] == 7
        N = data.shape[0]
        ann = []
        for i in range(N):
            if i % 1000000 == 0:
                print("{}/{}".format(i, N))
            ann += [
                {
                    "image_id": int(data[i, 0]),
                    "bbox": [data[i, 1], data[i, 2], data[i, 3], data[i, 4]],
                    "score": data[i, 5],
                    "category_id": int(data[i, 6]),
                }
            ]
        return ann

    def annToRLE(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE to RLE.
        :return: binary mask (numpy 2D array)
        """
        t = self.imgs[ann["image_id"]]
        h, w = t["height"], t["width"]
        segm = ann["segmentation"]
        if type(segm) == list:
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(segm, h, w)
            rle = maskUtils.merge(rles)
        elif type(segm["counts"]) == list:
            # uncompressed RLE
            rle = maskUtils.frPyObjects(segm, h, w)
        else:
            # rle
            rle = ann["segmentation"]
        return rle

    def annToMask(self, ann):
        """
        Convert annotation which can be polygons, uncompressed RLE, or RLE to binary mask.
        :return: binary mask (numpy 2D array)
        """
        rle = self.annToRLE(ann)
        m = maskUtils.decode(rle)
        return m

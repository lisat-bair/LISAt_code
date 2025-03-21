import json
from copy import deepcopy
from pathlib import Path
from shutil import copy
from typing import Any, Dict, List, OrderedDict, Tuple

from ..coco import COCO
from .darknet import CocoToDarknet
from .img import Img, bbox
from .utils import load_json, save_json

__all__ = ["CocoToComsat"]


class CocoToComsat:
    """Class that helps convert an MS COCO formatted dataset to the COMSAT format"""

    @staticmethod
    def convert(src_images: Path, src_instances: Path, dst_base: Path) -> None:
        """
        Convert source coco dataset to Comsat format.
        """
        coco = COCO(src_instances)

        # for each image
        for image in coco.dataset["images"]:

            # create nested sub directories for this image
            CocoToComsat.init_nested_dirs(dst_base, image)

            # add image to the "imagery" subfolder
            CocoToComsat.add_image(src_images, dst_base, image)

            # create "labels" json and add to subfolder
            CocoToComsat.add_labels(dst_base, image, coco)

            # create "metadata" json and add to subfolder
            CocoToComsat.add_metadata(dst_base, image)

    @staticmethod
    def init_nested_dirs(dst_base: Path, image):
        """
        Initializes a new set of nested folders.
        """
        # Create paths
        image_id = Path(str(image["id"]))
        imagery_path = dst_base / image_id / "imagery"
        labels_path = dst_base / image_id / "labels"
        metadata_path = dst_base / image_id / "metadata"

        # Make dirs
        imagery_path.mkdir(parents=True, exist_ok=True)
        labels_path.mkdir(parents=True, exist_ok=True)
        metadata_path.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def add_image(src_images: Path, dst_base: Path, image):
        """
        .
        """
        image_id = Path(str(image["id"]))
        image_file = Path(image["file_name"])
        source_path = src_images / image_file
        imagery_path = dst_base / image_id / "imagery"

        copy(source_path, imagery_path)

    @staticmethod
    def add_labels(dst_base: Path, image, coco):
        """
        .
        """
        comsat_labels = []

        default_comsat_label = {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[39.542, 46.534], [39.542, 46.534], [39.542, 46.534], [39.542, 46.534]]
                ],
            },
            "properties": {
                "label": {"name": "Category 1", "ontology_iri": ""},
                "pixel_coordinates": [[367, 520], [367, 520], [367, 520], [367, 520]],
                "label_acquisition_date": None,
                "image_acquisition_date": None,
                "peer_reviewed": False,
            },
        }

        # Get annotations for this image
        coco_anns = coco.imgToAnns[image["id"]]

        for ann in coco_anns:
            new_comsat_label = deepcopy(default_comsat_label)
            comsat_poly_coords = CocoToComsat.coco_to_comsat_poly_coords(ann, image)
            comsat_pixel_coords = CocoToComsat.coco_to_comsat_pixel_coords(ann)
            comsat_label_name = CocoToComsat.get_category_name(ann, coco)

            new_comsat_label["geometry"]["coordinates"] = comsat_poly_coords
            new_comsat_label["properties"]["pixel_coordinates"] = comsat_pixel_coords
            new_comsat_label["properties"]["label"]["name"] = comsat_label_name

            comsat_labels.append(new_comsat_label)

        root_json = {"type": "FeatureCollection", "features": comsat_labels}

        image_id = str(image["id"])
        labels_file_name = Path(f"LABELS_{image_id}.json")
        labels_file_path = dst_base / image_id / "labels" / labels_file_name
        save_json(labels_file_path, root_json)

    @staticmethod
    def coco_to_comsat_pixel_coords(ann):
        """
        Reformats coco segmentation to comsat in terms of pixel coordinates

            - coco poly format is [ [x1, y1, x2, y2, ... ] ]
            - comsat poly format is [ [ [x1, y1], [x2, y2], ... ] ]
        """
        coco_pixel_coords = ann["segmentation"]

        comsat_pixel_coords = []

        for group in coco_pixel_coords:
            # split the coco pixel coords by even/odd elements and zip
            comsat_pixel_coords.append([[int(x), int(y)] for x, y in zip(group[::2], group[1::2])])

        return comsat_pixel_coords

    @staticmethod
    def coco_to_comsat_poly_coords(ann, image):
        """
        Reformats coco segmentation to comsat in terms of image dim percentage coordinates

            - coco poly format is [ [x1, y1, x2, y2, ... ] ]
            - comsat poly format is [ [ [x1, y1], [x2, y2], ... ] ]
        """

        w = float(image["width"])
        h = float(image["height"])

        comsat_pixel_coords = CocoToComsat.coco_to_comsat_pixel_coords(ann)

        comsat_poly_coords = []

        for group in comsat_pixel_coords:

            # divide pixel coords by image dims
            comsat_poly_coords.append([[x[0] / w, x[1] / h] for x in group])

        return comsat_poly_coords

    @staticmethod
    def get_category_name(ann, coco):
        """
        Returns the category name for the annotation given
        """

        cats = coco.loadCats(coco.getCatIds())

        list.sort(cats, key=lambda c: c["id"])

        # Dictionary to lookup category name from category id:
        cat_name_lookup = {c["id"]: c["name"] for c in cats}

        return cat_name_lookup[ann["category_id"]]

    @staticmethod
    def add_metadata(dst_base: Path, image):
        """
        .
        """
        root_json = {
            "image_id": image["id"],
            "image_width": image["width"],
            "image_height": image["height"],
            "image_source": "XVIEW",
        }
        image_id = str(image["id"])
        metadata_file_name = f"METADATA_{image_id}.json"
        metadata_file_path = dst_base / image_id / "metadata" / metadata_file_name
        save_json(metadata_file_path, root_json)

    @staticmethod
    def generate_names(names_path: Path, coco: COCO) -> None:
        categories = [c["name"] + "\n" for c in coco.dataset["categories"]]
        with open(names_path, "w") as names_file:
            names_file.writelines(categories)

    @staticmethod
    def generate_label_files(
        images: Dict[int, Img],
        labels_path: Path,
        base_path: Path,
        dataset_name: str,
        data_split: str,
    ) -> List[str]:
        """
        Generates one .txt file for each image in the coco-formatted dataset. The .txt
        files contain the annotations in yolo/Darknet format.
        """
        # Convert:
        img_paths = set()
        for img_id, img in images.items():
            if img.has_anns():
                label_path = labels_path / img.get_label_path(labels_path)
                with open(label_path, "w") as label_file:
                    img.write_darknet_anns(label_file)
                img_path = img.get_img_path(base_path, dataset_name, data_split)
                assert img_path.exists(), f"Image doesn't exist {img_path}"
                img_paths.add(str(img_path) + "\n")
        return list(img_paths)

    @staticmethod
    def generate_image_list(
        base_path: Path, dataset_name: str, image_paths: List[str], data_split: str
    ) -> None:
        """Generates train.txt, val.txt, etc, txt file with list of image paths."""
        listing_path = base_path / dataset_name / f"{data_split}.txt"
        print("Listing path: ", listing_path)
        with open(listing_path, "w") as listing_file:
            listing_file.writelines(image_paths)

    @staticmethod
    def build_db(coco: COCO) -> Dict[int, Img]:
        """
        Builds a datastructure of images. All annotations are grouped into their
        corresponding images to facilitate generating the Darknet formatted metadata.

        Args:
            coco: a pycocotools.coco COCO instance

        Returns: Dictionary whose keys are image id's, and values are Img instances that
            are loaded with all the image info and annotations from the coco-formatted
            json
        """
        anns = coco.dataset["annotations"]
        images: Dict[int, Img] = {}
        # Build images data structure:
        for i, ann in enumerate(anns):
            ann = CocoToDarknet.get_ann(ann)
            if ann.img_id not in images:
                coco_img = coco.dataset["images"][ann.img_id]
                images[ann.img_id] = Img(
                    ann.img_id,
                    coco_img["file_name"],
                    float(coco_img["width"]),
                    float(coco_img["height"]),
                )
            img = images[ann.img_id]
            img.add_ann(ann)
        return images

    @staticmethod
    def get_ann(ann):
        """
        Gets a bbox instance from an annotation element pulled from the coco-formatted
        json
        """
        box = ann["bbox"]
        return bbox(ann["image_id"], ann["category_id"], box[0], box[1], box[2], box[3])

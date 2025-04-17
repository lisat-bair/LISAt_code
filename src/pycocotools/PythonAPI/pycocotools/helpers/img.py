from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, OrderedDict, Tuple


@dataclass
class bbox:
    """
    Data class to store a bounding box annotation instance
    """

    img_id: int
    cat_id: int
    x_center: float
    y_center: float
    width: float
    height: float


class Img:
    """A helper class to store image info and annotations."""

    anns: List[bbox]

    def __init__(self, id: int, filename: str, width: float, height: float) -> None:
        self.id: int = id
        self.filename: str = filename
        self.width: float = width
        self.height: float = height
        self.anns = []

    def add_ann(self, ann: bbox) -> None:
        """Add an annotation to the image"""
        self.anns.append(ann)

    def get_anns(self) -> List[bbox]:
        """
        Gets annotations, possibly filters them in prep for converting to yolo/Darknet
        format.
        """
        return self.anns

    def to_darknet(self, box: bbox) -> bbox:
        """Convert a BBox from coco to Darknet format"""
        # COCO bboxes define the topleft corner of the box, but yolo expects the x/y
        # coords to reference the center of the box. yolo also requires the coordinates
        # and widths to be scaled by image dims, down to the range [0.0, 1.0]
        return bbox(
            self.id,
            box.cat_id,
            (box.x_center + (box.width / 2.0)) / self.width,
            (box.y_center + (box.height / 2.0)) / self.height,
            box.width / self.width,
            box.height / self.height,
        )

    def write_darknet_anns(self, label_file) -> None:
        """Writes bounding boxes to specified file in yolo/Darknet format"""
        # It's a bit leaky abstraction to have Img handle writing to file but it's
        # convenient b/c we have access to img height and width here to scale the bbox
        # dims. Same goes for .to_darknet()
        anns = self.get_anns()
        for box in anns:
            box = self.to_darknet(box)
            label_file.write(
                f"{box.cat_id} {box.x_center} {box.y_center} {box.width} {box.height}\n"
            )

    def has_anns(self) -> List[bbox]:
        """
        Returns true if this image instance has at least one bounding box (after any
        filters are applied)
        """
        # TODO: Can add filter to only return true if annotations have non-zero area: I
        # saw around ~5 or 6 annotations in the v2_train_chipped.json that had zero
        # area, not sure if those might cause problems for yolo
        return self.anns

    def get_label_path(self, base_path: Path) -> Path:
        return base_path / self.filename.replace("jpeg", "txt").replace("jpg", "txt")

    def get_img_path(self, base_path: Path, dataset_name: str, data_split: str) -> Path:
        return base_path / dataset_name.replace("_tiny", "") / "images" / data_split / self.filename

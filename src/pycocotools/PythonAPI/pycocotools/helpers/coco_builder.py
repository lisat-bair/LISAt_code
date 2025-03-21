import pickle
from copy import deepcopy
from pathlib import Path
from typing import Any, Counter, Optional, Union

from ..coco import COCO, Ann, Cat, Image, Ref
from .utils import save_json

__all__ = ["CocoJsonBuilder", "COCOShrinker"]


class CocoJsonBuilder(object):
    """
    A class used to help build coco-formatted json from scratch.
    """

    def __init__(
        self,
        categories: list[Cat],
        subset_cat_ids: list[int] = [],
        dest_path: Union[str, Path] = "",
        dest_name="",
        keep_empty_images=True,
        is_ref_dataset: bool = False,
        dataset_name: str = None,
        split_by: str = None,
        source_coco: Optional[COCO] = None,
    ):
        """
        Args:

            categories: this can be the COCO.dataset['categories'] property if you
                are building a COCO json derived from an existing COCO json and don't want to modify
                the classes. It's a list of dictionary objects. Each dict has three keys: "id":int =
                category id, "supercatetory": str = name of parent category, and a "name": str =
                name of category.

            subset_cat_ids: list of category_id's. If specified, the builder will exclude
                annotations for any categories not in this list.

            dest_path: str or pathlib.Path instance, holding the path to directory where
                the new COCO formatted annotations file (dest_name) will be saved.

            dest_name: str of the filename where the generated json will be saved to.
        """
        if dest_path:
            if isinstance(dest_path, str):
                dest_path = Path(dest_path)
            assert dest_path.is_dir(), "dest_path should be a directory: " + str(
                dest_path
            )
        self.categories = categories
        self.subset_cat_ids = subset_cat_ids
        self.new_categories = []
        self.reindex_cat_id = {}  # maps from old to new cat id
        if self.subset_cat_ids:
            cat_counter = 1  # one-indexing
            for cat in self.categories:
                if cat["id"] in self.subset_cat_ids:
                    new_cat = deepcopy(cat)
                    new_cat["id"] = cat_counter
                    self.reindex_cat_id[cat["id"]] = cat_counter
                    cat_counter += 1
                    self.new_categories.append(new_cat)
        self.keep_empty_images = keep_empty_images
        self.dest_path = Path(dest_path)
        self.dest_name = dest_name
        self.images: list[Image] = []
        self.annotations: list[Ann] = []
        self.is_ref_dataset = is_ref_dataset
        self.refs: list[Ref] = []
        self.max_ann_id = -1
        self.max_ref_id = -1
        dest_path.mkdir(parents=True, exist_ok=True)
        self.source_coco = source_coco
        self.dataset_name = dataset_name
        self.split_by = split_by
        # assert self.dest_path.exists(), f"dest_path: '{self.dest_path}' does not exist"
        # assert (
        #     self.dest_path.is_dir()
        # ), f"dest_path: '{self.dest_path}' is not a directory"

    def generate_info(self) -> dict[str, str]:
        """
        Returns: A dictionary of descriptive info about the dataset.
        """
        info_json = (
            {
                "description": "Some Dataset",
                "url": "http://somedataset.org/",
                "version": "1.0",
                "year": 2013,
                "contributor": "[Contributor]",
                "date_created": "2023/01/01",
            }
            if not self.source_coco
            else self.source_coco.dataset["info"]
        )
        return info_json

    def generate_licenses(self) -> list[dict[str, Any]]:
        """Returns the json hash for the licensing info."""
        return (
            [
                {
                    "url": "http://creativecommons.org/licenses/by-nc-sa/4.0/",
                    "id": 1,
                    "name": "Attribution-NonCommercial-ShareAlike License",
                }
            ]
            if not self.source_coco
            else self.source_coco.dataset["licenses"]
        )

    def add_image(
        self, img: Image, annotations: list[Ann], refs: Optional[list[Ref]] = []
    ) -> None:
        """
        Add an image and it's annotations to the coco json.

        Args:
            img: A dictionary of image attributes. This gets added verbatim to the
                json, so in typical use cases when you are building a coco json from an existing
                coco json, you would just pull the entire coco.imgs[img_id] object and pass it as
                the value for this parameter.

            annotations: annotations of the image to add. list of dictionaries.
                Each dict is one annotation, it contains all the properties of the annotation that
                should appear in the coco json. For example, when using this json builder to build
                JSON's for a train/val split, the annotations can be copied straight from the coco
                object for the full dataset, and passed into this parameter.

        Returns: None
        """
        img = deepcopy(img)
        if self.keep_empty_images and not (annotations):
            self.images.append(img)
            for ann in annotations:
                self._add_ann(ann)
        elif annotations:
            self.images.append(img)
            for ann in annotations:
                self._add_ann(ann)
        if refs is not None and len(refs) > 0:
            for ref in refs:
                if "image_id" in ref:
                    assert (
                        ref["image_id"] == img["id"]
                    ), f"ref[img_id]:{ref['image_id']}!=img_id:{img['id']}"
                else:
                    ref["image_id"] = img["id"]
                assert (
                    "ann_id" in ref
                ), f"ref is missing ann_id. Ref: {ref}, img_id:{img['id']}"
                assert (
                    "sentences" in ref
                ), f"ref is missing 'sentences'. Ref: {ref}, img_id:{img['id']}"
                assert (
                    len(ref["sentences"]) > 0
                ), f"ref has no sentences. Ref: {ref}, img_id:{img['id']}"
                self._add_ref(ref)

    def _add_ann(self, ann: Ann) -> None:
        if not self.subset_cat_ids:
            new_ann = deepcopy(ann)
            self.annotations.append(new_ann)
        elif self.subset_cat_ids and ann["category_id"] in self.subset_cat_ids:
            new_ann = deepcopy(ann)
            new_ann["category_id"] = self.reindex_cat_id[ann["category_id"]]
            self.annotations.append(new_ann)

    def _add_ref(self, ref: Ref) -> None:
        assert self.is_ref_dataset, "Operation only valid for refer_seg datasets."
        if self.refs is None:
            self.refs = []
        assert "sentences" in ref
        assert "sent_ids" in ref
        assert len(ref["sentences"]) == len(ref["sent_ids"])
        new_ref = deepcopy(ref)

        if "ref_id" not in new_ref:
            new_ref["ref_id"] = -1
        self.refs.append(new_ref)

    def _reindex(self) -> None:
        "Set ann_id's and ref_id's"

        def assert_unique_ids(items: list[Any], id_key: str):
            id_counts = Counter([item[id_key] for item in items])
            for item_id, id_count in id_counts.items():
                if item_id >= 0:
                    assert id_count == 1, f"Duplicate items for {id_key}: {item_id}"

            max_id = max(list(id_counts.keys()))
            return max_id

        # Ensure existing ann_id's are unique:
        max_ann_id = assert_unique_ids(self.annotations, "id")
        # assign ann_ids:
        for ann in self.annotations:
            if ann["id"] < 0:
                max_ann_id += 1
                ann["id"] = max_ann_id

        # Ensure existing ref_id's are unique:
        max_ref_id = assert_unique_ids(self.refs, "ref_id")
        # Assign ref_ids
        if self.is_ref_dataset:
            assert len(self.refs) > 0, "self.refs is None or empty"
            for ref in self.refs:
                if ref["ref_id"] < 0:
                    max_ref_id += 1
                    ref["ref_id"] = max_ref_id

        total_anns = len(self.annotations)
        total_ann_ids = len({a["id"] for a in self.annotations})
        print(f"Total anns:{total_anns}")
        print(f"ann_ids: {total_ann_ids}")
        assert total_anns == total_ann_ids, f"{total_anns} != {total_ann_ids}"

    def get_json(self) -> dict[str, Any]:
        """Returns the full json for this instance of coco json builder."""
        root_json = {}
        if self.new_categories:
            root_json["categories"] = self.new_categories
        else:
            root_json["categories"] = self.categories
        root_json["info"] = self.generate_info()
        root_json["licenses"] = self.generate_licenses()
        root_json["images"] = self.images
        root_json["annotations"] = self.annotations
        return root_json

    def _save_ref_file(self, output_dir: Path) -> None:
        assert output_dir.is_dir(), str(output_dir)
        ref_path = output_dir / f"refs({self.split_by}).p"

        print(f"Saving {len(self.refs)} refs to file: ", ref_path)
        with open(ref_path, "wb") as file:
            pickle.dump(self.refs, file)
        file_size_MB = ref_path.stat().st_size / (1024 * 1024)
        print(f"Saved refs file {ref_path}' ({file_size_MB:.02f}MB)")

    def save(self) -> Path:
        """Saves the json to the dest_path/dest_name location."""
        self._reindex()
        file_path = (self.dest_path / self.dest_name).absolute()
        dataset = self.get_json()
        print(
            f"Writing coco_builder (num_img: { len(dataset['images']) }, "
            f"num_ann: { len(dataset['annotations']) }) output to: '{file_path}'"
        )
        save_json(file_path, data=dataset)
        file_size_MB = file_path.stat().st_size / (1024 * 1024)
        print(f"Saved {file_path}' ({file_size_MB:.02f}MB)")

        if self.is_ref_dataset:
            self._save_ref_file(file_path.parent)
        return file_path


class COCOShrinker:
    """
    Shrinker takes an MS COCO formatted dataset and creates a tiny version of it.
    """

    def __init__(self, dataset_path: Path, keep_empty_images=False, **kwargs) -> None:
        """
        Creates a shrunken version of a COCO formatted dataset.

        :param dataset_path: Path to source COCO json file
        :type dataset_path: Path
        :param keep_empty_images: Whether or not to keep images that don't have any annotations.
            Defaults to False
        :type keep_empty_images: bool, optional

        For kwargs, these are passed to the source COCO.__init__().
        """
        assert dataset_path.exists(), f"dataset_path: '{dataset_path}' does not exist"
        assert dataset_path.is_file(), f"dataset_path: '{dataset_path}' is not a file"
        self.base_path: Path = dataset_path.parent
        self.dataset_path: Path = dataset_path
        self.keep_empty_images = keep_empty_images
        self.kwargs = kwargs

    def shrink(
        self,
        target_filename: str,
        size: int = 512,
        output_dir: Path = None,
        **kwargs,
    ) -> Path:
        """
        Create a toy sized version of dataset so we can use it just for testing if code
        runs, not for real training.

        Args:
            name: filename to save the tiny dataset to.
            size: number of items to put into the output. The first <size>
                elements from the input dataset are placed into the output.

        Returns: Path where the new COCO json file is saved.
        """
        # Create subset
        assert target_filename, "'target_filename' argument must not be empty"
        if output_dir is not None:
            dest_path: Path = output_dir / target_filename
        else:
            dest_path: Path = self.base_path / target_filename
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Creating subset of {self.dataset_path}, of size: {size}, at: {dest_path}"
        )
        coco = COCO(self.dataset_path, **self.kwargs)
        print("Loaded coco source ", len(coco.imgs), len(coco.anns), len(coco.refs))
        print(coco.is_ref_dataset, coco.dataset_name, coco.split_by)
        builder = CocoJsonBuilder(
            coco.dataset["categories"],
            dest_path=dest_path.parent,
            dest_name=dest_path.name,
            source_coco=coco,
            **kwargs,
        )
        subset_img_ids = coco.getImgIds()[:size]
        for img_id in subset_img_ids:
            anns = coco.imgToAnns[img_id]
            refs = None
            if coco.is_ref_dataset:
                refs = coco.img_to_refs[img_id]
            if anns or self.keep_empty_images:
                builder.add_image(coco.imgs[img_id], anns, refs)
        builder.save()
        return dest_path

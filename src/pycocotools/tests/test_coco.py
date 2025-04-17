import unittest
from pathlib import Path

from pycocotools.coco import COCO


class TestCOCO(unittest.TestCase):
    def setUp(self):
        pass

    def test_coco_json_loading(self):
        ann_path = (
            Path("") / "data/annotations/xview_coco_v2_tiny_val_chipped.json"
        ).resolve()
        coco = COCO(ann_path)

        self.assertEqual(len(coco.imgs), 64)
        self.assertEqual(len(coco.anns), 3584)
        self.assertEqual(len(coco.cats), 35)


if __name__ == "__main__":
    unittest.main()

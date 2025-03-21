import os
import json
import shutil
from PIL import Image

# -----------------------------------------------------------------------
#  CONFIGURATION
# -----------------------------------------------------------------------

# Cherry-picked filenames (with .jpg extension)
CHERRY_PICKED_IMAGES = [
    "COCO_train_000000000264.jpg",
    "COCO_train_000000000420.jpg",
    "COCO_train_000000000534.jpg",
    "COCO_train_000000000629.jpg",
    "COCO_train_000000000692.jpg",
    "COCO_train_000000001068.jpg",
    "COCO_train_000000001151.jpg",
    "COCO_train_000000001291.jpg",
    "COCO_train_000000001322.jpg",
    "COCO_train_000000001455.jpg",
    "COCO_train_000000001525.jpg",
    "COCO_train_000000001648.jpg",
    "COCO_train_000000001677.jpg",
    "COCO_train_000000001727.jpg",
    "COCO_train_000000001807.jpg",
    "COCO_train_000000002226.jpg",
    "COCO_train_000000002295.jpg",
    "COCO_train_000000002657.jpg",
    "COCO_train_000000003055.jpg",
    "COCO_train_000000003205.jpg",
    "COCO_train_000000003463.jpg",
    "COCO_train_000000003615.jpg",
    "COCO_train_000000004268.jpg",
    "COCO_train_000000004626.jpg",
    "COCO_train_000000004779.jpg",
    "COCO_train_000000005090.jpg",
    "COCO_train_000000005513.jpg",
    "COCO_train_000000005598.jpg",
    "COCO_train_000000005810.jpg",
    "COCO_train_000000006227.jpg",
    "COCO_train_000000006463.jpg",
    "COCO_train_000000006816.jpg",
    "COCO_train_000000007717.jpg",
    "COCO_train_000000008383.jpg",
    "COCO_train_000000008436.jpg",
    "COCO_train_000000008854.jpg",
    "COCO_train_000000008859.jpg",
    "COCO_train_000000009585.jpg",
    "COCO_train_000000009693.jpg",
    "COCO_train_000000010186.jpg",
    "COCO_train_000000010323.jpg",
    "COCO_train_000000010575.jpg",
    "COCO_train_000000010718.jpg",
    "COCO_train_000000010732.jpg",
    "COCO_train_000000010806.jpg",
    "COCO_train_000000011431.jpg",
    "COCO_train_000000011546.jpg",
    "COCO_train_000000011850.jpg",
    "COCO_train_000000012020.jpg",
    "COCO_train_000000012041.jpg",
    "COCO_train_000000012557.jpg",
    "COCO_train_000000013309.jpg",
    "COCO_train_000000013414.jpg",
    "COCO_train_000000013453.jpg",
    "COCO_train_000000013689.jpg",
    "COCO_train_000000013888.jpg",
    "COCO_train_000000014188.jpg",
    "COCO_train_000000014386.jpg",
    "COCO_train_000000014707.jpg",
    "COCO_train_000000014917.jpg",
    "COCO_train_000000015739.jpg",
    "COCO_train_000000016109.jpg"
]

# JSON with queries
JSON_PATH = "/home/wenhan/Projects/Xview/xview_test_0110_large.json"

# Paths of each type of image
COMPARISON_FOLDER = "/home/wenhan/Projects/sesame/combined_comparison_outputs"
ORIGINAL_FOLDER   = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/large"
LISAt_FOLDER      = "/home/wenhan/Projects/sesame/.geo_0120_new_inference_dir/large_png_inference/seg_rgb"
LISA_FOLDER       = "/home/wenhan/Projects/sesame/LISA-13B-llama2-v1_inference_dir/large_png_inference/seg_rgb"
GT_FOLDER         = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/gt_large"

# Output root folder for the cherry-picked sets
CHERRY_PICKED_OUTPUT = "/home/wenhan/Projects/sesame/cherry_picked"

# -----------------------------------------------------------------------
#  MAIN SCRIPT
# -----------------------------------------------------------------------

def main():
    # 1) Read JSON to build { "COCO_train_000000010576.jpg": "query text", ... }
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    dict_of_queries = {}
    for entry in data:
        img_name = entry["image"]
        query_txt = entry["query"]["query"]
        dict_of_queries[img_name] = query_txt

    os.makedirs(CHERRY_PICKED_OUTPUT, exist_ok=True)

    for cherry_image_name in CHERRY_PICKED_IMAGES:
        base_name = os.path.splitext(cherry_image_name)[0]  # e.g. "COCO_train_000000000033"

        # 2) Make a subfolder for each image
        subfolder = os.path.join(CHERRY_PICKED_OUTPUT, base_name)
        os.makedirs(subfolder, exist_ok=True)

        # 3) Check the file paths we need
        comparison_in = os.path.join(COMPARISON_FOLDER, f"{base_name}_combined.png")
        original_in   = os.path.join(ORIGINAL_FOLDER, cherry_image_name)
        lisat_in      = os.path.join(LISAt_FOLDER, f"{base_name}_seg_rgb.png")
        lisa_in       = os.path.join(LISA_FOLDER, f"{base_name}_seg_rgb.png")
        gt_in         = os.path.join(GT_FOLDER, f"segmented_{base_name}.png")

        # Collect them into a dictionary for convenience
        paths_required = {
            "comparison": comparison_in,
            "original":   original_in,
            "lisat":      lisat_in,
            "lisa":       lisa_in,
            "gt":         gt_in
        }

        # 4) Verify all needed files exist
        missing = [key for key, path in paths_required.items() if not os.path.exists(path)]
        if missing:
            print(f"Skipping {cherry_image_name}. Missing: {missing}")
            for m in missing:
                print("  " + paths_required[m])
            continue
        
        # 5) Copy or convert the files to subfolder, renaming them
        # a) comparison -> <base_name>_comparison.png
        comparison_out = os.path.join(subfolder, f"{base_name}_comparison.png")
        shutil.copy2(paths_required["comparison"], comparison_out)

        # b) original -> <base_name>_original.png (converted from .jpg to .png)
        original_out = os.path.join(subfolder, f"{base_name}_original.png")
        with Image.open(paths_required["original"]) as img:
            img.save(original_out)

        # c) LISAt -> <base_name>_LISAt.png
        lisat_out = os.path.join(subfolder, f"{base_name}_LISAt.png")
        shutil.copy2(paths_required["lisat"], lisat_out)

        # d) LISA -> <base_name>_LISA.png
        lisa_out = os.path.join(subfolder, f"{base_name}_LISA.png")
        shutil.copy2(paths_required["lisa"], lisa_out)

        # e) GT -> <base_name>_GT.png
        gt_out = os.path.join(subfolder, f"{base_name}_GT.png")
        shutil.copy2(paths_required["gt"], gt_out)

        # 6) Create a .txt file with the query
        query_text = dict_of_queries.get(cherry_image_name, "")
        txt_out = os.path.join(subfolder, f"{base_name}.txt")
        with open(txt_out, "w") as tfile:
            tfile.write(query_text + "\n")

        print(f"Successfully packaged {cherry_image_name} in {subfolder}.")

if __name__ == "__main__":
    main()

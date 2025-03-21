import json
import os
from PIL import Image, ImageDraw
import numpy as np
import matplotlib.pyplot as plt

# Paths
json_file_path = '/home/wenhan/Projects/Xview/xview_coco_v2_train_chipped_balanced.json'
images_dir = '/home/gbiamby/data/xview/coco_chipped_512/train/'
output_dir = '/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_new/gt_train_new'
os.makedirs(output_dir, exist_ok=True)

def overlay_segmentation_and_save(data):
    image_path = os.path.join(images_dir, data["image"])
    image = Image.open(image_path).convert('RGBA')

    # Create transparent mask for segmentation
    mask_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
    mask_draw = ImageDraw.Draw(mask_layer)

    # Draw segmentation polygons
    for segmentation in data["segmentation"]:
        segmentation_pts = np.array(segmentation).reshape(-1, 2).astype(np.int32)
        mask_draw.polygon([tuple(pt) for pt in segmentation_pts], fill=(0, 255, 0, 100))  # Semi-transparent green

    # Apply mask to the original image
    final_image_with_mask = Image.alpha_composite(image, mask_layer)

    # Construct filename and save
    file_name = f"{os.path.splitext(data['image'])[0]}_segmented.png"
    full_file_path = os.path.join(output_dir, file_name)
    final_image_with_mask.save(full_file_path, 'PNG')

    print(f"Saved: {full_file_path}")

# Read JSON file and process each entry
with open(json_file_path, 'r') as file:
    annotations = json.load(file)
    for data in annotations:
        overlay_segmentation_and_save(data)

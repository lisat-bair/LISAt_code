import os
import shutil

# Define the source and destination directories
source_dir = '/home/wenhan/Projects/sesame/.geo_0120_new_inference_dir/large_png_inference'
source_dir = '/home/wenhan/Projects/sesame/LISA-13B-llama2-v1_inference_dir/large_png_inference'
seg_mask_dir = os.path.join(source_dir, 'seg_mask')
seg_rgb_dir = os.path.join(source_dir, 'seg_rgb')

# Create destination directories if they don't exist
os.makedirs(seg_mask_dir, exist_ok=True)
os.makedirs(seg_rgb_dir, exist_ok=True)

# Iterate through all files in the source directory
for filename in os.listdir(source_dir):
    # Check if the file ends with the expected patterns
    if filename.endswith('_seg_mask.png'):
        shutil.move(os.path.join(source_dir, filename), os.path.join(seg_mask_dir, filename))
    elif filename.endswith('_seg_rgb.png'):
        shutil.move(os.path.join(source_dir, filename), os.path.join(seg_rgb_dir, filename))

print("Images sorted successfully.")

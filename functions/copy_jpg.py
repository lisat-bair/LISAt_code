import os
import shutil

source_dir = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/train_7205"
dest_dir = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/train_7205_augment"

os.makedirs(dest_dir, exist_ok=True)

for filename in os.listdir(source_dir):
    if filename.lower().endswith(".jpg"):
        src_pth = os.path.join(source_dir, filename)
        dest_pth = os.path.join(dest_dir, filename)
        shutil.copy2(src_pth, dest_pth)

print("All jpg files are copied")
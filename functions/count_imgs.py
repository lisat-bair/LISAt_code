import os

# Specify the folder path
folder_path = '/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/train_7205_augment'

# Count the number of jpg files
jpg_files_count = len([file for file in os.listdir(folder_path) if file.endswith('.jpg')])

print(f"Number of JPG files: {jpg_files_count}")

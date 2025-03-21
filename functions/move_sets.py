import os
import random
import shutil

def move_files(val_dir, train_dir, num_to_move=1499):
    """
    Moves 'num_to_move' image+json pairs from val_dir to train_dir.
    """

    # 1. Collect all JPG files in 'val_dir'
    jpg_files = [f for f in os.listdir(val_dir) if f.endswith('.jpg')]

    # 2. Make sure we don't exceed the total count
    if num_to_move > len(jpg_files):
        raise ValueError(f"Requested to move {num_to_move} sets, but only {len(jpg_files)} JPG files found.")

    # 3. Randomly select 'num_to_move' JPG files
    selected_jpg_files = random.sample(jpg_files, num_to_move)

    # 4. For each selected JPG file, move both the JPG and its JSON
    for jpg_filename in selected_jpg_files:
        json_filename = jpg_filename.replace('.jpg', '.json')
        
        src_jpg = os.path.join(val_dir, jpg_filename)
        dst_jpg = os.path.join(train_dir, jpg_filename)
        
        src_json = os.path.join(val_dir, json_filename)
        dst_json = os.path.join(train_dir, json_filename)

        # Check JSON exists before moving (just in case)
        if not os.path.exists(src_json):
            print(f"WARNING: JSON file {json_filename} does not exist in {val_dir}. Skipping...")
            continue
        
        # Move JPG
        shutil.move(src_jpg, dst_jpg)
        # Move JSON
        shutil.move(src_json, dst_json)

    print(f"Moved {num_to_move} JPG+JSON pairs from '{val_dir}' to '{train_dir}'.")

if __name__ == "__main__":
    val_path = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/val"
    train_path = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/train"
    
    # Number of sets (jpg+json) to move
    num_files_to_move = 1499
    
    move_files(val_path, train_path, num_files_to_move)

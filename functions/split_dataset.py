import os
import shutil

train_dir = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/train"
test_dir = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/test"

# Create the test_dir if it does not exist
os.makedirs(test_dir, exist_ok=True)

# Step 1: Gather all .jpg files in the train_dir
jpg_files = sorted(f for f in os.listdir(train_dir) if f.endswith(".jpg"))

# Step 2: Determine how many to keep in train and how many to move
num_train = 4500  # first 4500
num_test = 1500   # remaining 1500

assert len(jpg_files) == 6000, "Expected 6000 JPG files in the directory."

# Step 3: Identify the test set (the last 1500 in sorted order)
test_jpg_files = jpg_files[num_train:]  # from index 4500 to the end

# Step 4: Move test pairs (.jpg and corresponding .json) to test_dir
for jpg_file in test_jpg_files:
    json_file = jpg_file.replace(".jpg", ".json")
    
    src_jpg = os.path.join(train_dir, jpg_file)
    src_json = os.path.join(train_dir, json_file)
    dst_jpg = os.path.join(test_dir, jpg_file)
    dst_json = os.path.join(test_dir, json_file)
    
    # Move .jpg
    if os.path.exists(src_jpg):
        shutil.move(src_jpg, dst_jpg)
    else:
        print(f"Warning: {src_jpg} does not exist.")
    
    # Move .json
    if os.path.exists(src_json):
        shutil.move(src_json, dst_json)
    else:
        print(f"Warning: {src_json} does not exist.")

print("Done. The first 4500 .jpg+.json pairs remain in train_dir; the last 1500 are moved to test_dir.")

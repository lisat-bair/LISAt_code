import os

def count_json_files(folder_path):
    # Count JSON files in the specified folder
    json_files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    return len(json_files)

# Specify the path to your folder
folder_path = '/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/train'

# Count and display the number of JSON files
json_file_count = count_json_files(folder_path)
print(f"Number of JSON files in '{folder_path}': {json_file_count}")

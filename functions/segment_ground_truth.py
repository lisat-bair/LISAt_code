import os
import json
from PIL import Image, ImageDraw
import numpy as np

# Define paths
base_image_path = '/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/large'
output_folder = '/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/gt_large'
os.makedirs(output_folder, exist_ok=True)

# Function to create and save segmentation masks
def generate_segmentation_images():
    for file_name in os.listdir(base_image_path):
        if file_name.endswith('.json'):
            # Load JSON data
            json_path = os.path.join(base_image_path, file_name)
            with open(json_path, 'r') as f:
                data = json.load(f)

            # Get image file name and points for the mask
            image_name = data['shapes'][0]['image_name']
            points = data['shapes'][0]['points']
            image_path = os.path.join(base_image_path, image_name)

            try:
                # Open the image
                image = Image.open(image_path).convert('RGBA')
            except FileNotFoundError:
                print(f"Image file '{image_name}' not found.")
                continue

            # Create a mask layer and draw the polygon
            mask_layer = Image.new('RGBA', image.size, (0, 0, 0, 0))
            mask_draw = ImageDraw.Draw(mask_layer)
            #mask_draw.polygon([tuple(pt) for pt in points], fill=(0, 255, 0, 100))  # Semi-transparent green
            mask_draw.polygon([tuple(pt) for pt in points], fill=(255, 0, 0, 178))  # Semi-transparent red

            # Composite the original image with the mask
            final_image = Image.alpha_composite(image, mask_layer)

            # Save the result
            output_path = os.path.join(output_folder, f"segmented_{os.path.splitext(image_name)[0]}.png")
            final_image.save(output_path, format="PNG")

            print(f"Saved segmented image: {output_path}")

# Run the function to generate segmentation images
generate_segmentation_images()

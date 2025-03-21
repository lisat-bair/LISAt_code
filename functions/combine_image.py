import os
import json
from PIL import Image, ImageDraw, ImageFont

# Adjust paths if necessary
JSON_PATH = "/home/wenhan/Projects/Xview/xview_test_0110_large.json"

LISAt_FOLDER = "/home/wenhan/Projects/sesame/.geo_0120_new_inference_dir/large_png_inference/seg_rgb"
LISA_FOLDER  = "/home/wenhan/Projects/sesame/LISA-13B-llama2-v1_inference_dir/large_png_inference/seg_rgb"
GT_FOLDER    = "/home/wenhan/Projects/sesame/dataset/reason_seg/geo_reason_seg_0110/gt_large"

# Output directory for combined images
OUTPUT_DIR   = "/home/wenhan/Projects/sesame/combined_comparison_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Optional: If you want to specify a font for drawing text
# Make sure to have a valid ttf font file, e.g., /usr/share/fonts/truetype/dejavu/DejaVuSans.ttf
try:
    FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", size=20)
except:
    # If the above font path doesn't exist, fall back to a default PIL font
    FONT = ImageFont.load_default()

def combine_images_and_text(query_text, image_name,
                            path_lisat, path_lisa, path_gt,
                            save_path, 
                            label_lisat="LISAt", 
                            label_lisa="LISA", 
                            label_gt="Ground Truth"):
    """
    Combine three images (LISAt, LISA, GT) side by side, and place
    text (query and image name) at the top plus labels under each image.
    Then save the combined image to `save_path`.
    """

    # Load the three images
    img_lisat = Image.open(path_lisat).convert("RGB")
    img_lisa  = Image.open(path_lisa).convert("RGB")
    img_gt    = Image.open(path_gt).convert("RGB")

    # Decide if you want to resize them to a uniform height or width
    # For simplicity, let's keep them as is and align them by height
    # so that they appear side-by-side in a row.

    # Step 1: Figure out combined width/height for the three images in a row
    widths  = [img_lisat.width, img_lisa.width, img_gt.width]
    heights = [img_lisat.height, img_lisa.height, img_gt.height]

    total_width = sum(widths)
    max_height  = max(heights)

    # Additional space at the top for the query and image_name text
    # Additional space under each image for the label text
    padding_top    = 80   # space for top text region
    padding_bottom = 40   # space for bottom labels
    spacing        = 10   # spacing between images horizontally

    combined_width  = total_width + spacing * 2  # horizontal padding between images
    combined_height = max_height + padding_top + padding_bottom

    # Create a new blank image (white background)
    combined_img = Image.new("RGB", (combined_width, combined_height), (255, 255, 255))
    draw = ImageDraw.Draw(combined_img)

    # Step 2: Draw the query text and the image_name at the top
    # We will center them or simply draw them line by line
    # You can customize the text layout as you like

    # Draw the "query_text" on the first line
    draw.text((10, 10), f"Query: {query_text}", fill=(0,0,0), font=FONT)

    # Draw the "image_name" on the second line
    draw.text((10, 40), f"Image Name: {image_name}", fill=(0,0,0), font=FONT)

    # Step 3: Paste the three images side-by-side
    # We'll place them with some spacing in x-direction
    current_x = 0
    images_and_labels = [
        (img_lisat, label_lisat),
        (img_lisa,  label_lisa),
        (img_gt,    label_gt)
    ]

    for img, label in images_and_labels:
        # Paste image onto the combined image
        combined_img.paste(img, (current_x, padding_top))

        # Draw the label below the image
        label_x = current_x + 10  # some left offset
        label_y = padding_top + img.height + 5
        draw.text((label_x, label_y), label, fill=(0,0,0), font=FONT)

        # Move current_x for the next image, add spacing
        current_x += img.width + spacing

    # Step 4: Save the final combined image
    combined_img.save(save_path)
    print(f"Saved combined image to {save_path}")


def main():
    # 1. Read the JSON file
    with open(JSON_PATH, "r") as f:
        data = json.load(f)

    # 2. For each item in the JSON, we build the relevant paths and combine
    for entry in data:
        # The raw image name (e.g. 'COCO_train_000000010576.jpg')
        image_name = entry["image"]

        # The base name without extension, so we can find the other images
        # e.g. "COCO_train_000000010576"
        base_name = os.path.splitext(image_name)[0]

        # The query text
        query_text = entry["query"]["query"]  # e.g. "Locate the truck in the image..."

        # Build the filenames for LISAt, LISA, GT
        # LISAt inference -> <base_name>_seg_rgb.png
        # LISA inference  -> <base_name>_seg_rgb.png
        # GT -> segmented_<base_name>.png

        lisat_path = os.path.join(LISAt_FOLDER, f"{base_name}_seg_rgb.png")
        lisa_path  = os.path.join(LISA_FOLDER,  f"{base_name}_seg_rgb.png")
        gt_path    = os.path.join(GT_FOLDER,    f"segmented_{base_name}.png")

        if not (os.path.exists(lisat_path) and os.path.exists(lisa_path) and os.path.exists(gt_path)):
            print(f"Skipping {image_name} because one or more files do not exist:")
            print(f"  LISAt: {lisat_path}")
            print(f"  LISA : {lisa_path}")
            print(f"  GT   : {gt_path}")
            continue

        # Build output save path
        # e.g. OUTPUT_DIR/COCO_train_000000010576_combined.png
        save_filename = f"{base_name}_combined.png"
        save_path = os.path.join(OUTPUT_DIR, save_filename)

        # 3. Combine them
        combine_images_and_text(
            query_text=query_text,
            image_name=image_name,
            path_lisat=lisat_path,
            path_lisa=lisa_path,
            path_gt=gt_path,
            save_path=save_path,
            label_lisat="LISAt",
            label_lisa="LISA",
            label_gt="Ground Truth"
        )


if __name__ == "__main__":
    main()

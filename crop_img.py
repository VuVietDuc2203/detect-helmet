import os
import glob
from PIL import Image


def crop_images_by_class_id(input_dir, output_dir, target_class_ids):
    """
    Crops images based on specific class IDs in corresponding label text files.

    Args:
        input_dir (str): Directory containing images and label files
        output_dir (str): Directory to save cropped images
        target_class_ids (list): List of class IDs to crop (e.g., [0, 1])
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get all image files in the input directory
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))

    print(f"Found {len(image_files)} image files.")

    for image_path in image_files:
        # Get corresponding text file path
        txt_path = os.path.splitext(image_path)[0] + '.txt'

        # Check if text file exists
        if not os.path.exists(txt_path):
            print(f"Warning: No text file found for {image_path}. Skipping.")
            continue

        try:
            # Open the image
            img = Image.open(image_path)
            img_width, img_height = img.size

            # Read the label file
            with open(txt_path, 'r') as f:
                lines = f.readlines()

            # Process each bounding box in the label file
            for i, line in enumerate(lines):
                if not line.strip():  # Skip empty lines
                    continue

                parts = line.strip().split()
                if len(parts) < 5:
                    print(f"Warning: Invalid format in {txt_path}, line {i + 1}. Skipping.")
                    continue

                # Parse the label
                class_id = int(parts[0])
                # Skip if class ID is not in our target list
                if class_id not in target_class_ids:
                    continue

                x_center = float(parts[1])
                y_center = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])

                # Convert normalized coordinates to pixel coordinates
                x1 = int((x_center - width / 2) * img_width)
                y1 = int((y_center - height / 2) * img_height)
                x2 = int((x_center + width / 2) * img_width)
                y2 = int((y_center + height / 2) * img_height)

                # Ensure coordinates are within image boundaries
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(img_width, x2)
                y2 = min(img_height, y2)

                # Skip if the bounding box is invalid
                if x1 >= x2 or y1 >= y2:
                    print(f"Warning: Invalid bounding box in {txt_path}, line {i + 1}. Skipping.")
                    continue

                # Crop the image
                cropped_img = img.crop((x1, y1, x2, y2))

                # Generate output filename
                base_name = os.path.basename(image_path)
                name_without_ext = os.path.splitext(base_name)[0]
                output_path = os.path.join(output_dir, f"{name_without_ext}_{class_id}_{i}.jpg")

                # Save the cropped image
                cropped_img.save(output_path)
                print(f"Saved cropped image: {output_path}")

        except Exception as e:
            print(f"Error processing {image_path}: {str(e)}")


if __name__ == "__main__":
    # Configuration - modify these values as needed
    INPUT_DIR = "frame_tests"  # Directory containing images and labels
    OUTPUT_DIR = "helmet_test"  # Directory to save cropped images
    TARGET_CLASS_IDS = [2]  # List of class IDs to crop (e.g., [0, 1] for classes 0 and 1)

    # Run the cropping function
    crop_images_by_class_id(INPUT_DIR, OUTPUT_DIR, TARGET_CLASS_IDS)
    print("Image cropping completed!")
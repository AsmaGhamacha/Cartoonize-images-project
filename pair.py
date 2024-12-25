import cv2
import os

# Directories
real_dir = "./data/data_split/test"  # Folder with real images
cartoonized_dir = "./data/test_cartoonized"  # Folder with cartoonized images for validation
output_dir = "data/paired_images_test"  # Folder for concatenated testing images
os.makedirs(output_dir, exist_ok=True)

# Process each image
for filename in os.listdir(real_dir):
    real_path = os.path.join(real_dir, filename)
    cartoon_path = os.path.join(cartoonized_dir, filename)

    # Ensure both images exist
    if os.path.exists(real_path) and os.path.exists(cartoon_path):
        real_img = cv2.imread(real_path)
        cartoon_img = cv2.imread(cartoon_path)

        # Check if images were loaded properly
        if real_img is None:
            print(f"Error: Could not read real image: {real_path}")
            continue
        if cartoon_img is None:
            print(f"Error: Could not read cartoonized image: {cartoon_path}")
            continue

        # Resize images to the same dimensions if necessary
        if real_img.shape != cartoon_img.shape:
            cartoon_img = cv2.resize(cartoon_img, (real_img.shape[1], real_img.shape[0]))

        # Concatenate images side by side
        paired_img = cv2.hconcat([real_img, cartoon_img])

        # Save the concatenated image
        output_path = os.path.join(output_dir, filename)
        cv2.imwrite(output_path, paired_img)
        print(f"Saved paired image: {output_path}")
    else:
        print(f"Mismatch: Real image or cartoonized image not found for {filename}")

print(f"Paired images saved in {output_dir}")

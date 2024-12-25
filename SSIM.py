from skimage.metrics import structural_similarity as ssim
from PIL import Image
import numpy as np
import os

def calculate_ssim_for_pair(output_path, target_path):
    """
    Calculate SSIM between an output and target image.
    :param output_path: Path to the generated (output) image.
    :param target_path: Path to the ground truth (target) image.
    :return: SSIM value.
    """
    # Load images
    output_image = Image.open(output_path).convert("RGB")
    target_image = Image.open(target_path).convert("RGB")

    # Resize images to 256x256 (if necessary)
    output_image = output_image.resize((256, 256))
    target_image = target_image.resize((256, 256))

    # Convert to numpy arrays
    output_array = np.array(output_image)
    target_array = np.array(target_image)

    # Compute SSIM
    return ssim(output_array, target_array, multichannel=True, win_size=3)

if __name__ == "__main__":
    # Paths and initialization
    validation_folder = "./validation_outputs_seperated"  # Update with your folder path
    total_ssim = 0
    count = 0

    # Loop through all output-target pairs
    for file_name in os.listdir(validation_folder):
        if file_name.startswith("output_") and file_name.endswith(".png"):
            # Find corresponding target image
            target_name = file_name.replace("output_", "target_")
            output_path = os.path.join(validation_folder, file_name)
            target_path = os.path.join(validation_folder, target_name)

            if os.path.exists(target_path):
                # Calculate SSIM for the pair
                ssim_value = calculate_ssim_for_pair(output_path, target_path)
                print(f"SSIM for {file_name}: {ssim_value:.4f}")
                total_ssim += ssim_value
                count += 1
            else:
                print(f"Target image {target_name} not found for {file_name}")

    # Compute average SSIM
    if count > 0:
        average_ssim = total_ssim / count
        print(f"Average SSIM for all pairs: {average_ssim:.4f}")
    else:
        print("No valid image pairs found for SSIM calculation.")

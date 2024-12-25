import cv2
import os
import numpy as np

def cartoonize_image(image_path, save_path):
    """
    Cartoonizes an input image and saves the output.
    Combines edge detection with quantized colors.
    """
    # Step 1: Load the image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to read image {image_path}")
        return

    # Step 2: Downscale the image to reduce processing time
    img_small = cv2.pyrDown(img)

    # Step 3: Apply bilateral filter to smooth the image
    smoothed = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 4: Convert to grayscale and use Canny for edge detection
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Step 5: Perform k-means clustering for color quantization
    img_small_flat = smoothed.reshape(-1, 3).astype(np.float32)
    num_clusters = 6  # Reduce colors to 6 clusters
    _, labels, centers = cv2.kmeans(
        img_small_flat, num_clusters, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    quantized = centers[labels.flatten()].reshape(img_small.shape).astype(np.uint8)

    # Step 6: Resize edges to match quantized image size
    edges_resized = cv2.resize(edges, (quantized.shape[1], quantized.shape[0]))

    # Step 7: Create a mask for edges
    edges_colored = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    edges_inverted = cv2.bitwise_not(edges_colored)

    # Step 8: Combine edges with the quantized image
    cartoon = cv2.bitwise_and(quantized, edges_inverted)

    # Step 9: Upscale back to original size
    cartoon_large = cv2.pyrUp(cartoon)

    # Save the cartoonized image
    cv2.imwrite(save_path, cartoon_large)

def cartoonize_dataset(input_dir, output_dir):
    """
    Applies cartoonization to all images in the input directory.
    :param input_dir: Path to the folder with input images.
    :param output_dir: Path to save cartoonized images.
    """
    os.makedirs(output_dir, exist_ok=True)

    # Loop through all images in the input directory
    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Apply cartoonization
        cartoonize_image(input_path, output_path)

    print(f"Cartoonized images saved in {output_dir}")

# Example usage
if __name__ == "__main__":
    input_dir = "./data/data_split/test"  # Folder with test images
    output_dir = "./data/test_cartoonized"  # Folder for cartoonized images for validation
    cartoonize_dataset(input_dir, output_dir)

import os
import cv2
import torch
import numpy as np

def cartoonize_image(image_tensor, device):
    """
    Cartoonizes an input image tensor using GPU and returns the output tensor.
    Combines edge detection with quantized colors.
    """
    # Move the image tensor to GPU
    image_tensor = image_tensor.to(device)
    image_tensor = image_tensor.permute(0, 3, 1, 2)  # Convert to (N, C, H, W)

    # Convert tensor to numpy array
    img = image_tensor[0].permute(1, 2, 0).cpu().numpy().astype(np.uint8)

    # Step 1: Downscale the image to reduce processing time
    img_small = cv2.pyrDown(img)

    # Step 2: Apply bilateral filter to smooth the image
    smoothed = cv2.bilateralFilter(img_small, d=9, sigmaColor=75, sigmaSpace=75)

    # Step 3: Convert to grayscale and use Canny for edge detection
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, threshold1=50, threshold2=150)

    # Step 4: Perform k-means clustering for color quantization
    img_small_flat = smoothed.reshape(-1, 3).astype(np.float32)
    num_clusters = 6  # Reduce colors to 6 clusters
    _, labels, centers = cv2.kmeans(
        img_small_flat, num_clusters, None,
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0),
        10, cv2.KMEANS_RANDOM_CENTERS
    )
    quantized = centers[labels.flatten()].reshape(img_small.shape).astype(np.uint8)

    # Step 5: Resize edges to match quantized image size
    edges_resized = cv2.resize(edges, (quantized.shape[1], quantized.shape[0]))

    # Step 6: Create a mask for edges
    edges_colored = cv2.cvtColor(edges_resized, cv2.COLOR_GRAY2BGR)
    edges_inverted = cv2.bitwise_not(edges_colored)

    # Step 7: Combine edges with the quantized image
    cartoon = cv2.bitwise_and(quantized, edges_inverted)

    # Step 8: Upscale back to original size
    cartoon_large = cv2.pyrUp(cartoon)

    return cartoon_large

def cartoonize_dataset(input_dir, output_dir, limit=500):
    """
    Applies cartoonization to a limited number of images in the input directory using GPU.
    :param input_dir: Path to the folder with input images.
    :param output_dir: Path to save cartoonized images.
    :param limit: Maximum number of images to process.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of image files in the input directory
    image_files = [f for f in os.listdir(input_dir) if f.endswith(('png', 'jpg', 'jpeg'))]

    # Limit to the specified number of images
    image_files = image_files[:limit]

    # Loop through the limited number of images
    for idx, filename in enumerate(image_files):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        # Read and convert image to tensor
        img = cv2.imread(input_path)
        if img is None:
            print(f"Error: Unable to read image {input_path}")
            continue

        img_tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0) / 255.0  # Normalize

        # Apply cartoonization
        cartoon = cartoonize_image(img_tensor, device)

        # Save the cartoonized image
        cv2.imwrite(output_path, cartoon)
        print(f"Processed {idx + 1}/{len(image_files)}: {filename}")

    print(f"Cartoonized images saved in {output_dir}")

# Example usage
if __name__ == "__main__":
    input_dir = "./data/data_split/test"  # Folder with test images
    output_dir = "./data/test_cartoonized"  # Folder for cartoonized images to validate
    cartoonize_dataset(input_dir, output_dir, limit=500)

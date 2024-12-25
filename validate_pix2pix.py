import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image, make_grid
from PIL import Image
import os
from pix2pix_model import GeneratorUNet  # Ensure to import the correct model

# Dataset Class for Validation
class TestDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Initialize the dataset with paired test images.
        :param root_dir: Path to the folder containing paired test images.
        :param transform: Transformations to apply to the images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = os.listdir(root_dir)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        img = Image.open(img_path).convert("RGB")

        # Split concatenated image into input (real) and target (cartoonized)
        w, h = img.size
        input_image = img.crop((0, 0, w // 2, h))
        target_image = img.crop((w // 2, 0, w, h))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return input_image, target_image

# Validation Function
def validate(generator, test_loader, device, output_dir):
    """
    Validate the generator on the test dataset and save results.
    :param generator: Trained generator model.
    :param test_loader: DataLoader for the test dataset.
    :param device: 'cuda' or 'cpu' for evaluation.
    :param output_dir: Directory to save validation outputs.
    """
    generator.eval()  # Set generator to evaluation mode
    os.makedirs(output_dir, exist_ok=True)

    with torch.no_grad():
        for i, (input_image, target_image) in enumerate(test_loader):
            input_image, target_image = input_image.to(device), target_image.to(device)

            # Generate cartoonized image
            fake_image = generator(input_image)

            # De-normalize images for saving
            fake_image = (fake_image * 0.5 + 0.5).clamp(0, 1)
            input_image = (input_image * 0.5 + 0.5).clamp(0, 1)
            target_image = (target_image * 0.5 + 0.5).clamp(0, 1)

            # Concatenate input, output, and target for better visualization
            combined_image = torch.cat((input_image, fake_image, target_image), dim=3)  # Concatenate along width

            # Save the combined image
            save_image(combined_image, os.path.join(output_dir, f"combined_{i+1}.png"))

            print(f"Saved validation image {i+1}/{len(test_loader)} to {output_dir}")

if __name__ == "__main__":
    # Paths and Parameters
    test_dir = "./data/paired_images_test"  # Path to the test dataset
    output_dir = "./validation_outputs"  # Directory to save results
    generator_path = "generator_final.pth"  # Path to the trained generator
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Data Transformations
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Ensure input size matches training
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])

    # Load Test Dataset
    test_dataset = TestDataset(test_dir, transform=transform)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load the trained generator
    generator = GeneratorUNet().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    print("Trained generator loaded.")

    # Run Validation
    validate(generator, test_loader, device, output_dir)
    print(f"Validation completed. Results saved in {output_dir}.")
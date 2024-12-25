import os
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class PairedDataset(Dataset):
    def __init__(self, root_dir, target_size=(256, 256)):
        self.root_dir = root_dir
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.jpg')]
        self.transform = transforms.Compose([
            transforms.Resize(target_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(image_path).convert("RGB")

        # Split concatenated image into two halves
        w, h = image.size
        input_image = image.crop((0, 0, w // 2, h))  # Left half
        target_image = image.crop((w // 2, 0, w, h))  # Right half

        input_image = self.transform(input_image)
        target_image = self.transform(target_image)

        return input_image, target_image

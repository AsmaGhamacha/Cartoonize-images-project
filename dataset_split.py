import os
import random
import shutil

# Define paths
source_dir = "./data/color"  # Path to the dataset
train_dir = "./data/data_split/train"  # Train directory
test_dir = "./data/data_split/test"  # Test directory

# Create directories
os.makedirs(train_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# List all files
all_images = [f for f in os.listdir(source_dir) if f.endswith(('.jpg', '.png', '.jpeg'))]

# Shuffle and split
random.shuffle(all_images)
split_point = int(len(all_images) * 0.9)
train_images = all_images[:split_point]
test_images = all_images[split_point:]

# Move files
for img in train_images:
    shutil.copy(os.path.join(source_dir, img), os.path.join(train_dir, img))

for img in test_images:
    shutil.copy(os.path.join(source_dir, img), os.path.join(test_dir, img))

print(f"Training set: {len(train_images)} images")
print(f"Test set: {len(test_images)} images")

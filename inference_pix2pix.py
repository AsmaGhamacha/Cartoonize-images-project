import torch
from torchvision import transforms
from PIL import Image
from pix2pix_model import GeneratorUNet

def preprocess_image(image_path):
    """
    Preprocess the input image for the model while keeping track of the original size.
    Args:
        image_path (str): Path to the input image.
    """
    image = Image.open(image_path).convert("RGB")
    original_size = image.size  # Save original size (width, height)

    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize to 256x256 for the model
        transforms.ToTensor(),  # Convert to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
    ])
    return transform(image).unsqueeze(0), original_size  # Add batch dimension

def postprocess_image(tensor, original_size):
    """
    Postprocess the output tensor from the model into a PIL Image resized back to the original size.
    Args:
        tensor (torch.Tensor): Output tensor from the model.
        original_size (tuple): Original size (width, height) of the input image.
    """
    tensor = (tensor.squeeze(0).detach().cpu() + 1) / 2  # Denormalize to [0, 1]
    output_image = transforms.ToPILImage()(tensor)
    return output_image.resize(original_size, Image.BICUBIC)  # Resize back to original size

def test_model(generator_path, image_path, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load the trained generator
    generator = GeneratorUNet().to(device)
    generator.load_state_dict(torch.load(generator_path, map_location=device))
    generator.eval()

    # Preprocess the input image
    input_tensor, original_size = preprocess_image(image_path)
    input_tensor = input_tensor.to(device)

    # Generate cartoonized image
    with torch.no_grad():
        output_tensor = generator(input_tensor)

    # Postprocess and save the output image
    output_image = postprocess_image(output_tensor, original_size)
    output_image.save(output_path)
    print(f"Cartoonized image saved to {output_path}")

if __name__ == "__main__":
    # Example usage
    generator_path = "generator_final.pth"  # Path  trained generator model
    image_path = "./test_images/Eiffel_tower.jpg"  # Path local image
    output_path = "./test_images/tower.jpg"  # save the cartoonized image

    test_model(generator_path, image_path, output_path)

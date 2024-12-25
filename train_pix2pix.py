import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pix2pix_model import GeneratorUNet, Discriminator
from prepare_data import PairedDataset
import time  # For measuring execution time

# Loss functions
bce_loss = nn.BCELoss()  # Binary Cross-Entropy for discriminator
l1_loss = nn.L1Loss()    # L1 loss for reconstruction

def train_pix2pix(num_epochs=50, batch_size=16, lambda_recon=100):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Initialize models
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    # Optimizers
    g_optimizer = optim.Adam(generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    d_optimizer = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Load dataset
    dataset = PairedDataset("./data/paired_images")
    dataset = torch.utils.data.Subset(dataset, range(3000))  # Train with 3000 pairs
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # Training loop
    for epoch in range(num_epochs):
        epoch_start_time = time.time()  # Start timing the epoch
        for batch_idx, (input_image, target_image) in enumerate(dataloader):
            input_image = input_image.to(device)
            target_image = target_image.to(device)

            # Train Discriminator
            fake_image = generator(input_image)
            real_output = discriminator(input_image, target_image)
            fake_output = discriminator(input_image, fake_image.detach())

            d_loss_real = bce_loss(real_output, torch.ones_like(real_output))
            d_loss_fake = bce_loss(fake_output, torch.zeros_like(fake_output))
            d_loss = (d_loss_real + d_loss_fake) / 2

            d_optimizer.zero_grad()
            d_loss.backward()
            d_optimizer.step()

            # Train Generator
            fake_output = discriminator(input_image, fake_image)
            g_adv_loss = bce_loss(fake_output, torch.ones_like(fake_output))
            g_recon_loss = l1_loss(fake_image, target_image) * lambda_recon
            g_loss = g_adv_loss + g_recon_loss

            g_optimizer.zero_grad()
            g_loss.backward()
            g_optimizer.step()

        # Measure epoch time
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time

        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"D Loss: {d_loss.item():.4f}, G Loss: {g_loss.item():.4f}, "
              f"Time: {epoch_duration:.2f} seconds")

        # Save models every 5 epochs
        if (epoch + 1) % 5 == 0:
            torch.save(generator.state_dict(), f"generator_epoch_{epoch+1}.pth")
            torch.save(discriminator.state_dict(), f"discriminator_epoch_{epoch+1}.pth")

    # Save final models
    torch.save(generator.state_dict(), "generator_final.pth")
    torch.save(discriminator.state_dict(), "discriminator_final.pth")
    print("Training complete. Models saved.")

if __name__ == "__main__":
    train_pix2pix()

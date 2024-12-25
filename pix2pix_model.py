import torch
import torch.nn as nn

# Generator (U-Net with skip connections)
class GeneratorUNet(nn.Module):
    def __init__(self):
        super(GeneratorUNet, self).__init__()

        def downsample(in_channels, out_channels, use_batchnorm=True):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
            ]
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)

        def upsample(in_channels, out_channels, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
            ]
            if dropout:
                layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder (Removed one downsampling block)
        self.down1 = downsample(3, 64, use_batchnorm=False)
        self.down2 = downsample(64, 128)
        self.down3 = downsample(128, 256)
        self.down4 = downsample(256, 512)
        self.down5 = downsample(512, 512)
        self.down6 = downsample(512, 512)

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1, bias=False),
            nn.ReLU(inplace=True),
        )

        # Decoder
        self.up1 = upsample(512, 512, dropout=True)
        self.up2 = upsample(1024, 512, dropout=True)
        self.up3 = upsample(1024, 512, dropout=True)
        self.up4 = upsample(1024, 256)
        self.up5 = upsample(512, 128)
        self.up6 = upsample(256, 64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, 3, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d6 = self.down6(d5)
        b = self.bottleneck(d6)
        u1 = self.up1(b)
        u2 = self.up2(torch.cat([u1, d6], dim=1))
        u3 = self.up3(torch.cat([u2, d5], dim=1))
        u4 = self.up4(torch.cat([u3, d4], dim=1))
        u5 = self.up5(torch.cat([u4, d3], dim=1))
        u6 = self.up6(torch.cat([u5, d2], dim=1))
        return self.final(torch.cat([u6, d1], dim=1))

# Discriminator (PatchGAN)
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        def block(in_channels, out_channels, stride):
            layers = [
                nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, inplace=True),
            ]
            return nn.Sequential(*layers)

        self.model = nn.Sequential(
            block(6, 64, stride=2),  # Input: Concatenated real and cartoonized images
            block(64, 128, stride=2),
            block(128, 256, stride=2),
            block(256, 512, stride=1),  # Keep stride=1 for PatchGAN
            nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=1),  # Patch-level decisions
            nn.Sigmoid(),
        )

    def forward(self, input_image, target_image):
        x = torch.cat([input_image, target_image], dim=1)  # Concatenate along channel dimension
        return self.model(x)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    generator = GeneratorUNet().to(device)
    discriminator = Discriminator().to(device)

    print(f"Generator initialized: {generator}")
    print(f"Discriminator initialized: {discriminator}")
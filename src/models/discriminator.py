import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, use_spectral_norm=True):
        super().__init__()
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        # Apply spectral normalization for training stability
        self.conv = spectral_norm(conv) if use_spectral_norm else conv
        self.bn = nn.InstanceNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        return out

class Discriminator(nn.Module):
    def __init__(self, input_channels=2, output_channels=3, features=[64, 128, 256, 512]):
        """
        PatchGAN Discriminator for Pix2Pix.

        Args:
            input_channels: Number of channels in input (segmentation), default 2 (buildings + roads)
            output_channels: Number of channels in output (RGB image), default 3
            features: Feature dimensions for each layer
        """
        super().__init__()
        # Apply spectral norm to initial layer for stability
        initial_conv = spectral_norm(
            nn.Conv2d(input_channels + output_channels, features[0], kernel_size=4, stride=2, padding=1)
        )
        self.initial = nn.Sequential(
            initial_conv,
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvBlock(in_channels, feature, kernel_size=4, stride=1 if feature == features[-1] else 2, padding=1)
            )
            in_channels = feature

        # Apply spectral norm to final output layer
        layers.append(
            spectral_norm(nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1))
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)
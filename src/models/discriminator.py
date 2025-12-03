import torch
import torch.nn as nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
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
        self.initial = nn.Sequential(
            nn.Conv2d(input_channels + output_channels, features[0], kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )

        layers = []
        in_channels = features[0]
        for feature in features[1:]:
            layers.append(
                ConvBlock(in_channels, feature, kernel_size=4, stride=1 if feature == features[-1] else 2, padding=1)
            )
            in_channels = feature

        layers.append(
            nn.Conv2d(in_channels, 1, kernel_size=4, stride=1, padding=1)
        )

        self.model = nn.Sequential(*layers)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.initial(x)
        return self.model(x)
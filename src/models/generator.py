import torch
from torch import nn

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_dropout=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=2, padding=1, use_dropout=True):
        super().__init__()
        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.use_dropout = use_dropout
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        if self.use_dropout:
            x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self, in_channels=2, out_channels=3, features=64):
        """
        U-Net Generator for Pix2Pix.

        Args:
            in_channels: Number of input channels (segmentation), default 2 (buildings + roads)
            out_channels: Number of output channels (RGB image), default 3
            features: Base number of feature channels, default 64
        """
        super().__init__()
        self.initial_down = nn.Sequential(
            nn.Conv2d(in_channels, features, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.down1 = DownBlock(features, features*2, use_dropout=False)
        self.down2 = DownBlock(features*2, features*4, use_dropout=False)
        self.down3 = DownBlock(features*4, features*8, use_dropout=False)
        self.down4 = DownBlock(features*8, features*8, use_dropout=False)
        self.down5 = DownBlock(features*8, features*8, use_dropout=False)
        self.down6 = DownBlock(features*8, features*8, use_dropout=False)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(features*8, features*8, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True)
        )

        self.up1 = UpBlock(features*8, features*8, use_dropout=True)
        self.up2 = UpBlock(features*8*2, features*8, use_dropout=True)
        self.up3 = UpBlock(features*8*2, features*8, use_dropout=True)
        self.up4 = UpBlock(features*8*2, features*8, use_dropout=False)
        self.up5 = UpBlock(features*8*2, features*4, use_dropout=False)
        self.up6 = UpBlock(features*4*2, features*2, use_dropout=False)
        self.up7 = UpBlock(features*2*2, features, use_dropout=False)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(features*2, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.initial_down(x)
        d2 = self.down1(d1)
        d3 = self.down2(d2)
        d4 = self.down3(d3)
        d5 = self.down4(d4)
        d6 = self.down5(d5)
        d7 = self.down6(d6)
        bottleneck = self.bottleneck(d7)
        up1 = self.up1(bottleneck)
        up2 = self.up2(torch.cat((up1, d7), dim=1))
        up3 = self.up3(torch.cat((up2, d6), dim=1))
        up4 = self.up4(torch.cat((up3, d5), dim=1))
        up5 = self.up5(torch.cat((up4, d4), dim=1))
        up6 = self.up6(torch.cat((up5, d3), dim=1))
        up7 = self.up7(torch.cat((up6, d2), dim=1))

        return self.final(torch.cat((up7, d1), dim=1))

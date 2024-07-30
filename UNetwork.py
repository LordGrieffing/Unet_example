import torch
import torch.nn as nn

from ULayers import DoubleConv, DownSample, UpSample

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes):
        super().__init__()

        # Left side of the U-shape
        self.down_conv_1 = DownSample(in_channels, 64)
        self.down_conv_2 = DownSample(64, 128)
        self.down_conv_3 = DownSample(128, 256)
        self.down_conv_4 = DownSample(256, 512)

        # Bottle neck of the U-shape
        self.bottle_neck = DoubleConv(512, 1024)

        # Right side of the U-shape
        self.up_conv_1 = UpSample(1024, 512)
        self.up_conv_2 = UpSample(512, 256)
        self.up_conv_3 = UpSample(256, 128)
        self.up_conv_4 = UpSample(128, 64)

        # Return the output to original input channels
        self.out = nn.Conv2d(in_channels=64, out_channels=num_classes, kernel_size=1)

    def forward(self, x):
        down_1, p1 = self.down_conv_1(x)
import torch
from torch import nn
from torch.nn import functional as F


class UnetGenerator(nn.Module):
    """Unet-like Encoder-Decoder model"""

    def __init__(self):
        super().__init__()

        def conv_block(in_ch, out_ch):
            return nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
            )

        # Encoder
        self.enc1 = conv_block(1, 32)
        self.pool1 = nn.MaxPool2d(2)  # 512x365 -> 256x182
        self.enc2 = conv_block(32, 64)
        self.pool2 = nn.MaxPool2d(2)  # 256x182 -> 128x91
        self.enc3 = conv_block(64, 128)
        self.pool3 = nn.MaxPool2d(2)  # 128x91 -> 64x45
        self.enc4 = conv_block(128, 256)

        # Bottleneck
        self.bottleneck = conv_block(256, 512)

        # Decoder
        self.up3 = nn.ConvTranspose2d(
            512, 256, kernel_size=2, stride=2
        )  # 64x45 -> 128x90(91)
        self.dec3 = conv_block(256 + 128, 256)

        self.up2 = nn.ConvTranspose2d(
            256, 128, kernel_size=2, stride=2
        )  # 128x91 -> 256x182
        self.dec2 = conv_block(128 + 64, 128)

        self.up1 = nn.ConvTranspose2d(
            128, 64, kernel_size=2, stride=2
        )  # 256x182 -> 512x364(365)
        self.dec1 = conv_block(64 + 32, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)

        self.resize = nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        b = self.bottleneck(e4)

        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        # d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return self.resize(out)


class BasicBlock(nn.Module):
    """Basic block"""
    def __init__(self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True):
        super().__init__()
        self.conv = nn.Conv2d(inplanes, outplanes, kernel_size, stride, padding)
        self.isn = None
        if norm:
            self.isn = nn.InstanceNorm2d(outplanes)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        fx = self.conv(x)

        if self.isn is not None:
            fx = self.isn(fx)

        fx = self.lrelu(fx)
        return fx


class ConditionalDiscriminator(nn.Module):
    """Conditional Discriminator"""
    def __init__(self,):
        super().__init__()
        self.block1 = BasicBlock(2, 64, norm=False)  # 256x256 -> 128x128
        self.block2 = BasicBlock(64, 128)  # 128x128 -> 64x64
        self.block3 = BasicBlock(128, 128)  # 64x64 -> 32x32
        self.block4 = BasicBlock(128, 128)  # 32x32 -> 16x16
        self.block5 = BasicBlock(128, 256)  # 16x16 -> 8x8
        self.block6 = BasicBlock(256, 1)  # 8x8 -> 4x4

        self.cond_resize = nn.AdaptiveAvgPool2d((256, 256))

    def forward(self, x, cond):
        cond = self.cond_resize(cond)  # Resize condition to match x's size
        x = torch.cat([x, cond], dim=1)

        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)
        fx = self.block6(fx)

        return fx

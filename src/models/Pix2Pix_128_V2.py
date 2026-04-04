import torch
from torch import nn
from torch.nn import functional as F


class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )

        self.shortcut = nn.Sequential()
        if in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


class UnetGenerator(nn.Module):
    """Unet-like Encoder-Decoder model with Residual Blocks"""

    def __init__(self):
        super().__init__()

        # ----- Residual Encoder -----
        self.enc1 = ResidualConvBlock(1, 128)
        self.pool1 = nn.MaxPool2d(2)  # 256x183 -> 128x91

        self.enc2 = ResidualConvBlock(128, 256)
        self.pool2 = nn.MaxPool2d(2)  # 128x91 -> 64x45

        self.enc3 = ResidualConvBlock(256, 512)
        self.pool3 = nn.MaxPool2d(2)  # 64x45 -> 32x22

        self.enc4 = ResidualConvBlock(512, 1024)

        # ----- Bottleneck -----
        self.bottleneck = ResidualConvBlock(1024, 1024)

        # ----- Decoder -----
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(1024, 512, kernel_size=3, padding=1),
        )
        self.dec3 = ResidualConvBlock(512 + 512, 512)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
        )
        self.dec2 = ResidualConvBlock(256 + 256, 256)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
        )
        self.dec1 = ResidualConvBlock(128 + 128, 128)

        self.final = nn.Conv2d(128, 1, kernel_size=1)
        self.resize = nn.AdaptiveAvgPool2d((128, 128))

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return self.resize(out)


class UnetGeneratorSmall(nn.Module):
    """Smaller Unet-like Encoder-Decoder model with Residual Blocks (half channels)"""

    def __init__(self):
        super().__init__()

        # ----- Residual Encoder -----
        self.enc1 = ResidualConvBlock(1, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.enc2 = ResidualConvBlock(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.enc3 = ResidualConvBlock(128, 256)
        self.pool3 = nn.MaxPool2d(2)

        self.enc4 = ResidualConvBlock(256, 512)

        # ----- Bottleneck -----
        self.bottleneck = ResidualConvBlock(512, 512)

        # ----- Decoder -----
        self.up3 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
        )
        self.dec3 = ResidualConvBlock(256 + 256, 256)

        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
        )
        self.dec2 = ResidualConvBlock(128 + 128, 128)

        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
        )
        self.dec1 = ResidualConvBlock(64 + 64, 64)

        self.final = nn.Conv2d(64, 1, kernel_size=1)
        self.resize = nn.AdaptiveAvgPool2d((128, 128))

    def forward(self, x):
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool1(e1))
        e3 = self.enc3(self.pool2(e2))
        e4 = self.enc4(self.pool3(e3))

        # Bottleneck
        b = self.bottleneck(e4)

        # Decoder
        d3 = self.up3(b)
        d3 = F.interpolate(d3, size=e3.shape[2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, e3], dim=1))

        d2 = self.up2(d3)
        d2 = F.interpolate(d2, size=e2.shape[2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))

        d1 = self.up1(d2)
        d1 = F.interpolate(d1, size=e1.shape[2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))

        out = self.final(d1)
        return self.resize(out)


class BasicBlock(nn.Module):
    """Basic block"""

    def __init__(
        self, inplanes, outplanes, kernel_size=4, stride=2, padding=1, norm=True
    ):
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

    def __init__(
        self,
    ):
        super().__init__()
        self.block1 = BasicBlock(2, 128, norm=False)  # 128x128 -> 64x64
        self.block2 = BasicBlock(128, 256)  # 64x64 -> 32x32
        self.block3 = BasicBlock(256, 512)  # 32x32 -> 16x16
        self.block4 = BasicBlock(512, 512)  # 16x16 -> 8x8
        self.block5 = BasicBlock(512, 1)  # 8x8 -> 4x4

        self.cond_resize = nn.AdaptiveAvgPool2d((128, 128))

    def forward(self, x, cond):
        cond = self.cond_resize(cond)  # Resize condition to match x's size
        x = torch.cat([x, cond], dim=1)

        fx = self.block1(x)
        fx = self.block2(fx)
        fx = self.block3(fx)
        fx = self.block4(fx)
        fx = self.block5(fx)

        return fx

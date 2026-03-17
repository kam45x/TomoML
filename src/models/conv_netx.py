#######################################################################################
# modified model of the authorship of Germer et al. presented in paper
# Limited-Angle Tomography Reconstruction via Deep End-To-End Learning on Synthetic Data
# https://arxiv.org/abs/2309.06948
#######################################################################################

from torch import nn


class Block(nn.Module):
    def __init__(self, dim, kernel_size=5, expansion=2):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = nn.Conv2d(dim, dim, kernel_size, 1, pad, padding_mode="zeros")
        self.bn = nn.BatchNorm2d(dim)
        hidden = int(expansion * dim)
        self.ln1 = nn.Conv2d(dim, hidden, 1, 1, 0)
        self.act = nn.GELU()
        self.ln2 = nn.Conv2d(hidden, dim, 1, 1, 0)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.ln1(out)
        out = self.act(out)
        out = self.ln2(out)
        return out + x


class ConvNetX(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()

        def blocks(dim, n):
            return [Block(dim) for _ in range(n)]

        # H: 512 -> 128 -> 32 -> 8
        # W: 256 -> 64  -> 16 -> 8
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_ch, 64, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), padding_mode="zeros"
            ),
            nn.BatchNorm2d(64),
            *blocks(64, n=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                64, 128, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), padding_mode="zeros"
            ),
            *blocks(128, n=3),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 256, kernel_size=(4, 3), stride=(4, 2), padding=(0, 1), padding_mode="zeros"
            ),
        )

        self.decoder = nn.Sequential(
            *blocks(256, n=6),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 2, 2, 0),  # 16
            *blocks(128, n=3),
            nn.ConvTranspose2d(128, 64, 2, 2, 0),  # 32
            *blocks(64, n=3),
            nn.ConvTranspose2d(64, 32, 2, 2, 0),  # 64
            *blocks(32, n=1),
            nn.ConvTranspose2d(32, 16, 2, 2, 0),  # 128
            *blocks(16, n=1),
            nn.ConvTranspose2d(16, 16, 2, 2, 0),  # 256
            *blocks(16, n=1),
            nn.Conv2d(16, 1, 3, 1, 1),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


class ConvNetX_128(nn.Module):
    def __init__(self, in_ch=1):
        super().__init__()

        def blocks(dim, n):
            return [Block(dim) for _ in range(n)]

        # H: 256 -> 64 -> 16 -> 4
        # W: 183 -> 46 -> 12 -> 4
        self.encoder = nn.Sequential(
            nn.Conv2d(
                in_ch, 64, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), padding_mode="zeros"
            ),
            nn.BatchNorm2d(64),
            *blocks(64, n=1),
            nn.BatchNorm2d(64),
            nn.Conv2d(
                64, 128, kernel_size=(4, 4), stride=(4, 4), padding=(1, 1), padding_mode="zeros"
            ),
            *blocks(128, n=3),
            nn.BatchNorm2d(128),
            nn.Conv2d(
                128, 256, kernel_size=(4, 3), stride=(4, 3), padding=(0, 0), padding_mode="zeros"
            ),
        )

        self.decoder = nn.Sequential(
            *blocks(256, n=6),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 2, 2, 0),  # 8
            *blocks(128, n=3),
            nn.ConvTranspose2d(128, 64, 2, 2, 0),  # 16
            *blocks(64, n=3),
            nn.ConvTranspose2d(64, 32, 2, 2, 0),  # 32
            *blocks(32, n=1),
            nn.ConvTranspose2d(32, 16, 2, 2, 0),  # 64
            *blocks(16, n=1),
            nn.ConvTranspose2d(16, 16, 2, 2, 0),  # 128
            *blocks(16, n=1),
            nn.Conv2d(16, 1, 3, 1, 1),
        )

    def forward(self, x):
        out = self.encoder(x)
        out = self.decoder(out)
        return out


if __name__ == "__main__":
    model = ConvNetX()
    print(model)

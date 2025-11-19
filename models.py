import torch
import torch.nn as nn

# U-Net Generator components

class UNetDown(nn.Module):
    def __init__(self, in_channels, out_channels, norm=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_channels, out_channels, 4, 2, 1, bias=not norm)
        ]
        if norm:
            layers.append(nn.InstanceNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2, True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class UNetUp(nn.Module):
    def __init__(self, in_channels, out_channels, dropout=False):
        super().__init__()
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(True)
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        self.block = nn.Sequential(*layers)

    def forward(self, x, skip):
        x = self.block(x)
        x = torch.cat([x, skip], dim=1)
        return x



# Pix2Pix U-Net Generator


class UNetGenerator(nn.Module):
    def __init__(self, input_nc=3, output_nc=3):
        super().__init__()

        # Downsampling
        self.d1 = UNetDown(input_nc,   64, norm=False)
        self.d2 = UNetDown(64,        128)
        self.d3 = UNetDown(128,       256)
        self.d4 = UNetDown(256,       512)
        self.d5 = UNetDown(512,       512)
        self.d6 = UNetDown(512,       512)
        self.d7 = UNetDown(512,       512)
        self.d8 = UNetDown(512,       512, norm=False)

        # Upsampling
        self.u1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(512),
            nn.ReLU(True),
            nn.Dropout(0.5)
        )
        self.u2 = UNetUp(512 + 512, 512, dropout=True)
        self.u3 = UNetUp(512 + 512, 512, dropout=True)
        self.u4 = UNetUp(512 + 512, 512)
        self.u5 = UNetUp(512 + 512, 256)
        self.u6 = UNetUp(256 + 256, 128)
        self.u7 = UNetUp(128 + 128,  64)

        self.final = nn.Sequential(
            nn.ConvTranspose2d(64 + 64, output_nc, 4, 2, 1),
            nn.Tanh()
        )

    def forward(self, x):
        d1 = self.d1(x)
        d2 = self.d2(d1)
        d3 = self.d3(d2)
        d4 = self.d4(d3)
        d5 = self.d5(d4)
        d6 = self.d6(d5)
        d7 = self.d7(d6)
        d8 = self.d8(d7)

        u1 = self.u1(d8)
        u1 = torch.cat([u1, d7], dim=1)

        u2 = self.u2(u1, d6)
        u3 = self.u3(u2, d5)
        u4 = self.u4(u3, d4)
        u5 = self.u5(u4, d3)
        u6 = self.u6(u5, d2)
        u7 = self.u7(u6, d1)

        return self.final(u7)

# PatchGAN Discriminator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if normalize:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return layers

        self.model = nn.Sequential(
            *block(6, 64, normalize=False),
            *block(64, 128),
            *block(128, 256),
            nn.Conv2d(256, 1, 4, 1, 1)
        )

    def forward(self, A, B):
        x = torch.cat([A, B], dim=1)
        return self.model(x)



    

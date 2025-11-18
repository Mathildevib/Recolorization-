import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.double_conv(x)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = DoubleConv(1, 32)
        self.pool1 = nn.MaxPool2d(2)

        self.conv2 = DoubleConv(32, 64)
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = DoubleConv(64, 128)
        self.pool3 = nn.MaxPool2d(2)

        self.conv4 = DoubleConv(128, 256)
        self.pool4 = nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(256, 512)

        self.transconv1 = nn.ConvTranspose2d(512, 256, 2, 2)
        self.concat_conv1 = DoubleConv(512, 256)

        self.transconv2 = nn.ConvTranspose2d(256, 128, 2, 2)
        self.concat_conv2 = DoubleConv(256, 128)

        self.transconv3 = nn.ConvTranspose2d(128, 64, 2, 2)
        self.concat_conv3 = DoubleConv(128, 64)

        self.transconv4 = nn.ConvTranspose2d(64, 32, 2, 2)
        self.concat_conv4 = DoubleConv(64, 32)

        self.final_conv = nn.Conv2d(32, 3, kernel_size=1)

    def forward(self, x):
        enc1 = self.conv1(x)
        p1 = self.pool1(enc1)

        enc2 = self.conv2(p1)
        p2 = self.pool2(enc2)

        enc3 = self.conv3(p2)
        p3 = self.pool3(enc3)

        enc4 = self.conv4(p3)
        p4 = self.pool4(enc4)

        b = self.bottleneck(p4)

        up1 = self.transconv1(b)
        dec1 = self.concat_conv1(torch.cat([up1, enc4], dim=1))

        up2 = self.transconv2(dec1)
        dec2 = self.concat_conv2(torch.cat([up2, enc3], dim=1))

        up3 = self.transconv3(dec2)
        dec3 = self.concat_conv3(torch.cat([up3, enc2], dim=1))

        up4 = self.transconv4(dec3)
        dec4 = self.concat_conv4(torch.cat([up4, enc1], dim=1))

        return self.final_conv(dec4)


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def conv_block(in_c, out_c, norm=True):
            layers = [nn.Conv2d(in_c, out_c, 4, 2, 1)]
            if norm:
                layers.append(nn.InstanceNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, True))
            return nn.Sequential(*layers)

    
        self.model = nn.Sequential(
            conv_block(4, 64, norm=False),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 1)
        )

    def forward(self, grey, color):
        x = torch.cat([grey, color], dim=1)
        return self.model(x)


    
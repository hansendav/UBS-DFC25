import torch
import torch.nn as nn
import torch.nn.functional as F
import sys 

sys.path.append('/home/gentleprotector/ubs_ws24/UBS-DFC25/codebase')
from location_encoder import wrap_encoder, Siren2d, Siren1d

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, dtype=torch.float64, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, dtype=torch.float64, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownBlock, self).__init__()
        self.maxpool_and_conv = nn.Sequential(
            nn.MaxPool2d(2), 
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_and_conv(x)


class UpBlock(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels):
        super(UpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(
            in_channels, in_channels // 2, kernel_size=2, stride=2, dtype=torch.float64
        )
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        # x1 is the output of the previous Up layer;
        # x2 is the skip connection from the last Downlayer with compatible
        # shape.
        x1 = self.up(x1)
        # input shape is (batch, channel, height, width)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(
            x1,
            [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2],
        )
        x = torch.cat([x2, x1], dim=1)  # Concat along the channel dim.
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        # Final classification to compute the per-pixel class logits using a
        # 1x1 convolution layer.
        # out_channels should match the number of possible classes.
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class UNet(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down1 = DownBlock(in_channels=64, out_channels=128)
        self.down2 = DownBlock(in_channels=128, out_channels=256)
        self.down3 = DownBlock(in_channels=256, out_channels=512)
        self.down4 = DownBlock(in_channels=512, out_channels=1024)
        self.up1 = UpBlock(in_channels=1024, out_channels=512)
        self.up2 = UpBlock(in_channels=512, out_channels=256)
        self.up3 = UpBlock(in_channels=256, out_channels=128)
        self.up4 = UpBlock(in_channels=128, out_channels=64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        """
        Parameters
        ==========
            x: Tensor - size (b,in_channels,h,w) input image
        Returns
        =======
            logits: Tensor - size (b,n_classes,h,w) output logits
        """
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        return logits

class UNetPlusLoc(nn.Module):
    def __init__(self, in_channels, n_classes, n_siren):
        super(UNetPlusLoc, self).__init__()
        self.in_channels = in_channels
        self.n_classes = n_classes

        self.inc = DoubleConv(in_channels=in_channels, out_channels=64)
        self.down1 = DownBlock(in_channels=64, out_channels=128)
        self.down2 = DownBlock(in_channels=128, out_channels=256)
        self.down3 = DownBlock(in_channels=256, out_channels=512)

        self.down4 = DownBlock(in_channels=512, out_channels=1024)
        self.up1 = UpBlock(in_channels=1024, out_channels=512)
        self.up2 = UpBlock(in_channels=512, out_channels=256)
        self.up3 = UpBlock(in_channels=256, out_channels=128)
        self.up4 = UpBlock(in_channels=128, out_channels=64)
        self.outc = OutConv(64, n_classes)

        self.pe = wrap_encoder()
        self.pos_nn = Siren1d()

        self.float()

    def forward(self, img_stack, coords): 
        
        # positional encoding
        pe = self.pe(coords)
        pos_nn_en = self.pos_nn(pe)
        
        # expand the positional encoding to match the input image size
        pos_nn_en = pos_nn_en.unsqueeze(2).unsqueeze(3).expand(-1, -1, img_stack.shape[2], img_stack.shape[3])

        # concatenate the positional encoding with the input image
        # could be also addition or multiplication?
        x = torch.cat((img_stack, pos_nn_en), dim=1) # concatenate input image with positional encoding along the channel dimension

        # unet forward pass
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        logits = self.outc(x)
        
        return logits
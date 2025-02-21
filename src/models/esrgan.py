import torch
import torch.nn as nn

class ResidualDenseBlock(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(nf, gc, 3, padding=1)
        self.conv2 = nn.Conv2d(nf + gc, gc, 3, padding=1)
        self.conv3 = nn.Conv2d(nf + 2 * gc, gc, 3, padding=1)
        self.conv4 = nn.Conv2d(nf + 3 * gc, gc, 3, padding=1)
        self.conv5 = nn.Conv2d(nf + 4 * gc, nf, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x

class RRDB(nn.Module):
    def __init__(self, nf=64, gc=32):
        super(RRDB, self).__init__()
        self.rdb1 = ResidualDenseBlock(nf, gc)
        self.rdb2 = ResidualDenseBlock(nf, gc)
        self.rdb3 = ResidualDenseBlock(nf, gc)

    def forward(self, x):
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x

class Generator(nn.Module):
    def __init__(self, num_rrdb=16):
        super(Generator, self).__init__()
        
        self.conv_first = nn.Conv2d(1, 64, 3, padding=1)
        self.body = nn.Sequential(*[RRDB() for _ in range(num_rrdb)])
        self.conv_body = nn.Conv2d(64, 64, 3, padding=1)
        
        self.upconv1 = nn.Conv2d(64, 64 * 4, 3, padding=1)
        self.upconv2 = nn.Conv2d(64, 1, 3, padding=1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.pixel_shuffle = nn.PixelShuffle(2)

    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat
        
        out = self.lrelu(self.pixel_shuffle(self.upconv1(feat)))
        out = self.upconv2(out)
        return torch.tanh(out)

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        def block(in_channels, out_channels, stride):
            layers = [
                nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=True),
                nn.BatchNorm2d(out_channels),
                nn.LeakyReLU(0.2, True)
            ]
            return layers

        self.model = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.LeakyReLU(0.2, True),
            *block(64, 64, 2),
            *block(64, 128, 1),
            *block(128, 128, 2),
            *block(128, 256, 1),
            *block(256, 256, 2),
            *block(256, 512, 1),
            *block(512, 512, 2),
            nn.Conv2d(512, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.model(x)

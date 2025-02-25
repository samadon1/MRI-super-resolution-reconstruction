import torch
import torch.nn as nn

class MobileBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor=6):
        super(MobileBlock, self).__init__()
        hidden_dim = in_channels * expansion_factor
        self.use_res_connect = in_channels == out_channels
        
        self.conv = nn.Sequential(
            # Pointwise expansion
            nn.Conv2d(in_channels, hidden_dim, 1, 1, 0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Depthwise
            nn.Conv2d(hidden_dim, hidden_dim, 3, 1, 1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True),
            # Pointwise linear projection
            nn.Conv2d(hidden_dim, out_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

class MobileESRGAN(nn.Module):
    def __init__(self, num_blocks=8, channels=32):
        super(MobileESRGAN, self).__init__()
        
        # Initial convolution
        self.conv_first = nn.Conv2d(1, channels, 3, 1, 1)
        
        # Mobile blocks
        body = []
        for _ in range(num_blocks):
            body.append(MobileBlock(channels, channels))
        self.body = nn.Sequential(*body)
        
        # Final processing before upsampling
        self.conv_body = nn.Conv2d(channels, channels, 3, 1, 1)
        
        # Upsampling
        self.upconv = nn.Conv2d(channels, 4 * channels, 3, 1, 1)
        self.pixel_shuffle = nn.PixelShuffle(2)
        self.conv_hr = nn.Conv2d(channels, channels, 3, 1, 1)
        self.conv_last = nn.Conv2d(channels, 1, 3, 1, 1)
        
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        feat = self.conv_first(x)
        body_feat = self.body(feat)
        body_feat = self.conv_body(body_feat)
        feat = feat + body_feat
        
        # Upsampling
        feat = self.upconv(feat)
        feat = self.pixel_shuffle(feat)
        feat = self.lrelu(feat)
        feat = self.conv_hr(feat)
        feat = self.lrelu(feat)
        out = self.conv_last(feat)
        
        return torch.tanh(out)

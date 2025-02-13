import torch
import torch.nn as nn
from typing import Tuple

class SRNet(nn.Module):
    """Super Resolution Network for upscaling low-resolution images.
    
    Architecture:
        - Initial upsampling using bicubic interpolation
        - Series of convolutional layers with ReLU activation
        - Final convolution to produce output image
        
    Args:
        in_channels (int): Number of input channels (default: 1 for grayscale)
        hidden_channels (int): Number of channels in hidden layers
        upscale_factor (int): Factor by which to upscale the input (default: 4)
    """
    
    def __init__(
        self, 
        in_channels: int = 1,
        hidden_channels: int = 64,
        upscale_factor: int = 4
    ):
        super(SRNet, self).__init__()
        
        self.upscale_factor = upscale_factor
  
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, 
            mode='bicubic', 
            align_corners=False
        )
      
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, hidden_channels//2, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(hidden_channels//2, hidden_channels//2, kernel_size=3, padding=1)
        self.conv5 = nn.Conv2d(hidden_channels//2, in_channels, kernel_size=3, padding=1)
        
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the model.
        
        Args:
            x (torch.Tensor): Input tensor of shape (B, C, H, W)
            
        Returns:
            torch.Tensor: Upscaled output tensor of shape (B, C, H*scale, W*scale)
        """
        x = self.upsample(x)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.relu(self.conv4(x))
        x = self.conv5(x)
        return x

    def get_output_shape(self, input_shape: Tuple[int, ...]) -> Tuple[int, ...]:
        """Calculate output shape for a given input shape.
        
        Args:
            input_shape (tuple): Input shape (B, C, H, W)
            
        Returns:
            tuple: Output shape (B, C, H*scale, W*scale)
        """
        B, C, H, W = input_shape
        return (B, C, H * self.upscale_factor, W * self.upscale_factor)

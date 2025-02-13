import torch
import torch.nn as nn
import torch.quantization as quant
from typing import Tuple

class QuantizedSRNet(nn.Module):
    """Quantization-ready version of SRNet for resource-efficient deployment.
    
    This model includes:
    - QConfig specification for quantization
    - Quantization stubs for activation quantization
    - Support for both static and dynamic quantization
    """
    
    def __init__(
        self, 
        in_channels: int = 1,
        hidden_channels: int = 64,
        upscale_factor: int = 4
    ):
        super(QuantizedSRNet, self).__init__()
        
        self.quant = quant.QuantStub()
        self.dequant = quant.DeQuantStub()
        
        self.upsample = nn.Upsample(
            scale_factor=upscale_factor, 
            mode='bicubic', 
            align_corners=False
        )
        
        self.conv_layers = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels, hidden_channels//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels//2, hidden_channels//2, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_channels//2, in_channels, 3, padding=1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.quant(x)
        
        x = self.upsample(x)
        x = self.conv_layers(x)
        
        x = self.dequant(x)
        return x

    def fuse_model(self):
        """Fuse Conv+ReLU layers for better performance"""
        torch.quantization.fuse_modules(
            self.conv_layers,
            [
                ['0', '1'],
                ['2', '3'],
                ['4', '5'],
                ['6', '7']
            ],
            inplace=True
        )

def prepare_model_for_quantization(model: nn.Module, backend='fbgemm') -> nn.Module:
    """Prepare model for static quantization"""
    model.eval()
  
    model.qconfig = torch.quantization.get_default_qconfig(backend)
    model.fuse_model()
    
    model_prepared = torch.quantization.prepare(model)
    
    return model_prepared

def quantize_model(model: nn.Module, calibration_loader) -> nn.Module:
    """Quantize model using calibration data"""
    with torch.no_grad():
        for inputs, _ in calibration_loader:
            model(inputs)
    
    model_quantized = torch.quantization.convert(model)
    
    return model_quantized

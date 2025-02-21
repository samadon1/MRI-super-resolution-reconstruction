import torch
import torch.nn.functional as F

def calculate_psnr(sr_img, hr_img):
    """Calculate PSNR (Peak Signal-to-Noise Ratio)"""
    mse = F.mse_loss(sr_img, hr_img)
    return 20 * torch.log10(2.0 / torch.sqrt(mse))

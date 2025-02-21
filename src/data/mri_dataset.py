import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchio as tio

class MedicalDataset(Dataset):
    def __init__(self, scale_factor=2):
        super().__init__()
        self.subject = tio.datasets.Colin27()
        t1_image = self.subject.t1.data[0]
        self.scale_factor = scale_factor
        
        # Extract and preprocess slices
        self.slices = []
        start_idx = t1_image.shape[0] // 4
        end_idx = 3 * t1_image.shape[0] // 4
        
        for i in range(start_idx, end_idx):
            slice_data = t1_image[i].numpy()
            slice_data = (slice_data - slice_data.min()) / (slice_data.max() - slice_data.min())
            slice_data = slice_data * 2 - 1  # Scale to [-1, 1]
            
            h, w = slice_data.shape
            new_h = (h // scale_factor) * scale_factor
            new_w = (w // scale_factor) * scale_factor
            slice_data = slice_data[:new_h, :new_w]
            
            self.slices.append(slice_data)
        
        print(f"Found {len(self.slices)} valid slices")
    
    def __len__(self):
        return len(self.slices)
    
    def __getitem__(self, idx):
        hr_img = torch.from_numpy(self.slices[idx]).float()
        
        lr_img = F.interpolate(
            hr_img.unsqueeze(0).unsqueeze(0),
            scale_factor=1/self.scale_factor,
            mode='bicubic',
            align_corners=False
        ).squeeze()
        
        return lr_img, hr_img

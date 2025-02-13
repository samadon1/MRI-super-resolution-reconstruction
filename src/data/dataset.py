import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from sklearn.datasets import load_digits
from tqdm import tqdm
import matplotlib.pyplot as plt

class DigitDataset:
    """Dataset generator for digit super-resolution development"""
    
    def __init__(self, base_dir='./data'):
        self.base_dir = Path(base_dir)
        self.processed_dir = self.base_dir / 'processed'
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def generate_sample_data(self):
        """Generate sample data using sklearn's digits dataset"""
        print("Generating sample data...")
        digits = load_digits()
        images = digits.images
        scaled_images = []
        for img in images:
            scaled = np.repeat(np.repeat(img, 4, axis=0), 4, axis=1)
            scaled_images.append(scaled)
        print(f"Generated {len(scaled_images)} images of size {scaled_images[0].shape}")
        return scaled_images

    def process_and_save(self):
        """Process and save the sample data"""
        print("Preprocessing data...")
        images = self.generate_sample_data()
        
        for i, img in enumerate(tqdm(images, desc="Processing files")):
            try:
                normalized = (img - img.min()) / (img.max() - img.min())
                output_path = self.processed_dir / f"sample_{i:03d}.npy"
                np.save(str(output_path), normalized)
                if i == 0:
                    self.save_visualization(normalized)
                
            except Exception as e:
                print(f"Error processing image {i}: {str(e)}")
        
        print("Preprocessing completed!")

    def save_visualization(self, image, filename='sample_image.png'):
        """Save a visualization of an image"""
        plt.figure(figsize=(6, 6))
        plt.imshow(image, cmap='gray')
        plt.axis('off')
        plt.savefig(filename, bbox_inches='tight', pad_inches=0)
        plt.close()

    def get_processed_files(self):
        """Return list of processed files"""
        return list(self.processed_dir.glob('*.npy'))

class SuperResolutionDataset(Dataset):
    """PyTorch Dataset for super-resolution training"""
    
    def __init__(self, file_paths, low_res_size=(8, 8), high_res_size=(32, 32)):
        self.file_paths = file_paths
        self.low_res_size = low_res_size
        self.high_res_size = high_res_size

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        img = np.load(str(self.file_paths[idx]))
        img_tensor = torch.FloatTensor(img).unsqueeze(0)
        low_res = torch.nn.functional.interpolate(
            img_tensor.unsqueeze(0),
            size=self.low_res_size,
            mode='bicubic',
            align_corners=False
        ).squeeze(0)
        high_res = img_tensor
        return low_res, high_res

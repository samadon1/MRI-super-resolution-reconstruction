# MRI Super Resolution Models

This repository contains code for developing and testing super-resolution models.
Used MNIST digits as a development dataset before moving to real MRI data.

## Project Structure

```
MRI-super-resolution-reconstruction/
├── data/               
├── src/               
├── scripts/          
└── tests/            
```

## Installation

```bash
# Clone repository
git clone https://github.com/yourusername/super-resolution.git
cd super-resolution

# Install requirements
pip install -r requirements.txt
```

## Usage

1. Download and process the dataset:
```bash
python scripts/download_data.py
```

2. Train the model:
```bash
python scripts/train.py
```

3. Evaluate results:
```bash
python scripts/evaluate.py
```

## Development Dataset

Currently using the MNIST digits dataset (scaled to 32x32) as a development dataset for rapid prototyping and testing of the super-resolution pipeline before moving to real MRI data.

Each image goes through the following process:
1. Original 8x8 digits are scaled to 32x32
2. Images are normalized to [0, 1] range
3. Saved as numpy arrays in data/processed/

## Super Resolution Pipeline

1. Input: 8x8 low-resolution image
2. Output: 32x32 high-resolution image
3. Training objective: Minimize MSE between predicted and actual high-res images

## Results
Single example comparison
![8772f586-ddaa-4047-8d9b-bb2c2fe263d1](https://github.com/user-attachments/assets/c5cc1505-b09a-45e7-9c30-1350196e8919)

Detailed comparison
![b7b287ce-edb7-41cd-a36e-e54ccaddc9e6](https://github.com/user-attachments/assets/9b3ff54c-8e86-40bf-971b-dabb46167ae3)

## Super-resolution model for enhancing low-resolution MRI scans using an Enhanced Super-Resolution GAN (ESRGAN) architecture.
- 2x upscaling of MRI images
- RRDB-based generator architecture
- Discriminator with spectral normalization
- Perceptual and adversarial loss
- Comprehensive visualization tools

## Results
- PSNR: 36.79 dB
- Stable training performance
- High-quality anatomical detail preservation

Multiple examples demonstrating consistent performance across different brain MRI slices. Each row shows a different slice with its corresponding low-resolution input (108x90), super-resolution output (216x180), and high-resolution ground truth (216x180). 

![download (15)](https://github.com/user-attachments/assets/cb855e9b-a6ff-4459-b650-46cc45c25bd9)

Detailed comparison showing the effectiveness of our super-resolution model. Top row shows full brain MRI scans, while bottom row shows zoomed regions highlighting the enhanced detail in brain tissue structures. The super-resolution output (middle) achieves comparable quality to the high-resolution ground truth (right), demonstrating significant improvement over the low-resolution input (left).

![download (16)](https://github.com/user-attachments/assets/66bc5903-5258-4c45-8126-13e95ce1c049)





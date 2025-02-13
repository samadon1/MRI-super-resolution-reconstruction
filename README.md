# Super Resolution Development

This repository contains code for developing and testing super-resolution models.
Currently using MNIST digits as a development dataset before moving to real MRI data.

## Project Structure

```
super-resolution/
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




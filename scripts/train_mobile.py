import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import matplotlib.pyplot as plt

from models.mobile_esrgan import MobileESRGAN
from data.dataset import MedicalDataset
from utils.metrics import calculate_psnr
from utils.visualization import save_sample_images
from utils.export import export_to_onnx, quantize_model, benchmark_model, optimize_for_mobile

def train_mobile_model(train_loader, val_loader, num_epochs=50, device='cuda'):
    os.makedirs('results/mobile', exist_ok=True)
  
    generator = MobileESRGAN(num_blocks=8, channels=32).to(device)
    discriminator = nn.Sequential(
        nn.Conv2d(1, 64, 3, stride=1, padding=1),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(64, 64, 3, stride=2, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(64, 128, 3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(128, 128, 3, stride=2, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(128, 256, 3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.LeakyReLU(0.2, True),
        nn.Conv2d(256, 1, 3, padding=1),
        nn.Sigmoid()
    ).to(device)
    
    criterion_pixel = nn.L1Loss()
    criterion_gan = nn.BCELoss()
    
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=0.0001, betas=(0.9, 0.999))
    
    scheduler_G = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_G, T_max=num_epochs)
    scheduler_D = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_D, T_max=num_epochs)
    
    history = {'g_loss': [], 'd_loss': [], 'psnr': []}
    best_psnr = 0
    
    for epoch in range(num_epochs):
        generator.train()
        discriminator.train()
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        for lr_imgs, hr_imgs in pbar:
            batch_size = lr_imgs.size(0)
            real = hr_imgs.unsqueeze(1).to(device)
            lr = lr_imgs.unsqueeze(1).to(device)
            
            optimizer_D.zero_grad()
            fake = generator(lr)
            pred_real = discriminator(real)
            pred_fake = discriminator(fake.detach())
            
            loss_real = criterion_gan(pred_real, torch.ones_like(pred_real))
            loss_fake = criterion_gan(pred_fake, torch.zeros_like(pred_fake))
            loss_D = (loss_real + loss_fake) / 2
            
            loss_D.backward()
            optimizer_D.step()
            

            optimizer_G.zero_grad()
            pred_fake = discriminator(fake)
            
            loss_G_gan = criterion_gan(pred_fake, torch.ones_like(pred_fake))
            loss_G_content = criterion_pixel(fake, real)
            loss_G = loss_G_content + 0.001 * loss_G_gan
            
            loss_G.backward()
            optimizer_G.step()
            
            pbar.set_postfix({'G_loss': loss_G.item(), 'D_loss': loss_D.item()})
        
        scheduler_G.step()
        scheduler_D.step()
        
        generator.eval()
        val_psnr = 0
        with torch.no_grad():
            for lr_imgs, hr_imgs in val_loader:
                lr = lr_imgs.unsqueeze(1).to(device)
                hr = hr_imgs.unsqueeze(1).to(device)
                sr = generator(lr)
                val_psnr += calculate_psnr(sr, hr).item()
            
            val_psnr /= len(val_loader)
            print(f"\nValidation PSNR: {val_psnr:.2f} dB")
            
            if val_psnr > best_psnr:
                best_psnr = val_psnr
                torch.save({
                    'generator_state_dict': generator.state_dict(),
                    'discriminator_state_dict': discriminator.state_dict(),
                    'epoch': epoch,
                    'psnr': val_psnr
                }, 'results/mobile/best_model.pth')
        
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, val_loader, device, epoch+1, 'results/mobile')
        
        history['g_loss'].append(loss_G.item())
        history['d_loss'].append(loss_D.item())
        history['psnr'].append(val_psnr)
    
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['g_loss'], label='Generator Loss')
    plt.plot(history['d_loss'], label='Discriminator Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history['psnr'], label='PSNR (dB)')
    plt.xlabel('Epoch')
    plt.ylabel('PSNR (dB)')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('results/mobile/training_history.png')
    
    return generator, history

def export_models(generator, test_loader, device):
    """Export and optimize models for deployment"""
    lr_imgs, _ = next(iter(test_loader))
    sample_input = lr_imgs.unsqueeze(1).to(device)
    onnx_path = export_to_onnx(
        generator, 
        sample_input, 
        'results/mobile', 
        'mobilesrgan'
    )
  
    optimize_for_mobile(
        onnx_path, 
        'results/mobile/mobilesrgan_optimized.onnx'
    )
  
    generator_quantized = quantize_model(generator.cpu(), test_loader, 'cpu')
    torch.save({
        'model_state_dict': generator_quantized.state_dict()
    }, 'results/mobile/quantized_model.pth')
  
    export_to_onnx(
        generator_quantized, 
        sample_input.cpu(), 
        'results/mobile', 
        'mobilesrgan_quantized'
    )
    
    print("Benchmarking original model...")
    original_benchmark = benchmark_model(generator.to('cpu'), test_loader, 'cpu')
    
    print("Benchmarking quantized model...")
    quantized_benchmark = benchmark_model(generator_quantized, test_loader, 'cpu')
    
    print("\nBenchmark Results:")
    print(f"Original Model:")
    print(f"  - PSNR: {original_benchmark['psnr']:.2f} dB")
    print(f"  - Inference Time: {original_benchmark['inference_time_ms']:.2f} ms")
    print(f"  - Model Size: {original_benchmark['model_size_mb']:.2f} MB")
    
    print(f"\nQuantized Model:")
    print(f"  - PSNR: {quantized_benchmark['psnr']:.2f} dB")
    print(f"  - Inference Time: {quantized_benchmark['inference_time_ms']:.2f} ms")
    print(f"  - Model Size: {quantized_benchmark['model_size_mb']:.2f} MB")
  
    benchmark_results = {
        'original': original_benchmark,
        'quantized': quantized_benchmark
    }
    
    torch.save(benchmark_results, 'results/mobile/benchmark_results.pth')
    
    return benchmark_results

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
  
    dataset = MedicalDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    generator, history = train_mobile_model(train_loader, val_loader, num_epochs=40, device=device)
    
    benchmark_results = export_models(generator, val_loader, device)

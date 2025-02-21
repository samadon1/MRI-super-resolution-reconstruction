import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

def train_model(generator, discriminator, train_loader, val_loader, num_epochs=50, device='cuda'):
    os.makedirs('results', exist_ok=True)
    
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
                }, 'results/best_model.pth')
        
      
        if (epoch + 1) % 5 == 0:
            save_sample_images(generator, val_loader, device, epoch+1, 'results')
        
        history['g_loss'].append(loss_G.item())
        history['d_loss'].append(loss_D.item())
        history['psnr'].append(val_psnr)
    
    return history

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    dataset = MedicalDataset()
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=4)
    
    generator = Generator().to(device)
    discriminator = Discriminator().to(device)
    
    history = train_model(generator, discriminator, train_loader, val_loader)

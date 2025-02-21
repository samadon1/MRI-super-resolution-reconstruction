import matplotlib.pyplot as plt

def save_sample_images(generator, dataloader, device, epoch, save_path):
    generator.eval()
    with torch.no_grad():
        lr_imgs, hr_imgs = next(iter(dataloader))
        lr = lr_imgs.unsqueeze(1).to(device)
        sr = generator(lr)
        
        # Convert from [-1, 1] to [0, 1]
        lr = (lr + 1) / 2
        sr = (sr + 1) / 2
        hr = (hr_imgs.unsqueeze(1).to(device) + 1) / 2
        
        plt.figure(figsize=(15, 5))
        plt.subplot(131)
        plt.imshow(lr[0, 0].cpu(), cmap='gray')
        plt.title(f'Low Resolution {lr.shape[-2:]}')
        plt.axis('off')
        
        plt.subplot(132)
        plt.imshow(sr[0, 0].cpu(), cmap='gray')
        plt.title(f'Super Resolution {sr.shape[-2:]}')
        plt.axis('off')
        
        plt.subplot(133)
        plt.imshow(hr[0, 0].cpu(), cmap='gray')
        plt.title(f'High Resolution {hr.shape[-2:]}')
        plt.axis('off')
        
        plt.savefig(f'{save_path}/epoch_{epoch}.png')
        plt.close()

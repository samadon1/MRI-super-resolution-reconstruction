import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

from src.data.dataset import SuperResolutionDataset
from src.models.srnet import SRNet

def train_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    num_epochs: int = 50,
    device: str = 'cuda',
    save_dir: str = 'results'
) -> tuple:
    """Train the super-resolution model.
    
    Args:
        model: The SRNet model
        train_loader: Training data loader
        val_loader: Validation data loader
        num_epochs: Number of training epochs
        device: Device to train on ('cuda' or 'cpu')
        save_dir: Directory to save results
        
    Returns:
        tuple: Training and validation losses
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    best_val_loss = float('inf')
    
    train_losses = []
    val_losses = []
    
    model = model.to(device)
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for low_res, high_res in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            low_res, high_res = low_res.to(device), high_res.to(device)
            
            optimizer.zero_grad()
            output = model(low_res)
            loss = criterion(output, high_res)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for low_res, high_res in val_loader:
                low_res, high_res = low_res.to(device), high_res.to(device)
                output = model(low_res)
                val_loss += criterion(output, high_res).item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch [{epoch+1}/{num_epochs}]')
        print(f'Training Loss: {avg_train_loss:.6f}')
        print(f'Validation Loss: {avg_val_loss:.6f}')
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), f'{save_dir}/best_model.pth')
    
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.savefig(f'{save_dir}/training_history.png')
    plt.close()
    
    return train_losses, val_losses

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    

    data_dir = Path('./data/processed')
    file_paths = list(data_dir.glob('*.npy'))
    

    train_paths, val_paths = train_test_split(file_paths, test_size=0.2, random_state=42)
    

    train_dataset = SuperResolutionDataset(train_paths)
    val_dataset = SuperResolutionDataset(val_paths)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    

    model = SRNet()

    train_losses, val_losses = train_model(
        model,
        train_loader,
        val_loader,
        num_epochs=50,
        device=device,
        save_dir='results'
    )

if __name__ == "__main__":
    main()

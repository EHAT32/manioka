from model import RootVolumeNet
from dataset import RootVolumeDataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

def train():
    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    
    # Dataset
    dataset = RootVolumeDataset(
        csv_path='data.csv',
        img_root='images/',
        target_width=2048,
        target_height=16
    )
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Model
    model = RootVolumeNet(max_slices=50).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.L1Loss()
    
    # Training loop
    for epoch in range(NUM_EPOCHS):
        model.train()
        train_loss = 0
        
        for batch in dataloader:
            images = batch['images'].to(device)
            volumes = batch['volume'].to(device)
            
            pred = model(images)
            loss = criterion(pred, volumes)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {train_loss/len(dataloader):.4f}")

if __name__ == '__main__':
    train()
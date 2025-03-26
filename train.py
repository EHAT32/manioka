from model import RootVolumeNet
from dataset import RootVolumeDataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def custom_collate_fn(batch):
    """
    Collate function to handle variable-sized sequences by padding them.
    
    Args:
        batch (list): List of dictionaries containing 'images' and 'volume'.
    
    Returns:
        dict: Batch with padded 'images' and stacked 'volume'.
    """
    # Extract images and volumes from the batch
    images = [item['images'] for item in batch]
    volumes = [item['volume'] for item in batch]
    
    # Pad images to the same length
    images_padded = pad_sequence(images, batch_first=True, padding_value=0)
    
    # Stack volumes
    volumes_stacked = torch.stack(volumes)
    
    return {
        'images': images_padded,
        'volume': volumes_stacked
    }

def nearest_power_of_2(n):
    return 2 ** int(np.ceil(np.log2(n)))

def find_max_slice(csv_path):
    df = pd.read_csv(csv_path)
    return nearest_power_of_2(np.max(df["End"] - df["Start"]) + 1)

def get_params():
    size_file = open("target_size.txt", "r")
    width, height = [int(line.replace("\n", "").split(" ")[-1]) for line in size_file.readlines()]
    params = {}
    params["width"] = width
    params["height"] = height
    params["max_slices"] = find_max_slice("Train.csv")
    return params

def train():
    torch.manual_seed(0)
    # Hyperparameters
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    NUM_EPOCHS = 50
    params = get_params()
    # Dataset
    dataset = RootVolumeDataset(
        csv_path='Train.csv',
        img_root='images/train/',
        target_width=params["width"],
        target_height=params["height"]
    )
    
    # DataLoader
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # Model
    model = RootVolumeNet(max_slices=params["max_slices"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()
    
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
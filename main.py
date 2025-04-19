from model import RootVolumeNet
from dataset import RootVolumeDataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from common import *

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
    dataset[0]
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
from dataset import ProcessedDataset
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from common import *
from torchvision.transforms import v2
from model import RootRegressor
import lightning as L
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch.callbacks import Callback
import matplotlib.pyplot as plt
import os

# 1. Define your metrics callback
class MetricsPlotter(Callback):
    def __init__(self):
        super().__init__()
        self.train_rmse = []
    
    def on_train_epoch_end(self, trainer, pl_module):
        current_rmse = trainer.callback_metrics.get("train_rmse")
        if current_rmse is not None:
            self.train_rmse.append(current_rmse.item())

def get_model_preds(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    preds, targets = [], []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images, labels = batch
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)  # Predict
            preds.extend(outputs.cpu().numpy().flatten())
            targets.extend(labels.cpu().numpy().flatten())

    return np.array(preds), np.array(targets)

def calculate_rmse(preds, targets):
    """
    Compute Root Mean Squared Error (RMSE) between predictions and ground truth targets.
    """
    preds = np.array(preds) if not isinstance(preds, np.ndarray) else preds
    targets = np.array(targets) if not isinstance(targets, np.ndarray) else targets
    
    return np.sqrt(np.mean((preds - targets) ** 2))

def get_test_preds(model, dataloader, device="cuda" if torch.cuda.is_available() else "cpu"):
    preds = []
    model.to(device)
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch if isinstance(batch, torch.Tensor) else batch[0]
            images = images.to(device)

            outputs = model(images)
            preds.extend(outputs.cpu().numpy().flatten()) 

    return np.array(preds)

if __name__ == '__main__':
    torch.manual_seed(0)
    
    train_transform = v2.Compose([
    v2.Resize(size=(20, 150), antialias=True),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  
    v2.Normalize(mean=[0.5], std=[0.5])
    ])
    
    test_transform = v2.Compose([
    v2.Resize(size=(20, 150), antialias=True),
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),  
    v2.Normalize(mean=[0.5], std=[0.5])
    ])
    
    train_df = pd.read_csv("processed_train.csv")
    test_df = pd.read_csv("processed_test.csv")
    train_dataset = ProcessedDataset(train_df, train_transform)
    test_dataset = ProcessedDataset(test_df, test_transform, is_train = False)
    
    train_dataloader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
    num_workers=0,
    pin_memory=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )
    
    metrics_plotter = MetricsPlotter()
    csv_logger = CSVLogger("logs", name="root_volume_experiment")
    
    # trainer = L.Trainer(
    #     max_epochs=20,
    #     callbacks=[metrics_plotter],
    #     log_every_n_steps=5,  # Log every 5 batches (20% of your epoch)
    #     logger=csv_logger,
    #     enable_progress_bar=True,
    #     enable_model_summary=True,
    #     deterministic=True  # For reproducibility
    # )
    
    model = RootVolumeRegressor()
    
    trainer = L.Trainer(max_epochs=20)
    trainer.fit(model, train_dataloader)
    
    # Plot results with proper error checking
    plt.figure(figsize=(10, 5))
    recent = os.listdir("lightning_logs/")[-1] + "/" + "metrics.csv"
    recent = "lightning_logs/" + recent
    data = pd.read_csv(recent)
    plt.plot(data["epoch"], data["train_loss"])
    plt.show()
    train_preds, target = get_model_preds(model, train_dataloader)
    
    err = calculate_rmse(train_preds, target)
    print("ERROR: ", err)
    
    test_preds = get_test_preds(model, test_dataloader)
    
    test_df["preds"] = test_preds
    
    print(test_df.head())
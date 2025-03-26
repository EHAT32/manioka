import torch
import torch.nn as nn

class RootVolumeNet(nn.Module):
    def __init__(self, max_slices=50):
        super().__init__()
        self.max_slices = max_slices
        
        # Shared encoder for 4-channel input (RGB + mask)
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(32, 64, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            
            nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2))
        )
        
        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(128 * 256 * 2, 512),  # Adjusted for 2048x16 input
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        batch_size, num_slices, C, H, W = x.shape
        
        # Encode all slices
        features = self.encoder(x.view(-1, C, H, W))
        features = features.view(batch_size, num_slices, -1)
        
        # Handle variable slice counts
        if num_slices < self.max_slices:
            padding = torch.zeros(batch_size, self.max_slices - num_slices, 
                                features.shape[-1], device=x.device)
            features = torch.cat([features, padding], dim=1)
        
        # Aggregate features
        agg_features = features.sum(dim=1)
        
        # Predict volume
        return self.regressor(agg_features).squeeze()
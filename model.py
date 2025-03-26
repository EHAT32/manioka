import numpy as np
import cv2
import torch
import torch.nn as nn
from ultralytics import YOLO 
import os

class RootVolumeNet(nn.Module):
    def __init__(self, seg_model_path, max_slices=50):
        super().__init__()
        self.max_slices = max_slices
        
        # YOLOv11 segmentation model
        self.seg_model = YOLO(seg_model_path).eval()
        
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

    def _apply_segmentation(self, images):
        """
        Applies YOLOv11 segmentation to a batch of images and returns masks.
        
        Args:
            images (torch.Tensor): Batch of images, shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Segmentation masks, shape (B, 1, H, W).
        """
        masks = []
        for img in images:
            # Convert tensor to NumPy array
            img_np = img.permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            img_np = (img_np * 255).astype(np.uint8)  # Denormalize
            
            # Get segmentation mask
            results = self.seg_model(img_np)
            mask = results[0].masks.data[0].cpu().numpy()  # Get first mask
            
            # Resize mask to match image size
            mask = cv2.resize(mask, (images.shape[2], images.shape[3]))
            masks.append(mask)
        
        # Convert masks to tensor
        masks = torch.tensor(np.stack(masks), dtype=torch.float32).unsqueeze(1).to(images.device)
        return masks

    def forward(self, x):
        batch_size, num_slices, C, H, W = x.shape
        
        # Apply segmentation to each slice
        masks = torch.cat([self._apply_segmentation(x[:, i]) for i in range(num_slices)], dim=1)
        
        # Combine RGB + mask as 4-channel input
        x = torch.cat([x, masks], dim=2)  # Shape: (B, num_slices, 4, H, W)
        
        # Encode all slices
        features = self.encoder(x.view(-1, 4, H, W))
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
    
    def verify_segmentation(self, images, save_dir='segmentation_verification'):
        """
        Verifies the segmentation model by saving input images and their masks.
        
        Args:
            images (torch.Tensor): Batch of images, shape (B, C, H, W).
            save_dir (str): Directory to save verification results.
        """
        import matplotlib.pyplot as plt
        os.makedirs(save_dir, exist_ok=True)
        
        # Apply segmentation
        masks = self._apply_segmentation(images)
        
        # Save images and masks
        for i in range(images.shape[0]):
            img_np = images[i].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
            img_np = (img_np * 255).astype(np.uint8)  # Denormalize
            mask_np = masks[i].squeeze().cpu().numpy()  # (H, W)
            
            # Plot and save
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            ax[0].imshow(img_np)
            ax[0].set_title('Input Image')
            ax[0].axis('off')
            
            ax[1].imshow(mask_np, cmap='gray')
            ax[1].set_title('Segmentation Mask')
            ax[1].axis('off')
            
            plt.savefig(os.path.join(save_dir, f'segmentation_{i}.png'))
            plt.close()
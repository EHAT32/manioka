import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
from ultralytics import YOLO 
import matplotlib.pyplot as plt

class RootVolumeDataset(Dataset):
    def __init__(self, csv_path, img_root, target_width, target_height, transform=None):
        """
        Args:
            csv_path (str): Path to CSV file with annotations.
            img_root (str): Root directory containing image folders.
            target_width (int): Desired width after padding.
            target_height (int): Desired height after padding.
            transform (callable): Optional transform to be applied.
        """
        self.df = pd.read_csv(csv_path)
        self.img_root = img_root
        self.target_width = target_width
        self.target_height = target_height
        self.transform = transform or transforms.Compose([
            transforms.ToTensor(),
            # transforms.Normalize([0.5]*4, [0.5]*4)  # 4 channels (RGB + mask)
        ])
        self.yolo_seg = YOLO('seg_models/best_full.pt').eval()  # YOLOv11 for segmentation
        
    def __len__(self):
        return len(self.df)
    
    def _zero_pad_image(self, image):
        """
        Pads an image with zeros to reach the target dimensions exactly.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: Padded image with exact target dimensions (H, W).
        """
        # Convert image to NumPy array
        img_array = np.array(image)
        original_height, original_width = img_array.shape[:2]
        
        # Calculate required padding
        pad_height = max(0, self.target_height - original_height)
        pad_width = max(0, self.target_width - original_width)
        
        # Divide padding into top/bottom and left/right
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top  # Ensure total height matches
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left  # Ensure total width matches
        
        # Apply zero-padding
        padded_array = np.pad(
            img_array,
            ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # print(f"Original shape: {img_array.shape}, Padded shape: {padded_array.shape}")
        
        # Convert back to PIL Image
        return Image.fromarray(padded_array)
    
    def _apply_segmentation(self, image):
        """
        Applies YOLOv11 segmentation to an image and returns the mask.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            numpy.ndarray: Segmentation mask.
        """
        # Convert to NumPy array for YOLOv11
        img_array = np.array(image)
        
        # Get segmentation mask
        results = self.yolo_seg(img_array)
        
        # Check if masks were detected
        if results[0].masks is None:
            # Return a zero mask if no objects detected
            return np.zeros((self.target_height, self.target_width), dtype=np.float32)
        
        # Get first mask
        mask = results[0].masks.data[0].cpu().numpy()
        
        # Resize mask to match image size
        mask = cv2.resize(mask, (self.target_width, self.target_height))
        return mask
    
    def _load_slice_sequence(self, folder, side, start, end):
        """Load sequence of slices for given range"""
        images = []
        for i in range(start, end+1):
            img_path = os.path.join(
                self.img_root,
                folder,
                f"{folder}_{side}_{i:03d}.png"
            )
            if os.path.exists(img_path):
                img = Image.open(img_path).convert('RGB')
                img = self._zero_pad_image(img)  # Apply zero padding
                mask = self._apply_segmentation(img)  # Get segmentation mask
                
                # Combine RGB + mask as 4-channel input
                combined = np.concatenate([img, mask[..., None]], axis=-1)
                images.append(combined)
            else:
                # Handle missing slices with zero padding
                images.append(np.zeros((self.target_height, self.target_width, 4), dtype=np.uint8))
        return np.stack(images)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = row['FolderName']
        plant_num = row['PlantNumber']
        side = row['Side']
        start = row['Start']
        end = row['End']
        volume = row['RootVolume']
        
        # Load image sequence
        images = self._load_slice_sequence(folder, side, start, end)
        
        # Apply transforms
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])
        
        return {
            'images': images,  # Shape: (num_slices, 4, H, W)
            'volume': torch.tensor(volume, dtype=torch.float32),
            'plant_num': plant_num,  # For debugging/analysis
            'num_slices': end - start + 1  # For dynamic padding
        }

    def verify_segmentation(self, idx, save_dir='segmentation_verification'):
            """
            Verifies the segmentation by saving input images and their masks for inspection.
            
            Args:
                idx (int): Index of the sample to verify.
                save_dir (str): Directory to save verification results.
            """
            os.makedirs(save_dir, exist_ok=True)
            
            # Load the sample
            row = self.df.iloc[idx]
            folder = row['FolderName']
            side = row['Side']
            start = row['Start']
            end = row['End']
            
            # Load image sequence
            images = self._load_slice_sequence(folder, side, start, end)
            
            # Save images and masks
            for i in range(images.shape[0]):
                img_np = images[i][:, :, :3]  # RGB channels
                mask_np = images[i][:, :, 3]  # Mask channel
                
                # Plot and save
                fig, ax = plt.subplots(1, 2, figsize=(10, 5))
                ax[0].imshow(img_np)
                ax[0].set_title('Input Image')
                ax[0].axis('off')
                
                ax[1].imshow(mask_np, cmap='gray')
                ax[1].set_title('Segmentation Mask')
                ax[1].axis('off')
                
                plt.savefig(os.path.join(save_dir, f'segmentation_{idx}_{i}.png'))
                plt.close()
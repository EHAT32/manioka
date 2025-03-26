import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pandas as pd

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
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])
        
    def __len__(self):
        return len(self.df)
    
    def _zero_pad_image(self, image):
        """
        Pads an image with zeros to reach the target dimensions.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            PIL.Image: Padded image.
        """
        # Convert image to NumPy array
        img_array = np.array(image)
        
        # Calculate padding amounts
        pad_height = (self.target_height - img_array.shape[0]) // 2
        pad_width = (self.target_width - img_array.shape[1]) // 2
        
        # Pad the image with zeros
        padded_array = np.pad(
            img_array,
            ((pad_height, pad_height), (pad_width, pad_width), (0, 0)),
            mode='constant',
            constant_values=0
        )
        
        # Convert back to PIL Image
        return Image.fromarray(padded_array)
    
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
                images.append(np.array(img))
            else:
                # Handle missing slices with zero padding
                images.append(np.zeros((self.target_height, self.target_width, 3), dtype=np.uint8))
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
            'images': images,  # Shape: (num_slices, 3, H, W)
            'volume': torch.tensor(volume, dtype=torch.float32),
            'plant_num': plant_num,  # For debugging/analysis
            'num_slices': end - start + 1  # For dynamic padding
        }

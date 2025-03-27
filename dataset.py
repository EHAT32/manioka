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
        self.yolo_seg = YOLO('seg_models/best_early.pt').eval()  # YOLOv11 for segmentation
        
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
    
    def _merge_left_right(self, limg, rimg):
        width = min(limg.width, rimg.width)
        height = limg.height
        limg = limg.crop((0, 0, width, height))
        height = rimg.height
        rimg = rimg.crop((0, 0, width, height))
        
        stacked_img = Image.new("RGB", (width, limg.height+rimg.height))
        stacked_img.paste(rimg.rotate(180), (0,0))
        stacked_img.paste(limg, (0, rimg.height))
        # limg.show()
        # rimg.show()
        # stacked_img.show()
        return stacked_img
    
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
            return np.zeros((image.height, image.width, 3), dtype=np.float32)
        else:
            mask = results[0].plot()
        # mask = np.transpose(mask, (1, 0, 2))
        # Resize mask to match image size
        mask = cv2.resize(mask, (image.width, image.height))
        return mask
    
    def _load_slice_sequence(self, folder, side, start, end):
        """Load sequence of slices for given range"""
        images = []
        original_images = []  # List to store original images
        for i in range(start, end + 1):
            limg_path = os.path.join(
                self.img_root,
                folder,
                f"{folder}_L_{i:03d}.png"
            )
            rimg_path = os.path.join(
                self.img_root,
                folder,
                f"{folder}_R_{i:03d}.png"
            )
            if os.path.exists(limg_path) and os.path.exists(rimg_path):
                limg = Image.open(limg_path).convert('RGB')
                rimg = Image.open(rimg_path).convert('RGB')
                full_img = self._merge_left_right(limg, rimg)
                original_images.append(np.array(full_img)) # add to the list before converting to numpy array
                mask = self._apply_segmentation(full_img)#.transpose()  # Get segmentation mask
                
                # Combine RGB + mask as 4-channel input
                combined = np.concatenate([np.array(full_img), np.array(mask)], axis=-1)
                images.append(combined)
            else:
                # Handle missing slices with zero padding
                images.append(np.zeros((self.target_height, self.target_width, 6), dtype=np.uint8))
        
        # Plot all original images stacked vertically in a single figure
        num_images = len(original_images)
        if num_images > 0:
            fig, axes = plt.subplots(num_images, 1, figsize=(5, 3 * num_images))  # Adjust size as needed
            
            for i, img in enumerate(original_images):
                axes[i].imshow(img)
                axes[i].axis('off')
                axes[i].set_title(f"Slice {start + i}")  # Set title for each slice

            plt.tight_layout()  # Adjust layout to prevent overlapping titles
            plt.show()
            
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
            'images': images,  # Shape: (num_slices, 6, H, W)
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
            img_np = images[i][:, :, :3].astype('uint8')  # RGB channels
            mask_np = images[i][:, :, 3:].astype('uint8')  # Mask channel
            
            # Normalize the image data to the range [0, 1]
            img_np = (img_np - np.min(img_np)) / (np.max(img_np) - np.min(img_np))
            
            # Plot and save
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            # Display the image (no transposition for now)
            ax[0].imshow(img_np, origin='upper')
            ax[0].set_title('Input Image')
            ax[0].axis('off')
            
            # Display the mask (no transposition for now)
            ax[1].imshow(mask_np, cmap='gray', origin='upper')
            ax[1].set_title('Segmentation Mask')
            ax[1].axis('off')
            
            # Save before showing
            save_path = os.path.join(save_dir, f'segmentation_{idx}_{i}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=600)  # Adjust DPI as needed
            plt.close(fig)  # Close the figure after saving to free memory

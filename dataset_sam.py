import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from segment_anything import SamPredictor, sam_model_registry
from tqdm import tqdm

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

        # Initialize SAM
        sam_checkpoint = "seg_models/sam_vit_l_0b3195.pth"  # Replace with your SAM checkpoint
        model_type = "vit_l"
        device = "cuda" if torch.cuda.is_available() else "cpu"

        sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
        sam.to(device=device)
        self.sam_predictor = SamPredictor(sam)
        
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
        Applies SAM segmentation to an image and returns the mask.
        
        Args:
            image (PIL.Image): Input image.
        
        Returns:
            numpy.ndarray: Segmentation mask.
        """
        img_array = np.array(image)
        self.sam_predictor.set_image(img_array)

        # Generate a default box around the entire image
        input_box = np.array([0, 0, img_array.shape[1], img_array.shape[0]])

        masks, _, _ = self.sam_predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=False,
        )

        mask = masks[0].astype(np.uint8) * 255  # Convert to binary mask
        # Convert to RGB
        mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
        return mask

    def draw_polygons(self, image, mask):
        """
        Draws polygons on an image based on the segmentation mask.

        Args:
            image (PIL.Image): Input image.
            mask (numpy.ndarray): Segmentation mask.

        Returns:
            PIL.Image: Image with drawn polygons.
        """
        img = image.copy()
        draw = ImageDraw.Draw(img)

        # Find contours in the mask
        mask_gray = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)
        contours, _ = cv2.findContours(mask_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Draw polygons for each contour
        for contour in contours:
            contour = contour.flatten().tolist()
            if len(contour) > 6:  # Minimum points to form a polygon
                draw.polygon(contour, outline="red", width=2)

        return img
    
    def _load_slice_sequence(self, folder, side, start, end):
        """Load sequence of slices for given range"""
        images = []
        original_images = []  # List to store original images
        for i in tqdm(range(start, end + 1)):
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
                
                # Apply SAM segmentation
                mask = self._apply_segmentation(full_img)
                
                # Draw polygons on the image
                full_img_with_polygons = self.draw_polygons(full_img, mask)
                original_images.append(np.array(full_img_with_polygons))  # add to the list before converting to numpy array
                
                # Combine RGB + mask as 4-channel input
                combined = np.concatenate([np.array(full_img_with_polygons), np.array(mask)], axis=-1)
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
                axes[i].set_title(f"Slice {start + i}")  # Set title for each slice... plt.tight_layout()  # Adjust layout to prevent overlapping titles
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
            
            # Plot and save
            fig, ax = plt.subplots(1, 2, figsize=(10, 5))
            
            # Display the image (no transposition for now)
            ax[0].imshow(img_np, origin='upper')
            ax[0].set_title('Input Image')
            ax[0].axis('off')
            
            # Display the mask (no transposition for now)
            ax[1].imshow(mask_np, origin='upper')
            ax[1].set_title('Segmentation Mask')
            ax[1].axis('off')
            
            # Save before showing
            save_path = os.path.join(save_dir, f'segmentation_{idx}_{i}.png')
            plt.savefig(save_path, bbox_inches='tight', dpi=600)  # Adjust DPI as needed
            plt.close(fig)  # Close the figure after saving to free memory

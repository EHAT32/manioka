import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import pandas as pd
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
from tqdm import tqdm

class RootVolumeDataset(Dataset):
    def __init__(self, csv_path, img_root, label_root,
                 target_width, target_height, device, train = True, 
                 transform=None, pre_segment = False):
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
        self.label_root = label_root
        self.target_width = target_width
        self.target_height = target_height
        self.transform = transform 
        self.is_train = train
        self.pre_segment = pre_segment
        self.device = device
        self.yolo_seg = YOLO('seg_models/best_full.pt').eval().to(self.device)

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

    def draw_polygons(self, image : Image, results):
        """
        Draws polygons on an image based on YOLO segmentation results.

        Args:
            image (PIL.Image): Input image.
            results: YOLO segmentation results containing polygon data.

        Returns:
            PIL.Image: Image with drawn polygons.
        """
        img = image.copy()  # Create a copy to avoid modifying the original image
        draw = ImageDraw.Draw(img)

        # Iterate over detected objects
        for result in results:
            if result.masks is not None:  # Ensure masks are present
                for mask in result.masks.xy:  # Access polygon data
                    # Draw the polygon on the image
                    draw.polygon(mask, outline="red", width=2)  # Customize color and width

        return img

    def _load_slice_sequence(self, folder, side, start, end):
        """Load sequence of slices for given range"""
        images = []
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
                crops = self._crop_segmented(full_img)
                # # Apply YOLO segmentation and draw polygons
                # yolo_results = self.yolo_seg(np.array(full_img))
                # full_img_with_polygons = self.draw_polygons(full_img, yolo_results)
                # original_images.append(np.array(full_img_with_polygons)) # add to the list before converting to numpy array

                # mask = self._apply_segmentation(full_img_with_polygons)  # Get segmentation mask

                # # Combine RGB + mask as 4-channel input
                # combined = np.concatenate([np.array(full_img_with_polygons), np.array(mask)], axis=-1)
                images.append(crops)
            else:
                # Handle missing slices with zero padding
                images.append([Image.new('RGB', (64, 64), (0,0,0))])

        # # Plot all original images stacked vertically in a single figure
        # num_images = len(original_images)
        # if num_images > 0:
        #     fig, axes = plt.subplots(num_images, 1, figsize=(5, 3 * num_images))  # Adjust size as needed

        #     for i, img in enumerate(original_images):
        #         axes[i].imshow(img)
        #         axes[i].axis('off')
        #         axes[i].set_title(f"Slice {start + i}")  # Set title for each slice... plt.tight_layout()  # Adjust layout to prevent overlapping titles
        #     plt.show()

        return images

    def _crop_segmented(self, image):
        """
        Crops each detected segment from the image and returns a list of cropped segments.
        If no segments are found, returns the original image.

        Args:
            image (PIL.Image): Input image.

        Returns:
            List[PIL.Image] or PIL.Image: List of cropped segments. If no segments are found, returns the original image.
        """
        # Convert the input image to a NumPy array for YOLO segmentation
        img_array = np.array(image)

        # Perform segmentation using YOLO
        results = self.yolo_seg(img_array, verbose=False)

        # Initialize a list to store cropped segments
        cropped_segments = []

        # Check if there are any detections
        if results[0].boxes.xyxy is not None and len(results[0].boxes.xyxy) > 0:
            # Iterate over the detected bounding boxes
            for box in results[0].boxes.xyxy:
                # Extract bounding box coordinates
                x1, y1, x2, y2 = map(int, box.tolist())

                # Crop the segment from the original image
                cropped_segment = image.crop((x1, y1, x2, y2))

                # Append the cropped segment to the list
                cropped_segments.append(cropped_segment)
        else:
            return [image]  # Return the original image if results are None

        return cropped_segments
    
    def _merge_crops(self, cropped_segments):
        """Merge all crops into one image
        """
        #squeeze into one list
        crops = [crop for slice in cropped_segments for crop in slice]
        total_width = sum(img.width for img in crops)
        max_height = max(img.height for img in crops)
        
        merged_img = Image.new("RGB", (total_width, max_height), (0,0,0))
        
        x_offset = 0
        for img in crops:
            merged_img.paste(img, (x_offset,0))
            x_offset += img.width
        return merged_img



    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        folder = row['FolderName']
        plant_num = row['PlantNumber']
        side = row['Side']
        start = row['Start']
        end = row['End']
        if self.is_train:
            volume = row['RootVolume']

        # Load image sequence
        images = self._load_slice_sequence(folder, side, start, end)
        merged_images = self._merge_crops(images)

        # Apply transforms
        if self.transform:
            images = torch.stack([self.transform(img) for img in images])
        res = {'images': merged_images,
            'plant_num': plant_num,  # For debugging/analysis
            'num_slices': end - start + 1  # For dynamic padding}
            }
        if self.is_train:
            res['volume'] = torch.tensor(volume, dtype=torch.float32)
        return res

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

    def verify_cropping(self, idx, save_dir='cropping_verification'):
        """
        Verifies the cropping by displaying original images and their cropped segments side by side.
        Each slice gets its own row showing the original image followed by its cropped segments.

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

        # Prepare lists to store images for plotting
        original_images = []
        all_cropped_segments = []

        # Process each slice in the sequence
        for i in range(images.shape[0]):
            # Get the RGB image (first 3 channels)
            img_np = images[i][:, :, :3].astype('uint8')
            img_pil = Image.fromarray(img_np)
            
            original_images.append(img_pil)
            
            # Crop segments using _crop_segmented
            cropped_segments = self._crop_segmented(img_pil)
            all_cropped_segments.append(cropped_segments)

        # Determine the maximum number of cropped segments per image
        max_crops = max(len(crops) for crops in all_cropped_segments) if all_cropped_segments else 0

        # Create a figure with one row per slice and enough columns for original + max crops
        num_slices = len(original_images)
        fig, axes = plt.subplots(num_slices, max_crops + 1, figsize=(15, 3 * num_slices))
        
        # If there's only one slice, axes won't be 2D, so we adjust
        if num_slices == 1:
            axes = axes.reshape(1, -1)

        # Plot each slice and its cropped segments
        for i in range(num_slices):
            # Plot original image
            axes[i, 0].imshow(original_images[i])
            axes[i, 0].set_title(f"Original Slice {start + i}")
            axes[i, 0].axis('off')

            # Plot cropped segments
            for j, segment in enumerate(all_cropped_segments[i]):
                axes[i, j+1].imshow(segment)
                axes[i, j+1].set_title(f"Crop {j+1}")
                axes[i, j+1].axis('off')

            # Hide empty subplots if this slice has fewer crops than max
            for j in range(len(all_cropped_segments[i]) + 1, max_crops + 1):
                axes[i, j].axis('off')

        plt.tight_layout()
        
        # Save the figure
        save_path = os.path.join(save_dir, f'cropping_{idx}.png')
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        plt.close(fig)


class ProcessedDataset(Dataset):
    def __init__(self, df : pd.DataFrame, transform = None, is_train = True):
        super().__init__()
        self.df = df
        self.transform = transform
        self.is_train = is_train

    def __getitem__(self, index):
        image = Image.open(self.df['segments'].iloc[index]).convert("RGB")

        if self.transform:
            image = self.transform(image)

        if self.is_train:
            label = self.df['RootVolume'].iloc[index]

            return image, torch.tensor(label, dtype=torch.float32)

        return image

    def __len__(self):
        return len(self.df)
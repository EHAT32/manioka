from model import RootVolumeNet
from dataset import RootVolumeDataset
# from dataset_sam import RootVolumeDataset
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

if __name__ == '__main__':
    params = get_params()
    dataset = RootVolumeDataset(
        csv_path='Train.csv',
        img_root='images/train/',
        target_width=params["width"],
        target_height=params["height"],
        pre_segment = False
    )
    dataset.verify_segmentation(23)
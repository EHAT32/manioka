from tqdm import tqdm
import pandas as pd
from dataset import RootVolumeDataset
from common import get_params

if __name__ == '__main__':
    params = get_params()
    # Dataset
    train_dataset = RootVolumeDataset(
        csv_path='Train.csv',
        img_root='images/train/',
        target_width=params["width"],
        target_height=params["height"]
    )
    train_df = pd.read_csv('Train.csv')
    imgs = []
    for i in tqdm(range(len(train_df)), desc="Processing train"):
        img = train_dataset[i]["images"]
        name = train_df['FolderName'][i]
        plantNum = train_df['PlantNumber'][i]
        img_name = name+"_"+str(plantNum)+".png"
        path = "processed_data/train/"+ img_name
        img.save(path)
        imgs.append(path)
    train_df["segments"] = imgs    
    train_df.to_csv("processed_train.csv")
    test_dataset = RootVolumeDataset(
        csv_path='Test.csv',
        img_root='images/test/',
        target_width=params["width"],
        target_height=params["height"],
        train=False
    )
    test_df = pd.read_csv('Test.csv')
    imgs = []
    for i in tqdm(range(len(test_df)), desc="Processing test"):
        img = test_dataset[i]["images"]
        name = test_df['FolderName'][i]
        plantNum = test_df['PlantNumber'][i]
        img_name = name+"_"+str(plantNum)+".png"
        path = "processed_data/test/"+ img_name
        img.save(path)
        imgs.append(path)
    test_df["segments"] = imgs
    test_df.to_csv("processed_test.csv")
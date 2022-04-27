from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch

class ClothingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.image_df = pd.read_csv('data/training/image_label_list.csv')
        self.transform = transform

    def __len__(self):
        return len(self.image_df)

    def __getitem__(self, index):

        #get image path
        img_path = self.image_df.iloc[index, 0]

        #get image
        img = Image.open(os.path.join(self.data_dir, img_path)).convert("RGB")

        #get cluster number
        y_label = torch.tensor(float(self.image_df.iloc[index, 1]))


        #apply training transforms for dataset
        if self.transform is not None:
            img = self.transform(img)

        return (img, y_label)
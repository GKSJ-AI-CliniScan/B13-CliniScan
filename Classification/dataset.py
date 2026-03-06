import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image

class ChestXrayDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.df = pd.read_csv(csv_file)
        self.image_dir = image_dir
        self.transform = transform
        
        # label columns (everything except image_id)
        self.label_columns = self.df.columns[1:]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        
        img_name = row["image_id"] + ".png"
        img_path = os.path.join(self.image_dir, img_name)
        image = Image.open(img_path).convert("RGB")
        
        labels = torch.tensor(row[self.label_columns].values.astype("float32"))
        
        if self.transform:
            image = self.transform(image)
            
        return image, labels
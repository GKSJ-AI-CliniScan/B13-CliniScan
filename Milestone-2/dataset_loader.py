import torch
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import os

class ChestDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        self.data = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_name = self.data.iloc[idx]['image_id'] + ".png"
        image = Image.open(os.path.join(self.img_dir, img_name)).convert("RGB")
        labels = torch.tensor(self.data.iloc[idx][1:].values.astype(float))

        if self.transform:
            image = self.transform(image)

        return image, labels
import os
import cv2
import pandas as pd
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class VinDrCXRClassificationDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None):
        """
        Original dataset implementation for AI-CliniScan classification.
        Labels are aggregated per image_id.
        """
        self.img_dir = img_dir
        self.df = pd.read_csv(csv_file)
        
        # Aggregate unique labels per image
        self.image_labels = self.df.groupby('image_id')['class_id'].apply(lambda x: list(set(x))).to_dict()
        self.image_ids = list(self.image_labels.keys())
        
        # Limit to 2000 images for speedy training
        import random
        random.seed(42)
        if len(self.image_ids) > 2000:
            self.image_ids = random.sample(self.image_ids, 2000)
            
        self.num_classes = 15 # VinDr-CXR has 14 abnormalities + 1 'No finding'
        
        if transform is None:
            self.transform = A.Compose([
                A.Resize(256, 256),
                A.Normalize(mean=(0.485,), std=(0.229,)), 
                ToTensorV2()
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        # Append .png since the 256x256 kaggle dataset contains pngs
        img_path = os.path.join(self.img_dir, img_id + '.png')
        
        # Load grayscale and convert to RGB format for ResNet
        image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
        
        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']
            
        # Multi-label one-hot encoding
        labels = self.image_labels[img_id]
        target = torch.zeros(self.num_classes, dtype=torch.float32)
        for label in labels:
            if not pd.isna(label):
                target[int(label)] = 1.0
                
        return image, target

def get_train_val_transforms():
    train_transform = A.Compose([
        A.Resize(256, 256),
        A.RandomCrop(224, 224),
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=15, p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    val_transform = A.Compose([
        A.Resize(256, 256),
        A.CenterCrop(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    return train_transform, val_transform

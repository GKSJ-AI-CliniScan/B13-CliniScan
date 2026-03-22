import os
import cv2
import torch
from torch.utils.data import Dataset
import pandas as pd

class VinDrCXRBoxesDataset(Dataset):
    def __init__(self, csv_file, image_dir, transform=None):
        self.image_dir = image_dir
        
        # Optimized: Pre-group and filter the dataframe
        df = pd.read_csv(csv_file)
        self.df = df.dropna(subset=['x_min'])
        
        # Create a dictionary for O(1) group access
        self.annotations = {img_id: group for img_id, group in self.df.groupby('image_id')}
        self.image_ids = list(self.annotations.keys())
        
        # Limit to 2000 images for speedy training
        import random
        random.seed(42)
        if len(self.image_ids) > 2000:
            self.image_ids = random.sample(self.image_ids, 2000)
            
        # Load metadata for scaling
        meta_df = pd.read_csv('./data/train_meta.csv')
        self.meta_dict = meta_df.set_index('image_id').to_dict('index')
        
        self.transform = transform

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        img_id = self.image_ids[idx]
        group = self.annotations[img_id]
        
        img_path = os.path.join(self.image_dir, f"{img_id}.png")
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        for _, row in group.iterrows():
            # Scale boxes to 256x256
            meta = self.meta_dict.get(img_id, {'dim0': 3000, 'dim1': 3000})
            orig_h, orig_w = meta['dim0'], meta['dim1']
            
            xmin = (row['x_min'] / orig_w) * 256.0
            ymin = (row['y_min'] / orig_h) * 256.0
            xmax = (row['x_max'] / orig_w) * 256.0
            ymax = (row['y_max'] / orig_h) * 256.0
            
            # Clip boxes to [0, 256]
            xmin = max(0, min(xmin, 255))
            ymin = max(0, min(ymin, 255))
            xmax = max(0, min(xmax, 256))
            ymax = max(0, min(ymax, 256))
            if xmax > xmin and ymax > ymin:
                boxes.append([xmin, ymin, xmax, ymax])
                # Shift labels by +1 because torchvision Faster R-CNN reserves 0 for background
                labels.append(int(row['class_id']) + 1)
            
        if not boxes: # Fallback for images with no valid boxes
            boxes = [[0, 0, 1, 1]]
            labels = [15] # 14 + 1 = 15 (No Finding)
            
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        
        # Area and iscrowd for COCO Evaluation formatting
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(boxes),), dtype=torch.int64)
        
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area
        target["iscrowd"] = iscrowd
        
        if self.transform:
            transformed = self.transform(image=image, bboxes=boxes, labels=labels)
            image = transformed['image']
            target['boxes'] = torch.as_tensor(transformed['bboxes'], dtype=torch.float32)
            
        # Convert image to [C, H, W] tensor formatted 0.0-1.0
        if not isinstance(image, torch.Tensor):
            image = torch.as_tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
            
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

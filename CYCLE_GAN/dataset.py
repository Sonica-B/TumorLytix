import os
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

class TrainData(Dataset):
    def __init__(self, root_healthy, root_tumor, transform=None):
        self.root_healthy = root_healthy
        self.root_tumor = root_tumor
        self.transform = transform

        self.healthy_images = [f for f in os.listdir(root_healthy) if f.endswith(("png", "jpg", "jpeg"))]
        self.tumor_images = [f for f in os.listdir(root_tumor) if f.endswith(("png", "jpg", "jpeg"))]

        self.length_dataset = max(len(self.healthy_images), len(self.tumor_images))
        self.healthy_len = len(self.healthy_images)
        self.tumor_len = len(self.tumor_images)

    def __len__(self):
        return self.length_dataset

    def __getitem__(self, index):
        healthy_img = Image.open(os.path.join(self.root_healthy, self.healthy_images[index % self.healthy_len])).convert("RGB")
        tumor_img = Image.open(os.path.join(self.root_tumor, self.tumor_images[index % self.tumor_len])).convert("RGB")

        healthy_img = np.array(healthy_img)
        tumor_img = np.array(tumor_img)

        if self.transform:
            healthy_img = self.transform(image=healthy_img)["image"]
            tumor_img = self.transform(image=tumor_img)["image"]

        return healthy_img, tumor_img

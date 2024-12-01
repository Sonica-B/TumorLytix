from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np
from torchvision import transforms
from typing import List, Optional, Tuple

class TrainData(Dataset):

    def __init__(self, 
                 root_normal: str,
                 root_abnormal: str,
                 transform: Optional[transforms.Compose] = None) -> None:

        self.root_normal: str = root_normal   # normal brain images directory
        self.root_abnormal: str = root_abnormal # abnormal brain images directory

        self.transform = transform

        self.normal_images: List[str] = os.listdir(self.root_normal)
        self.abnormal_images = os.listdir(self.root_abnormal)

        self.length_dataset = max(len(self.normal_images),len(self.abnormal_images))

        self.normal_len = len(self.normal_images)
        self.abnormal_len = len(self.abnormal_images)

    def __len__(self):

        return self.length_dataset
    

    def __getitem__(self,index):

        normal_img = self.normal_images[index % self.normal_len]
        abnormal_img = self.abnormal_images[index % self.abnormal_len]

        normal_path = os.path.join(self.root_normal,normal_img)
        abnormal_path = os.path.join(self.root_abnormal,abnormal_img)

        normal_img = Image.open(normal_path).convert('RGB')
        abnormal_img = Image.open(abnormal_path).convert('RGB')



        # to_tensor = transforms.ToTensor()  # Converts (H, W, C) in [0, 255] to (C, H, W) in [0, 1]
        # normal_img_tensor = to_tensor(normal_img)
        # abnormal_img_tensor = to_tensor(abnormal_img)

        if self.transform:
            # augmentations = self.transform(image=normal_img, image1=abnormal_img)
            # normal_img = augmentations['image']
            normal_img = self.transform(normal_img)
            abnormal_img = self.transform(abnormal_img)
            # abnormal_img = augmentations['image1']



        return normal_img,abnormal_img
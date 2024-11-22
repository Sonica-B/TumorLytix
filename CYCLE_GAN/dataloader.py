from torch.utils.data import Dataset
import os
from PIL import Image
import numpy as np

class TrainData(Dataset):

    def __init__(self,root_normal,root_abnormal,transform=None):

        self.root_normal = root_normal   # normal brain images directory
        self.root_abnormal = root_abnormal # abnormal brain images directory

        self.transform = transform

        self.normal_images = os.listdir(self.root_normal)
        self.abnormal_images = os.listdir(self.root_abnormal)

        self.length_dataset = max(len(self.normal_images),len(self.abnormal_images))

        self.normal_len = len(self.normal_images)
        self.abnormal_len = len(self.abnormal_images)

    def __len__(self):

        return self.length_dataset
    

    def __get__item(self,index):

        normal_img = self.normal_images[index % self.normal_len]
        abnormal_img = self.abnormal_images[index % self.abnormal_len]

        normal_path = os.path.join(self.root_normal,normal_img)
        abnormal_path = os.path.join(self.root_abnormal,abnormal_img)

        normal_img = np.array(Image.open(normal_path))
        abnormal_img = np.array(Image.open(abnormal_path))

        if self.transform:

            normal_img = self.transform(normal_img)
            abnormal_img = self.transform(abnormal_img)




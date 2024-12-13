import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class PairedTrainData(Dataset):
    def __init__(self, root_dir_normal, root_dir_abnormal, resize_to=256):
        self.root_dir_normal = root_dir_normal
        self.root_dir_abnormal = root_dir_abnormal
        self.normal_images = os.listdir(root_dir_normal)
        self.abnormal_images = os.listdir(root_dir_abnormal)
        self.resize_to = resize_to

        # Define transformations, including resizing
        self.transform = transforms.Compose([
            transforms.Resize((self.resize_to, self.resize_to)),  # Resize to 256×256
            transforms.ToTensor(),                               # Convert to tensor
            transforms.Normalize([0.5], [0.5])                   # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.normal_images)

    def __getitem__(self, index):
        # Load image paths
        normal_image_path = os.path.join(self.root_dir_normal, self.normal_images[index])
        abnormal_image_path = os.path.join(self.root_dir_abnormal, self.abnormal_images[index])

        # Open and convert images to grayscale
        normal_image = Image.open(normal_image_path).convert("L")
        abnormal_image = Image.open(abnormal_image_path).convert("L")

        # Apply transformations
        normal_image = self.transform(normal_image)
        abnormal_image = self.transform(abnormal_image)

        return normal_image, abnormal_image

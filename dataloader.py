import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from utils import *

class TrainData(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the image files.
            transform (callable, optional): Optional transform to be applied on the image.
        """
        self.root_dir = root_dir

        # Collect all image file paths
        self.image_paths = []
        for filename in os.listdir(root_dir):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tif')):
                self.image_paths.append(os.path.join(root_dir, filename))

        # Print diagnostic information
        print(f"Found {len(self.image_paths)} images in {root_dir}")

        # Default transform if none provided
        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.5], std=[0.5])
            ])
        else:
            self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load the image
        image_path = self.image_paths[idx]

        # Open image and convert to grayscale
        image = Image.open(image_path).convert('L')  # 'L' mode for grayscale

        # Apply transformations
        if self.transform:
            image = self.transform(image)

        # Create a dummy segmentation mask (all zeros)
        seg_mask = torch.zeros_like(image)

        return image, seg_mask


def main_step2():
    # Set random seed for reproducibility
    seed_everything()

    print("Starting Stage 2: VE-JP Diffusion Training...")

    # Load normal images
    train_dataset = TrainData(root_dir=config.train_dir_normal, transform=config.transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )
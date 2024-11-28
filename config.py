import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch
import torchvision.transforms as transforms


# Define the device for training
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Directories for training and validation data
# train_dir = "D:/WPI Assignments/TumorLytix/data/BraTS2020_TrainingData/MICCAI_BraTS2020_TrainingData"
# val_dir = "D:/WPI Assignments/TumorLytix/data/BraTS2020_ValidationData/MICCAI_BraTS2020_ValidationData"
train_dir_normal = "D:/WPI Assignments/TumorLytix/data/normal"
train_dir_abnormal = "D:/WPI Assignments/TumorLytix/data/abnormal"

# Training parameters
batch_size = 1
learning_rate = 1e-4
num_epochs = 10
num_workers = 4
load_model = True
save_model = True

# Checkpoint file names
CHECKPOINT_GEN_NORMAL = "generate_normal.pth.tar"
CHECKPOINT_GEN_ABNORMAL = "generate_abnormal.pth.tar"
CHECKPOINT_CRITIC_NORMAL = "critic_normal.pth.tar"
CHECKPOINT_CRITIC_ABNORMAL = "critic_abnormal.pth.tar"

# Image transformations for training data

# Image transformations for training data
transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5])
])

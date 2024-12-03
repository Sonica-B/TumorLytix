import albumentations as A
from albumentations.pytorch import ToTensorV2
import torch

# Device configuration
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Hyperparameters
BATCH_SIZE = 16
NUM_EPOCHS = 50
LEARNING_RATE = 1e-4
LAMBDA_CYCLE = 20.0
LAMBDA_IDENTITY = 0.5
NUM_WORKERS = 4
SAVE_MODEL = True
LOAD_MODEL = False

# Directories for data and checkpoints
TRAIN_DIR = "./input_images"
CHECKPOINT_GEN_NORMAL = "./checkpoints/gen_healthy.pth.tar"
CHECKPOINT_GEN_ABNORMAL = "./checkpoints/gen_tumor.pth.tar"
CHECKPOINT_DISC_NORMAL = "./checkpoints/disc_healthy.pth.tar"
CHECKPOINT_DISC_ABNORMAL = "./checkpoints/disc_tumor.pth.tar"

# Data augmentations
TRANSFORMS = A.Compose(
    [
        A.Resize(height=256, width=256),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ToTensorV2(),
    ]
)

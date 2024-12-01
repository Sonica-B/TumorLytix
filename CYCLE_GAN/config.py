import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.transforms import transforms

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = 'D:\Devesh\codes\DL\project\CYCLE_GAN\input_images'
val_dir = ''

batch_size = 1
learning_rate =  1e-4 # set based on paper
lambda_identity = 0.0
lambda_cycle = 10
num_workers = 4
num_epochs = 100
load_model = False
save_model = True


LAMBDA_CYCLE = 10.0
LAMBDA_IDENTITY = 5.0
CHECKPOINT_GEN_NORMAL = 'generate_healthy.pth.tar'
CHECKPOINT_GEN_ABNORMAL = 'generate_tumor.pth.tar'

CHECKPOINT_DISC_NORMAL = 'critic_healthy.pth.tar'
CHECKPOINT_DISC_ABNORMAL = 'critic_tumor.pth.tar'

transforms = transforms.Compose(
    [
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        
    ]
)

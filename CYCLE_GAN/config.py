import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

train_dir = ''
val_dir = ''

batch_size = 1
learning_rate =  1e-4 # set based on paper
lambda_identity = 0.0
lambda_cycle = 10
num_workers = 4
num_epochs = 200
load_model = True
save_model = True

CHECKPOINT_GEN_NORMAL = 'generate_normal.pth.tar'
CHECKPOINT_GEN_ABNORMAL = 'generate_abnormal.pth.tar'

CHECKPOINT_CRITIC_NORMAL = 'critic_normal.pth.tar'
CHECKPOINT_CRITIC_ABNORMAL = 'critic_abnormal.pth.tar'


transforms = A.Compose(
    [
        A.Resize(width = 256, height=256),
        A.Normalize(mean=[0.5,0.5,0.5],std = [0.5,0.5,0.5],max_pixel_value=255),
        ToTensorV2()
    ]
)
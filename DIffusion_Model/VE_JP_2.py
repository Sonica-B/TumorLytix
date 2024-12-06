import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import PairedTrainData
from utils import load_checkpoint, seed_everything
import config
from torch.utils.tensorboard import SummaryWriter

# Updated Double Convolution Block
import torch
import torch.nn as nn
import torch.nn.functional as F



# Simplified Double Convolution Block with Batch Normalization
class SimpleDoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleDoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)


# Simplified Downsampling Block
class SimpleDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleDown, self).__init__()
        self.conv = SimpleDoubleConv(in_channels, out_channels)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self, x):
        x = self.conv(x)
        x_pooled = self.pool(x)
        return x, x_pooled


# Simplified Upsampling Block with Residual Connections
class SimpleUp(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SimpleUp, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.conv = SimpleDoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        
        # Adding residual connection
        x_res = x + x1  # residual connection
        return self.conv(x_res)


# Squeeze-and-Excitation (SE) Block
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        b, c, _, _ = x.size()
        y = x.mean([2, 3], keepdim=True)  # Global Average Pooling
        y = self.fc1(y.view(b, c))
        y = self.relu(y)
        y = self.fc2(y)
        y = self.sigmoid(y)
        return x * y.view(b, c, 1, 1)


# Simplified U-Net with Residual Connections and SE Block
class SimplifiedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(SimplifiedUNet, self).__init__()
        self.down1 = SimpleDown(in_channels, 64)
        self.down2 = SimpleDown(64, 128)
        self.down3 = SimpleDown(128, 256)
        self.down4 = SimpleDown(256, 512)

        self.bottleneck = SimpleDoubleConv(512, 1024)

        self.up1 = SimpleUp(1024, 512)
        self.up2 = SimpleUp(512, 256)
        self.up3 = SimpleUp(256, 128)
        self.up4 = SimpleUp(128, 64)

        self.final = nn.Conv2d(64, out_channels, kernel_size=1)

        # Attention Mechanism (SEBlock) added after each convolution block in encoder and decoder
        self.attn1 = SEBlock(64)
        self.attn2 = SEBlock(128)
        self.attn3 = SEBlock(256)
        self.attn4 = SEBlock(512)
        self.attn5 = SEBlock(1024)

    def forward(self, x):
        x1, x1_pooled = self.down1(x)
        x2, x2_pooled = self.down2(x1_pooled)
        x3, x3_pooled = self.down3(x2_pooled)
        x4, x4_pooled = self.down4(x3_pooled)

        # Apply attention to features
        x1 = self.attn1(x1)
        x2 = self.attn2(x2)
        x3 = self.attn3(x3)
        x4 = self.attn4(x4)

        x_bottleneck = self.bottleneck(x4_pooled)

        x_bottleneck = self.attn5(x_bottleneck)  # Apply attention to bottleneck features

        x = self.up1(x_bottleneck, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)

        return self.final(x)


class EMA:
    def __init__(self, model, decay=0.999):
        self.decay = decay
        self.shadow = {name: param.clone().detach() for name, param in model.named_parameters()}

    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model):
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.data.copy_(self.shadow[name])

class VEJP_Diffusion(nn.Module):
    def __init__(self, noise_schedule, input_channels=1):
        super(VEJP_Diffusion, self).__init__()
        self.noise_schedule = noise_schedule
        self.unet = SimplifiedUNet(in_channels=input_channels + 1, out_channels=input_channels)

    def forward_process(self, xA, xB, t):
        noise = torch.randn_like(xA)
        sigma_t = self.noise_schedule[t]
        noisy_xA = xA + sigma_t * noise
        return noisy_xA, noise, xB

    def forward(self, xA, xB, t):
        noisy_xA, _, _ = self.forward_process(xA, xB, t)
        input_tensor = torch.cat([noisy_xA, xB], dim=1)
        return self.unet(input_tensor)


def train_model():
    seed_everything()

    # Dataset and DataLoader
    dataset = PairedTrainData(
        root_dir_normal=config.train_dir_normal,
        root_dir_abnormal=config.train_dir_abnormal
    )
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Define model and optimizer
    timesteps = 1000
    sigma_max = 348
    sigma_min = 0.1
    noise_schedule = torch.logspace(torch.log10(torch.tensor(sigma_min)), torch.log10(torch.tensor(sigma_max)), timesteps).to(config.DEVICE)

    model = VEJP_Diffusion(noise_schedule=noise_schedule).to(config.DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    # EMA for parameter smoothing
    ema = EMA(model)

    # Load checkpoint if available
    # if config.load_model:
    #     load_checkpoint(config.CHECKPOINT_GEN_NORMAL, model, optimizer)
    writer = SummaryWriter(log_dir='runs/vejp_diffusion')
    # Training Loop
    model.train()
    for epoch in range(config.num_epochs):
        for xA, xB in dataloader:
            xA, xB = xA.to(config.DEVICE), xB.to(config.DEVICE)
            t = torch.randint(0, timesteps, (1,)).item()
            reconstructed = model(xA, xB, t)
            loss = F.mse_loss(reconstructed, xA)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update EMA
            ema.update(model)

        # Print Epoch Loss

        avg_epoch_loss = loss / len(dataloader)

        # Log the loss to TensorBoard
        writer.add_scalar('Loss/train', avg_epoch_loss, epoch + 1)

        # Print Epoch Loss
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {avg_epoch_loss}")
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Loss: {loss.item()}")

        # Save checkpoint
        torch.save({
            "state_dict": model.state_dict(),
            "optimizer": optimizer.state_dict(),
        }, f"output_images/checkpoints/diffusion_epoch_{epoch + 1}.pth.tar")

    # Apply EMA for testing
    ema.apply_shadow(model)

    writer.close()


if __name__ == "__main__":
    train_model()
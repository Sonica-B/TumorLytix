import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataloader import PairedTrainData
from utils import load_checkpoint, save_checkpoint, seed_everything
import config
import matplotlib.pyplot as plt


# Residual Block with Skip Connections
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        identity = self.skip(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)


# Encoder Block
class Encoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Encoder, self).__init__()
        self.residual_block = ResidualBlock(in_channels, out_channels)
        self.pool = nn.MaxPool2d(2)

    def forward(self, x):
        out = self.residual_block(x)
        pooled = self.pool(out)
        return out, pooled


# Decoder Block
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        self.residual_block = ResidualBlock(in_channels, out_channels)

    def forward(self, x, skip_connection):
        x = self.up(x)
        diffY = skip_connection.size(2) - x.size(2)
        diffX = skip_connection.size(3) - x.size(3)
        x = F.pad(x, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        return self.residual_block(torch.cat([x, skip_connection], dim=1))


# Improved U-Net Architecture
class ImprovedUNet(nn.Module):
    def __init__(self, in_channels=1, out_channels=1):
        super(ImprovedUNet, self).__init__()
        self.encoder1 = Encoder(in_channels, 64)
        self.encoder2 = Encoder(64, 128)
        self.encoder3 = Encoder(128, 256)
        self.encoder4 = Encoder(256, 512)

        self.bottleneck = ResidualBlock(512, 1024)

        self.decoder1 = Decoder(1024, 512)
        self.decoder2 = Decoder(512, 256)
        self.decoder3 = Decoder(256, 128)
        self.decoder4 = Decoder(128, 64)

        self.final_conv = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        skip1, pooled1 = self.encoder1(x)
        skip2, pooled2 = self.encoder2(pooled1)
        skip3, pooled3 = self.encoder3(pooled2)
        skip4, pooled4 = self.encoder4(pooled3)

        bottleneck = self.bottleneck(pooled4)

        up1 = self.decoder1(bottleneck, skip4)
        up2 = self.decoder2(up1, skip3)
        up3 = self.decoder3(up2, skip2)
        up4 = self.decoder4(up3, skip1)

        return self.final_conv(up4)


# VE-JP Diffusion with Improved U-Net
class VEJP_Diffusion(nn.Module):
    def __init__(self, noise_schedule, input_channels=1):
        super(VEJP_Diffusion, self).__init__()
        self.noise_schedule = noise_schedule
        self.unet = ImprovedUNet(in_channels=input_channels + 1, out_channels=input_channels)

    def forward_process(self, xA, xB, t):
        noise = torch.randn_like(xA)
        alpha_t = torch.sqrt(self.noise_schedule[t])
        noisy_xA = alpha_t * xA + torch.sqrt(1 - alpha_t ** 2) * noise
        return noisy_xA, noise, xB

    def forward(self, xA, xB, t):
        noisy_xA, _, _ = self.forward_process(xA, xB, t)
        input_tensor = torch.cat([noisy_xA, xB], dim=1)
        return self.unet(input_tensor)


# Training Workflow
def main():
    seed_everything()

    # Load dataset and DataLoader
    dataset = PairedTrainData(
        root_dir_normal=config.train_dir_normal,
        root_dir_abnormal=config.train_dir_abnormal,
        resize_to=256,
    )
    print(f"Dataset loaded with size: {len(dataset)}")
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)

    # Model Initialization
    timesteps = 1000
    noise_schedule = torch.linspace(0.01, 0.999, timesteps).to(config.DEVICE)
    model = VEJP_Diffusion(noise_schedule=noise_schedule).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # Load checkpoint if available
    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_NORMAL, model, optimizer)

    # Training loop
    model.train()
    loss_values = []

    for epoch in range(config.num_epochs):
        epoch_loss = 0
        for xA, xB in dataloader:
            xA, xB = xA.to(config.DEVICE), xB.to(config.DEVICE)
            t = torch.randint(0, timesteps, (1,)).item()
            reconstructed = model(xA, xB, t)

            loss = F.mse_loss(reconstructed, xA)
            if torch.isnan(loss):
                print(f"NaN loss encountered at Epoch {epoch+1}, . Skipping...")
                continue

            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch + 1}], Loss: {loss.item():.6f}")

        avg_loss = epoch_loss / len(dataloader)
        loss_values.append(avg_loss)
        print(f"Epoch [{epoch + 1}/{config.num_epochs}], Avg Loss: {avg_loss:.6f}")

        # Save checkpoint
        save_checkpoint(model, optimizer, filename=f"output_images/diffusion_epoch_{epoch + 1}.pth.tar")

    # Plot loss graph
    plt.plot(range(1, len(loss_values) + 1), loss_values)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("output_images/loss_curve.png")
    plt.show()

if __name__ == "__main__":
    main()


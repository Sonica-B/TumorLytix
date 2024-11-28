import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import *
import config
from dataloader import TrainData
import torch.nn.functional as F


class VEJP_Diffusion(nn.Module):
    def __init__(self, noise_schedule, input_channels=1):
        super(VEJP_Diffusion, self).__init__()
        self.noise_schedule = noise_schedule
        self.unet = nn.Sequential(
            nn.Conv3d(input_channels + 1, 64, kernel_size=3, stride=1, padding=1),  # T1ce + mask
            nn.ReLU(),
            nn.Conv3d(64, input_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward_process(self, xA, xB, t):
        noise = torch.randn_like(xA)
        alpha_t = self.noise_schedule[t]
        noisy_xA = alpha_t * xA + (1 - alpha_t).sqrt() * noise
        return noisy_xA, noise, xB


def train_diffusion(model, dataloader, optimizer, noise_schedule, epochs, save_dir="output_images/"):
    # Ensure save directory exists
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        loop = tqdm(dataloader, leave=True)

        for xA, xB in loop:
            # Move data to device
            xA = xA.to(config.DEVICE)
            xB = xB.to(config.DEVICE)

            # Ensure 5D tensor shape [batch_size, channels, depth, height, width]
            if xA.dim() == 4:
                xA = xA.unsqueeze(1)
            if xB.dim() == 4:
                xB = xB.unsqueeze(1)

            # Process segmentation mask
            xB_condition = xB.float().mean(dim=1, keepdim=True)

            # Select random timestep
            t = torch.randint(0, len(noise_schedule), (1,)).item()

            # Apply forward process
            noisy_xA, true_noise, xB_condition = model.forward_process(xA, xB_condition, t)

            # Concatenate input for UNet
            input_tensor = torch.cat([noisy_xA, xB_condition], dim=1)

            # Predict noise
            predicted_noise = model.unet(input_tensor)

            # Compute loss
            loss = F.mse_loss(predicted_noise, true_noise)

            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update total loss and progress bar
            total_loss += loss.item()
            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        # Calculate average loss for the epoch
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{epochs}, Average Loss: {avg_loss:.4f}")

        # Save checkpoint after each epoch
        save_checkpoint(model, optimizer, filename=f"{save_dir}/diffusion_epoch_{epoch + 1}.pth.tar")

    print("Training complete.")

def main_step2():
    # Set random seed for reproducibility
    seed_everything()

    print("Starting Stage 2: VE-JP Diffusion Training...")

    # Load pseudo-paired data from normal directory
    train_dataset = TrainData(root_dir=config.train_dir_normal, transform=config.transforms)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    # Define diffusion model parameters
    timesteps = 1000
    noise_schedule = torch.linspace(0.01, 0.1, timesteps).to(config.DEVICE)

    # Initialize VE-JP Diffusion model
    model = VEJP_Diffusion(noise_schedule=noise_schedule, input_channels=1).to(config.DEVICE)

    # Setup optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.5, 0.999)
    )

    # Load checkpoint if available
    if config.load_model:
        load_checkpoint(
            config.CHECKPOINT_GEN_NORMAL,
            model,
            optimizer,
            config.learning_rate
        )

    # Train the diffusion model
    train_diffusion(
        model,
        train_loader,
        optimizer,
        noise_schedule,
        epochs=config.num_epochs,
        save_dir="output_images/"
    )

    print("Stage 2 Complete!")

if __name__ == "__main__":
    main_step2()
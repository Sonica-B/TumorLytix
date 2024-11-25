import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from CYCLE_GAN.dataloader import TrainData
from CYCLE_GAN.generator import Generator
from CYCLE_GAN.utils import save_checkpoint, load_checkpoint, seed_everything
from CYCLE_GAN import config


class VEJP_Diffusion(nn.Module):
    def __init__(self, noise_schedule, input_channels=3):
        super(VEJP_Diffusion, self).__init__()
        self.noise_schedule = noise_schedule  # Noise levels for forward diffusion
        self.unet = Generator(channels=input_channels)  # UNet-based generator model

    def forward_process(self, xA, xB, t):
        """Applies forward diffusion by adding noise, conditioned on xB."""
        noise = torch.randn_like(xA)
        alpha_t = self.noise_schedule[t]
        noisy_xA = alpha_t * xA + (1 - alpha_t).sqrt() * noise
        return noisy_xA, noise, xB

    def reverse_process(self, noisy_xA, xB, timesteps):
        """Reconstructs the image using reverse diffusion, guided by xB."""
        for t in reversed(range(timesteps)):
            input_combined = torch.cat([noisy_xA, xB], dim=1)
            predicted_noise = self.unet(input_combined)
            alpha_t = self.noise_schedule[t]
            noisy_xA = (noisy_xA - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        return noisy_xA


def train_diffusion(model, dataloader, optimizer, noise_schedule, epochs):
    """Training loop for VE-JP Diffusion model."""
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for xA, xB in loop:
            xA = xA.to(config.DEVICE)  # Healthy images
            xB = xB.to(config.DEVICE)  # Abnormal images

            # Randomly sample a timestep
            t = torch.randint(0, len(noise_schedule), (1,)).item()

            # Forward Diffusion Process
            noisy_xA, true_noise, xB_condition = model.forward_process(xA, xB, t)

            # Reverse Diffusion Process
            predicted_noise = model.unet(torch.cat([noisy_xA, xB_condition], dim=1))

            # Score Matching Loss
            loss = ((predicted_noise - true_noise) ** 2).mean()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Update Progress
            loop.set_description(f"Epoch [{epoch+1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        # Save checkpoint at the end of every epoch
        if config.save_model:
            save_checkpoint(model, optimizer, filename=f"diffusion_epoch_{epoch+1}.pth.tar")


def main():
    # Set seed for reproducibility
    seed_everything()

    # Noise schedule
    timesteps = 1000
    noise_schedule = torch.linspace(0.01, 0.1, timesteps).to(config.DEVICE)

    # Initialize model
    diffusion_model = VEJP_Diffusion(noise_schedule, input_channels=6).to(config.DEVICE)  # 3 (xA) + 3 (xB)

    # Optimizer
    optimizer = optim.Adam(diffusion_model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

    # Load dataset
    train_dataset = TrainData(config.train_dir, config.val_dir, config.transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    # Load checkpoint if required
    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_NORMAL, diffusion_model, optimizer, config.learning_rate)

    # Train the model
    train_diffusion(diffusion_model, train_loader, optimizer, noise_schedule, epochs=config.num_epochs)


if __name__ == "__main__":
    main()

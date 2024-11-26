import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from utils import save_checkpoint, load_checkpoint
from dataloader import TrainData
import config


class VEJP_Diffusion(nn.Module):
    def __init__(self, noise_schedule, input_channels=4):
        super(VEJP_Diffusion, self).__init__()
        self.noise_schedule = noise_schedule
        self.unet = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, input_channels, kernel_size=3, padding=1),
        )

    def forward_process(self, xA, xB, t):
        noise = torch.randn_like(xA)
        alpha_t = self.noise_schedule[t]
        noisy_xA = alpha_t * xA + (1 - alpha_t).sqrt() * noise
        return noisy_xA, noise, xB

    def reverse_process(self, noisy_xA, xB, timesteps):
        for t in reversed(range(timesteps)):
            input_combined = torch.cat([noisy_xA, xB], dim=1)
            predicted_noise = self.unet(input_combined)
            alpha_t = self.noise_schedule[t]
            noisy_xA = (noisy_xA - (1 - alpha_t).sqrt() * predicted_noise) / alpha_t.sqrt()
        return noisy_xA


def train_diffusion(model, dataloader, optimizer, noise_schedule, epochs, save_dir='output_images/'):
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        for xA, xB in loop:
            xA = xA.to(config.DEVICE)  # Healthy images
            xB = xB.to(config.DEVICE)  # Abnormal images

            t = torch.randint(0, len(noise_schedule), (1,)).item()

            noisy_xA, true_noise, xB_condition = model.forward_process(xA, xB, t)

            predicted_noise = model.unet(torch.cat([noisy_xA, xB_condition], dim=1))

            loss = ((predicted_noise - true_noise) ** 2).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item())

        if config.save_model:
            save_checkpoint(model, optimizer, filename=f"diffusion_epoch_{epoch + 1}.pth.tar")


def main_step2():
    train_dataset = TrainData(root_dir=config.train_dir, transform=config.transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    timesteps = 1000
    noise_schedule = torch.linspace(0.01, 0.1, timesteps).to(config.DEVICE)

    model = VEJP_Diffusion(noise_schedule=noise_schedule, input_channels=4).to(config.DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_NORMAL, model, optimizer, config.learning_rate)

    train_diffusion(model, train_loader, optimizer, noise_schedule, epochs=config.num_epochs, save_dir="output_images/")

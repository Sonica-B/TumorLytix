import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from dataloader import TrainData
from CYCLE_GAN.generator import Generator
from utils import save_checkpoint, load_checkpoint, seed_everything
import config
from config import train_dir


class VEJP_Diffusion(nn.Module):
    def __init__(self, noise_schedule, input_channels=3):
        super(VEJP_Diffusion, self).__init__()
        self.noise_schedule = noise_schedule
        self.unet = Generator(channels=input_channels)

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


def calculate_metrics(original, reconstructed):
    """Calculate PSNR and SSIM between the original and reconstructed image."""
    original = original.cpu().detach().numpy()
    reconstructed = reconstructed.cpu().detach().numpy()

    psnr_value = psnr(original, reconstructed)
    ssim_value = ssim(original, reconstructed, multichannel=True)
    return psnr_value, ssim_value


def visualize_output(epoch, original, reconstructed, save_dir='output_images/'):
    """Visualize and save output images for inspection."""
    original = original.cpu().detach().numpy().squeeze()
    reconstructed = reconstructed.cpu().detach().numpy().squeeze()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(original, cmap='gray')
    axes[0].set_title(f'Epoch {epoch} - Original')
    axes[1].imshow(reconstructed, cmap='gray')
    axes[1].set_title(f'Epoch {epoch} - Reconstructed')
    for ax in axes:
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f"{save_dir}/epoch_{epoch}.png")
    plt.show()


def train_diffusion(model, dataloader, optimizer, noise_schedule, epochs, save_dir='output_images/'):
    
    """Training loop with progress visualization."""
    for epoch in range(epochs):
        loop = tqdm(dataloader, leave=True)
        total_psnr = 0
        total_ssim = 0
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

            # Calculate and track PSNR and SSIM
            psnr_value, ssim_value = calculate_metrics(xA, predicted_noise)
            total_psnr += psnr_value
            total_ssim += ssim_value

            loop.set_description(f"Epoch [{epoch + 1}/{epochs}]")
            loop.set_postfix(loss=loss.item(), psnr=psnr_value, ssim=ssim_value)

        # Calculate average PSNR and SSIM for the epoch
        avg_psnr = total_psnr / len(dataloader)
        avg_ssim = total_ssim / len(dataloader)
        print(f"Epoch {epoch + 1} - Avg PSNR: {avg_psnr:.4f}, Avg SSIM: {avg_ssim:.4f}")

        # Save and visualize outputs at regular intervals
        if (epoch + 1) % 10 == 0:  # Save and visualize every 10 epochs
            visualize_output(epoch + 1, xA[0], predicted_noise[0], save_dir)

        if config.save_model:
            save_checkpoint(model, optimizer, filename=f"diffusion_epoch_{epoch + 1}.pth.tar")


def main():
    # Initialize dataset
    train_dataset = TrainData(train_dir, transform=config.transforms)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers
    )

    seed_everything()

    timesteps = 1000
    noise_schedule = torch.linspace(0.01, 0.1, timesteps).to(config.DEVICE)

    diffusion_model = VEJP_Diffusion(noise_schedule, input_channels=6).to(config.DEVICE)  # 3 (xA) + 3 (xB)

    optimizer = optim.Adam(diffusion_model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))


    # train_dataset = TrainData(config.train_dir, config.val_dir, config.transforms)
    # train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers,
    #                           pin_memory=True)

    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_NORMAL, diffusion_model, optimizer, config.learning_rate)

    train_diffusion(diffusion_model, train_loader, optimizer, noise_schedule, epochs=config.num_epochs)


if __name__ == "__main__":
    main()

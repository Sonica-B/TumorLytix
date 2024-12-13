import torch
import torch.nn as nn
import torch.optim as optim
import torch.amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils

from config import DEVICE, CHECKPOINT_GEN_NORMAL, CHECKPOINT_GEN_ABNORMAL, CHECKPOINT_DISC_NORMAL, CHECKPOINT_DISC_ABNORMAL, TRANSFORMS, BATCH_SIZE, NUM_EPOCHS, NUM_WORKERS, LAMBDA_CYCLE, LAMBDA_IDENTITY, SAVE_MODEL, LOAD_MODEL, TRAIN_DIR
from generator import Generator
from discriminator import PatchGANDiscriminator
from dataset import TrainData
from utils import save_checkpoint, load_checkpoint


# Initialize TensorBoard writer
writer = SummaryWriter(log_dir="runs/cycle_gan_brain_tumor")
print(f"Using device: {DEVICE}")


def train_step(
    healthy_images: torch.Tensor,
    tumor_images: torch.Tensor,
    disc_healthy: nn.Module,
    disc_tumor: nn.Module,
    gen_healthy: nn.Module,
    gen_tumor: nn.Module,
    opt_gen: optim.Optimizer,
    opt_disc: optim.Optimizer,
    l1_loss: nn.L1Loss,
    mse_loss: nn.MSELoss,
    gen_scaler: torch.cuda.amp.GradScaler,
    disc_scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    batch_idx: int,
) -> tuple[float, float]:
    """Executes one training step for the CycleGAN model."""
    healthy_images = healthy_images.to(DEVICE)
    tumor_images = tumor_images.to(DEVICE)

    # Update Discriminators
    with torch.amp.autocast(device_type="cuda"):
        fake_tumor = gen_tumor(healthy_images)
        real_tumor_loss = mse_loss(disc_tumor(tumor_images), torch.ones_like(disc_tumor(tumor_images)))
        fake_tumor_loss = mse_loss(disc_tumor(fake_tumor.detach()), torch.zeros_like(disc_tumor(fake_tumor)))
        disc_tumor_loss = real_tumor_loss + fake_tumor_loss

        fake_healthy = gen_healthy(tumor_images)
        real_healthy_loss = mse_loss(disc_healthy(healthy_images), torch.ones_like(disc_healthy(healthy_images)))
        fake_healthy_loss = mse_loss(disc_healthy(fake_healthy.detach()), torch.zeros_like(disc_healthy(fake_healthy)))
        disc_healthy_loss = real_healthy_loss + fake_healthy_loss

        total_disc_loss = (disc_tumor_loss + disc_healthy_loss) / 2

    opt_disc.zero_grad()
    disc_scaler.scale(total_disc_loss).backward()
    disc_scaler.step(opt_disc)
    disc_scaler.update()

    # Update Generators
    with torch.amp.autocast(device_type="cuda"):
        fake_tumor_pred = disc_tumor(fake_tumor)
        fake_healthy_pred = disc_healthy(fake_healthy)
        adversarial_loss_tumor = mse_loss(fake_tumor_pred, torch.ones_like(fake_tumor_pred))
        adversarial_loss_healthy = mse_loss(fake_healthy_pred, torch.ones_like(fake_healthy_pred))

        # Cycle consistency loss
        cycle_healthy_loss = l1_loss(healthy_images, gen_healthy(fake_tumor))
        cycle_tumor_loss = l1_loss(tumor_images, gen_tumor(fake_healthy))

        # Identity loss
        identity_healthy_loss = l1_loss(healthy_images, gen_healthy(healthy_images))
        identity_tumor_loss = l1_loss(tumor_images, gen_tumor(tumor_images))

        total_gen_loss = (
            adversarial_loss_tumor
            + adversarial_loss_healthy
            + LAMBDA_CYCLE * (cycle_healthy_loss + cycle_tumor_loss)
            + LAMBDA_IDENTITY * (identity_healthy_loss + identity_tumor_loss)
        )

    opt_gen.zero_grad()
    gen_scaler.scale(total_gen_loss).backward()
    gen_scaler.step(opt_gen)
    gen_scaler.update()

    # Save generated images only on every 10th epoch
    if epoch % 10 == 0:
        save_dir = Path("./saved_images")
        save_dir.mkdir(exist_ok=True)

        # Save each image with a unique filename
        for i in range(healthy_images.size(0)):
            vutils.save_image(
                fake_tumor[i],
                save_dir / f"tumor/fake_tumor_epoch{epoch}_batch{batch_idx}_img{i}.png",
                normalize=True,
            )
            vutils.save_image(
                fake_healthy[i],
                save_dir / f"healthy/fake_healthy_epoch{epoch}_batch{batch_idx}_img{i}.png",
                normalize=True,
            )

    return total_disc_loss.item(), total_gen_loss.item()

def train_model(
    disc_healthy: nn.Module,
    disc_tumor: nn.Module,
    gen_healthy: nn.Module,
    gen_tumor: nn.Module,
    train_loader: DataLoader,
    opt_gen: optim.Optimizer,
    opt_disc: optim.Optimizer,
    scheduler_gen: optim.lr_scheduler._LRScheduler,
    scheduler_disc: optim.lr_scheduler._LRScheduler,
    l1_loss: nn.L1Loss,
    mse_loss: nn.MSELoss,
    gen_scaler: torch.cuda.amp.GradScaler,
    disc_scaler: torch.cuda.amp.GradScaler,
) -> None:
    """Main training loop for the CycleGAN model."""
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(train_loader, leave=True)
        for batch_idx, (healthy, tumor) in enumerate(loop):
            disc_loss, gen_loss = train_step(
                healthy, tumor, disc_healthy, disc_tumor, gen_healthy, gen_tumor,
                opt_gen, opt_disc, l1_loss, mse_loss, gen_scaler, disc_scaler,
                epoch, batch_idx
            )

            # Update tqdm progress bar
            loop.set_description(f"Epoch [{epoch}/{NUM_EPOCHS}]")
            loop.set_postfix(Disc_Loss=disc_loss, Gen_Loss=gen_loss)

            # Log metrics to TensorBoard
            writer.add_scalar("Loss/Discriminator", disc_loss, epoch * len(train_loader) + batch_idx)
            writer.add_scalar("Loss/Generator", gen_loss, epoch * len(train_loader) + batch_idx)

        scheduler_gen.step()
        scheduler_disc.step()

        if SAVE_MODEL and epoch % 10 == 0:
            save_checkpoint(gen_healthy, opt_gen, CHECKPOINT_GEN_NORMAL)
            save_checkpoint(gen_tumor, opt_gen, CHECKPOINT_GEN_ABNORMAL)
            save_checkpoint(disc_healthy, opt_disc, CHECKPOINT_DISC_NORMAL)
            save_checkpoint(disc_tumor, opt_disc, CHECKPOINT_DISC_ABNORMAL)




def main():
    # Initialize models
    disc_healthy = PatchGANDiscriminator(in_channels=3).to(DEVICE)
    disc_tumor = PatchGANDiscriminator(in_channels=3).to(DEVICE)
    gen_healthy = Generator(channels=3, num_residual_blocks=9).to(DEVICE)
    gen_tumor = Generator(channels=3, num_residual_blocks=9).to(DEVICE)

    # Optimizers and schedulers
    opt_disc = optim.Adam(list(disc_healthy.parameters()) + list(disc_tumor.parameters()), lr=1e-4, betas=(0.5, 0.999))
    opt_gen = optim.Adam(list(gen_healthy.parameters()) + list(gen_tumor.parameters()), lr=1e-4, betas=(0.5, 0.999))
    scheduler_gen = optim.lr_scheduler.StepLR(opt_gen, step_size=100, gamma=0.5)
    scheduler_disc = optim.lr_scheduler.StepLR(opt_disc, step_size=100, gamma=0.5)

    # Loss functions
    l1_loss = nn.L1Loss()
    mse_loss = nn.MSELoss()

    # Load models if required
    if LOAD_MODEL:
        load_checkpoint(CHECKPOINT_GEN_NORMAL, gen_healthy, opt_gen, 1e-4)
        load_checkpoint(CHECKPOINT_GEN_ABNORMAL, gen_tumor, opt_gen, 1e-4)
        load_checkpoint(CHECKPOINT_DISC_NORMAL, disc_healthy, opt_disc, 1e-4)
        load_checkpoint(CHECKPOINT_DISC_ABNORMAL, disc_tumor, opt_disc, 1e-4)

    # Dataset and DataLoader
    train_dataset = TrainData(f"{TRAIN_DIR}/healthy", f"{TRAIN_DIR}/tumor", TRANSFORMS)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)

    # Gradient scalers
    gen_scaler = torch.cuda.amp.GradScaler()
    disc_scaler = torch.cuda.amp.GradScaler()

    # Train the model
    train_model(
        disc_healthy, disc_tumor, gen_healthy, gen_tumor, train_loader, opt_gen, opt_disc, scheduler_gen, scheduler_disc, l1_loss, mse_loss, gen_scaler, disc_scaler
    )


if __name__ == "__main__":
    main()

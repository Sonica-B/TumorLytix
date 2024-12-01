import torch
import torch.amp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
import torchvision.utils as vutils
from pathlib import Path
from generator import Generator
from discriminator import Discriminator
from dataset import TrainData
from utils import save_checkpoint, load_checkpoint
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter(log_dir="runs/cycle_gan_brain_tumor")

print(config.DEVICE)

def train_network(
    epoch: int,
    disc_healthy: nn.Module,
    disc_tumor: nn.Module,
    gen_healthy: nn.Module,
    gen_tumor: nn.Module,
    loader: DataLoader,
    opt_gen: optim.Optimizer,
    opt_disc: optim.Optimizer,
    l1: nn.L1Loss,
    mse: nn.MSELoss,
    gen_sc: torch.cuda.amp.GradScaler,
    disc_sc: torch.cuda.amp.GradScaler
) -> None:
    loop = tqdm(loader, leave=True)

    for index, (healthy, tumor) in enumerate(loop):
        healthy = healthy.to(config.DEVICE)
        tumor = tumor.to(config.DEVICE)

        with torch.cuda.amp.autocast():  # Discriminator update
            fake_tumor = gen_tumor(healthy)
            d_tumor_real = disc_tumor(tumor)
            d_tumor_fake = disc_tumor(fake_tumor.detach())

            d_tumor_real_loss = mse(d_tumor_real, torch.ones_like(d_tumor_real))
            d_tumor_fake_loss = mse(d_tumor_fake, torch.zeros_like(d_tumor_fake))
            d_tumor_loss = d_tumor_real_loss + d_tumor_fake_loss

            fake_healthy = gen_healthy(tumor)
            d_healthy_real = disc_healthy(healthy)
            d_healthy_fake = disc_healthy(fake_healthy.detach())

            d_healthy_real_loss = mse(d_healthy_real, torch.ones_like(d_healthy_real))
            d_healthy_fake_loss = mse(d_healthy_fake, torch.zeros_like(d_healthy_fake))
            d_healthy_loss = d_healthy_real_loss + d_healthy_fake_loss

            Discriminator_Loss = (d_healthy_loss + d_tumor_loss) / 2

        opt_disc.zero_grad()
        disc_sc.scale(Discriminator_Loss).backward()
        disc_sc.step(opt_disc)
        disc_sc.update()

        with torch.cuda.amp.autocast():  # Generator update
            d_tumor_fake = disc_tumor(fake_tumor)
            d_healthy_fake = disc_healthy(fake_healthy)

            loss_gen_healthy = mse(d_healthy_fake, torch.ones_like(d_healthy_fake))
            loss_gen_tumor = mse(d_tumor_fake, torch.ones_like(d_tumor_fake))

            cycle_healthy = gen_healthy(fake_tumor)
            cycle_tumor = gen_tumor(fake_healthy)

            cycle_healthy_loss = l1(healthy, cycle_healthy)
            cycle_tumor_loss = l1(tumor, cycle_tumor)

            identity_healthy_loss = l1(healthy, gen_healthy(healthy))
            identity_tumor_loss = l1(tumor, gen_tumor(tumor))

            Generator_Loss = (
                loss_gen_healthy
                + loss_gen_tumor
                + config.LAMBDA_CYCLE * (cycle_healthy_loss + cycle_tumor_loss)
                + config.LAMBDA_IDENTITY * (identity_healthy_loss + identity_tumor_loss)
            )

        opt_gen.zero_grad()
        gen_sc.scale(Generator_Loss).backward()
        gen_sc.step(opt_gen)
        gen_sc.update()

        # Log metrics to TensorBoard
        writer.add_scalar("Loss/Discriminator", Discriminator_Loss.item(), index + epoch * len(loader))
        writer.add_scalar("Loss/Generator", Generator_Loss.item(), index + epoch * len(loader))
        writer.add_images("Images/Healthy_Input", healthy, index + epoch * len(loader))
        writer.add_images("Images/Generated_Tumor", fake_tumor, index + epoch * len(loader))
        writer.add_images("Images/Tumor_Input", tumor, index + epoch * len(loader))
        writer.add_images("Images/Generated_Healthy", fake_healthy, index + epoch * len(loader))

        if epoch == config.num_epochs - 1:
            save_dir = Path("saved_images")
            save_dir.mkdir(exist_ok=True)

            vutils.save_image(healthy, save_dir / f"healthy_input_{index}.png", normalize=True)
            vutils.save_image(fake_tumor, save_dir / f"generated_tumor_{index}.png", normalize=True)
            vutils.save_image(tumor, save_dir / f"tumor_input_{index}.png", normalize=True)
            vutils.save_image(fake_healthy, save_dir / f"generated_healthy_{index}.png", normalize=True)


def main() -> None:
    discriminator_normal = Discriminator(in_channels=3).to(config.DEVICE)
    discriminator_abnormal = Discriminator(in_channels=3).to(config.DEVICE)

    generator_normal = Generator(channels=3, num_residual_blocks=9).to(config.DEVICE)
    generator_abnormal = Generator(channels=3, num_residual_blocks=9).to(config.DEVICE)

    # take values from paper
    optimizer_discriminator = optim.Adam(
        list(discriminator_normal.parameters()) + list(discriminator_abnormal.parameters()), 
        lr=config.learning_rate, 
        betas=(0.5, 0.999)
    )

    optimizer_generator = optim.Adam(
        list(generator_normal.parameters()) + list(generator_abnormal.parameters()), 
        lr=config.learning_rate, 
        betas=(0.5, 0.999)
    )

    L1 = nn.L1Loss()  # for cycle consistency loss --> check paper
    MSE = nn.MSELoss()  # for individual adversarial loss --> check paper

    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_NORMAL, generator_normal, optimizer_generator, config.learning_rate)
        load_checkpoint(config.CHECKPOINT_GEN_ABNORMAL, generator_abnormal, optimizer_generator, config.learning_rate)
        load_checkpoint(config.CHECKPOINT_DISC_NORMAL, discriminator_normal, optimizer_discriminator, config.learning_rate)
        load_checkpoint(config.CHECKPOINT_DISC_ABNORMAL, discriminator_abnormal, optimizer_discriminator, config.learning_rate)

    # add path here for healthy and tumor images 
    train_dataset = TrainData(config.train_dir + '\\healthy', config.train_dir + '\\tumor', config.transforms)

    train_loader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

    generator_scaler = torch.cuda.amp.GradScaler()  # controls gradient underflow during backpropogations
    discriminator_scaler = torch.cuda.amp.GradScaler()

    for epoch in range(config.num_epochs):
        train_network(
            epoch,
            discriminator_normal,
            discriminator_abnormal,
            generator_normal,
            generator_abnormal,
            train_loader,
            optimizer_discriminator,
            optimizer_discriminator,
            L1,
            MSE,
            generator_scaler,
            discriminator_scaler
        )
    if config.save_model:
        save_checkpoint(generator_normal, optimizer_generator, filename=config.CHECKPOINT_GEN_NORMAL)
        save_checkpoint(generator_abnormal, optimizer_generator, filename=config.CHECKPOINT_GEN_ABNORMAL)
        save_checkpoint(discriminator_normal, optimizer_discriminator, filename=config.CHECKPOINT_DISC_NORMAL)
        save_checkpoint(discriminator_abnormal, optimizer_discriminator, filename=config.CHECKPOINT_DISC_ABNORMAL)

if __name__ == '__main__':
    main()
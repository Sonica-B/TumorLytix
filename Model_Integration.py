import torch
import torch.amp
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


import config
# from CYCLE_GAN.dataloader import TrainData
from CYCLE_GAN.discriminator import Discriminator
from CYCLE_GAN.generator import Generator
from CYCLE_GAN.train import train_network
from DIffusion_Model.VE_JP import VEJP_Diffusion, train_diffusion
from DIffusion_Model.dataloader import TrainData
from segmentation import visualize_results, apply_segmentation
from utils import load_checkpoint


def stage1_cyclegan():
    """Stage 1: Train CycleGAN for pseudo-paired generation."""
    print("Starting Stage 1: CycleGAN Training...")

    discriminator_normal = Discriminator(in_channels=3).to(config.DEVICE)
    discriminator_abnormal = Discriminator(in_channels=3).to(config.DEVICE)

    generator_normal = Generator(channels=3,num_residual_blocks=9).to(config.DEVICE)
    generator_abnormal = Generator(channels=3,num_residual_blocks=9).to(config.DEVICE)

    # take values from paper
    optimizer_discriminator = optim.Adam(list(discriminator_normal.parameters()) + list(discriminator_abnormal.parameters()), lr = config.learning_rate, betas=(0.5,0.999))

    optimizer_generator = optim.Adam(list(generator_normal.parameters()) + list(generator_abnormal.parameters()), lr = config.learning_rate, betas=(0.5,0.999))

    L1 = nn.L1Loss()  # for cycle consistency loss --> check paper
    MSE = nn.MSELoss()   # for individual adversarial loss --> check paper

    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_NORMAL,generator_normal,optimizer_generator,config.learning_rate)
        load_checkpoint(config.CHECKPOINT_GEN_ABNORMAL,generator_abnormal,optimizer_generator,config.learning_rate)
        load_checkpoint(config.CHECKPOINT_CRITIC_NORMAL,discriminator_normal,optimizer_discriminator,config.learning_rate)
        load_checkpoint(config.CHECKPOINT_CRITIC_ABNORMAL,discriminator_abnormal,optimizer_discriminator,config.learning_rate)


    train_dataset = TrainData(
        root_dir=config.train_dir,
        transform=config.transforms
    )



    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    generator_scaler = torch.cuda.amp.GradScaler()  # Controls gradient underflow during backpropagation

    discriminator_scaler = torch.cuda.amp.GradScaler()


    for epoch in range(config.num_epochs):
        train_network(discriminator_normal,discriminator_abnormal,generator_normal,generator_abnormal,train_loader,optimizer_discriminator,optimizer_discriminator,L1,MSE,generator_scaler,discriminator_scaler)

    print("Stage 1 Complete!")


def stage2_diffusion():
    """Stage 2: Train VE-JP Diffusion Model."""
    print("Starting Stage 2: VE-JP Diffusion Training...")

    # Load pseudo-paired data generated by CycleGAN
    train_dataset = TrainData(root_dir=config.train_dir, transform=config.transforms)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    # Define diffusion model parameters
    timesteps = 1000
    noise_schedule = torch.linspace(0.01, 0.1, timesteps).to(config.DEVICE)

    # Initialize VE-JP Diffusion model
    model = VEJP_Diffusion(noise_schedule=noise_schedule, input_channels=2).to(config.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate, betas=(0.5, 0.999))

    # Load checkpoint if available
    if config.load_model:
        load_checkpoint(config.CHECKPOINT_GEN_NORMAL, model, optimizer, config.learning_rate)

    # Train the diffusion model
    train_diffusion(model, train_loader, optimizer, noise_schedule, epochs=config.num_epochs, save_dir="output_images/")
    print("Stage 2 Complete!")


def stage3_segmentation():
    """Stage 3: Single Modality (T1ce) Segmentation."""
    print("Starting Stage 3: T1ce Segmentation...")

    # Example T1ce data (replace with real data)
    t1ce = torch.randn(64, 256, 256)  # Example 3D T1ce volume [Depth, Height, Width]

    # Placeholder reconstructed data (replace with real reconstruction output)
    reconstructed = t1ce + torch.randn_like(t1ce) * 0.1  # Simulate reconstructed image with slight noise

    # Anomaly Detection and Segmentation
    anomaly_map, mask = apply_segmentation(reconstructed.numpy(), t1ce.numpy())

    # Visualization
    visualize_results(
        t1ce.numpy(),
        reconstructed.numpy(),
        anomaly_map,
        mask,
        epoch=1,
        save_dir="output_images/"
    )
    print("Stage 3 Complete!")


def main():
    """Main workflow for integrating all stages."""
    # Run Stage 1: Train CycleGAN
    if config.run_stage1:
        stage1_cyclegan()

    # Run Stage 2: Train VE-JP Diffusion
    if config.run_stage2:
        stage2_diffusion()

    # Run Stage 3: Multi-Modality Ensemble and Segmentation
    if config.run_stage3:
        stage3_segmentation()


if __name__ == "__main__":
    main()

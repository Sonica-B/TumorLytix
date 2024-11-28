import torch
import torch.amp
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import config
from torchvision.utils import save_image
from generator import Generator
from discriminator import Discriminator
from dataset import TrainData
from utils import save_checkpoint, load_checkpoint


def train_network(epoch,disc_healthy,disc_tumor,gen_healthy,gen_tumor,loader,opt_gen,opt_disc,l1,mse,gen_sc,disc_sc):
      
    loop = tqdm(loader,leave=True)

    for index,(healthy,tumor) in enumerate(loop):
        healthy = healthy.to(config.DEVICE)
        tumor = tumor.to(config.DEVICE)

        with torch.cuda.amp.autocast():  # for discriminator

            fake_tumor = gen_tumor(healthy)  # generate tumor image from healthy brain scan 

            d_tumor_real = disc_tumor(tumor)  
            d_tumor_fake = disc_tumor(fake_tumor.detach())

            d_tumor_real_loss = mse(d_tumor_real,torch.ones_like(d_tumor_real)) # identity loss
            d_tumor_fake_loss = mse(d_tumor_fake,torch.zeros_like(d_tumor_fake))

            d_tumor_loss = d_tumor_fake_loss + d_tumor_real_loss

            fake_healthy = gen_healthy(tumor)
            d_healthy_real = disc_healthy(healthy)
            d_healthy_fake = disc_healthy(fake_healthy.detach())

            d_healthy_real_loss = mse(d_healthy_real,torch.ones_like(d_healthy_real))
            d_healthy_fake_loss = mse(d_healthy_fake,torch.zeros_like(d_healthy_fake))

            d_healthy_loss = d_healthy_fake_loss + d_healthy_real_loss

            Discriminator_Loss = (d_healthy_loss + d_tumor_loss) / 2

        opt_disc.zero_grad()
        disc_sc.scale(Discriminator_Loss).backward()
        disc_sc.step(opt_disc)
        disc_sc.update()


        with torch.cuda.amp.autocast():  # for generator

            d_tumor_fake = disc_tumor(fake_tumor)
            d_healthy_fake = disc_healthy(fake_healthy)

            loss_gen_healthy = mse(d_healthy_fake,torch.ones_like(d_healthy_fake))
            loss_gen_tumor = mse(d_tumor_fake,torch.ones_like(d_tumor_fake))

            # cycle loss

            cycle_healthy = gen_healthy(fake_tumor)
            cycle_tumor = gen_tumor(fake_healthy)

            cycle_healthy_loss = l1(healthy,cycle_healthy)
            cycle_tumor_loss = l1(tumor,cycle_tumor)

            # identity loss

            identity_healthy = gen_healthy(healthy)
            identity_tumor = gen_tumor(tumor)

            identity_healthy_loss = l1(healthy,identity_healthy)
            identity_tumor_loss = l1(tumor,identity_tumor)

            Generator_Loss = (loss_gen_healthy + loss_gen_tumor + cycle_healthy_loss * config.lambda_cycle + cycle_tumor_loss * config.lambda_cycle + identity_healthy_loss * config.lambda_identity + identity_tumor_loss * config.lambda_identity)

        opt_gen.zero_grad()
        gen_sc.scale(Generator_Loss).backward()
        gen_sc.step(opt_gen)
        gen_sc.update()
        

        if index % 5 == 0:
                save_image(fake_healthy * 0.5 + 0.5, f"saved_images/healthy/healthy_epoch_{epoch}_{index}.png")
                save_image(fake_tumor * 0.5 + 0.5, f"saved_images/tumor/tumor_epoch_{epoch}_{index}.png")

    pass

def main():

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
        load_checkpoint(config.CHECKPOINT_DISC_NORMAL,discriminator_normal,optimizer_discriminator,config.learning_rate)
        load_checkpoint(config.CHECKPOINT_DISC_ABNORMAL,discriminator_abnormal,optimizer_discriminator,config.learning_rate)

    # add path here for healthy and tumor images 
    train_dataset = TrainData(config.train_dir + '\\healthy' ,config.train_dir + '\\tumor',config.transforms)

    train_loader = DataLoader(train_dataset,config.batch_size,shuffle=True,num_workers=config.num_workers,pin_memory=True)

    generator_scaler = torch.cuda.amp.GradScaler()  # controls gradient underflow during backpropogations
    discriminator_scaler = torch.cuda.amp.GradScaler()


    for epoch in range(config.num_epochs):
        train_network(epoch,discriminator_normal,discriminator_abnormal,generator_normal,generator_abnormal,train_loader,optimizer_discriminator,optimizer_discriminator,L1,MSE,generator_scaler,discriminator_scaler)
    if config.save_model:
        save_checkpoint(generator_normal,optimizer_generator,filename=config.CHECKPOINT_GEN_NORMAL)
        save_checkpoint(generator_abnormal,optimizer_generator,filename=config.CHECKPOINT_GEN_ABNORMAL)
        save_checkpoint(discriminator_normal,optimizer_discriminator,filename=config.CHECKPOINT_DISC_NORMAL)
        save_checkpoint(discriminator_abnormal,optimizer_discriminator,filename=config.CHECKPOINT_DISC_ABNORMAL)

if __name__ == '__main__':
    main()
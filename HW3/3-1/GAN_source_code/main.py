from __future__ import print_function
# %matplotlib inline
import argparse
import os
import random
import torch
from torch.nn import functional as F
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import pickle
import argparse
import logging
import time
from IPython.display import HTML
from Model import Generator, Discriminator, nc, nz, ngf, ndf

# Set random seed for reproducibility
manualSeed = 999
# manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

ngpu = 1
device = torch.device("cuda:0" if (
    torch.cuda.is_available() and ngpu > 0) else "cpu")

WIDTH = 218
HEIGHT = 178


def common_arg_parser():
    """ get args """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dataroot', default='data',
                        type=str, help='Root directory of images.')
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_epochs', default=5, type=int)
    parser.add_argument('--lr', default=0.0002, type=float,
                        help='Learning rate (default: 0.0002)')

    return parser.parse_args()


def get_logger(log_dir='log'):
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format='%(asctime)s [%(levelname)s:%(name)s]: %(message)s', level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    log_file_name = time.strftime("%Y%m%d-%H%M%S") + '.log'
    file_handler = logging.FileHandler(os.path.join(log_dir, log_file_name))
    formatter = logging.Formatter(
        '%(asctime)s : %(levelname)s : %(name)s : %(message)s')
    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    return logger


def train(dataloader, generator, discriminator, optimizer_g, optimizer_d, criterion, num_epochs, logger):
    # Each epoch, we have to go through every data in dataset
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0
    real_label = 1
    fake_label = 0
    fixed_noise = torch.randn(64, nz, 1, 1, device=device)
    for epoch in range(num_epochs):
        epoch_G_losses = []
        epoch_D_losses = []
        # Each iteration, we will get a batch data for training
        for i, data in enumerate(dataloader, 0):

            # initialize gradient for network
            # send the data into device for computation
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            b_size = real_cpu.size(0)
            label = torch.full((b_size,), real_label, device=device)
            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            output = discriminator(real_cpu).view(-1)
            errD_real = criterion(output, label)
            errD_real.backward()
            D_x = output.mean().item()
            # Using Fake data, other steps are the same.
            # Generate a batch fake data by using generator
            noise = torch.randn(b_size, nz, 1, 1, device=device)
            fake = generator(noise)
            label.fill_(fake_label)
            # Send data to discriminator and calculate the loss and gradient
            # For calculate loss, you need to create label for your data
            output = discriminator(fake.detach()).view(-1)
            errD_fake = criterion(output, label)
            errD_fake.backward()
            D_G_z1 = output.mean().item()
            errD = errD_real + errD_fake
            optimizer_d.step()

            # Update your network
            generator.zero_grad()
            label.fill_(real_label)
            output = discriminator(fake).view(-1)
            # Record your loss every iteration for visualization
            errG = criterion(output, label)
            errG.backward()
            D_G_z2 = output.mean().item()
            optimizer_g.step()

            # Use this function to output training procedure while training
            # You can also use this function to save models and samples after fixed number of iteration
            # if i % 50 == 0:
            #     print(.....)
            if i % 50 == 0:
                # print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                #       % (epoch, num_epochs, i, len(dataloader),
                #          errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))
                logger.info('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                            % (epoch, num_epochs, i, len(dataloader),
                               errD.item(), errG.item(), D_x, D_G_z1, D_G_z2))

            # Remember to save all things you need after all batches finished!!!
            epoch_G_losses.append(errG.item())
            epoch_D_losses.append(errD.item())

            if (iters % 500 == 0) or ((epoch == num_epochs-1) and (i == len(dataloader)-1)):
                with torch.no_grad():
                    fake = generator(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(
                    fake, padding=2, normalize=True))

        iters += 1
        G_losses = append(epoch_G_losses)
        D_losses = append(epoch_D_losses)
        # save model
        # os.makedirs('info', exist_ok=True)
        os.makedirs('model', exist_ok=True)
        model = {
            'discriminator': discriminator.state_dict(),
            'generator': generator.state_dict()
        }
        torch.save(model, os.path.join(
            'model', 'model_' + epoch + '.pt'))
    with open('img_list.pkl', 'wb') as fp:
        pickle.dump(img_list, fp)
    with open('G_loss.pkl', 'wb') as fp:
        pickle.dump(G_losses, fp)
    with open('D_loss.pkl', 'wb') as fp:
        pickle.dump(D_losses, fp)


# def get_sample_image(generator: nn.Module, n_noise: int):
#     """
#     get 100 images sample
#     """
#     z = torch.randn(100, n_noise).to(device)
#     y_hat = generator(z).view(100, WIDTH, HEIGHT)  # (100, 218, 178)
#     result = y_hat.cpu().data.numpy()
#     img = np.zeros([WIDTH * 10, HEIGHT * 10])
#     for j in range(10):
#         img[j*HEIGHT:(j+1)*HEIGHT] = np.concatenate(
#             [x for x in result[j*10:(j+1)*10]], axis=-1)

#     return img


def main(args, logger):

    # Create the dataset by using ImageFolder(get extra point by using customized dataset)
    # remember to preprocess the image by using functions in pytorch
    # https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder
    # https://pytorch.org/docs/stable/torchvision/transforms.html?highlight=totensor#torchvision.transforms.ToTensor
    dataset = dset.ImageFolder(args.dataroot, transform=transforms.Compose([
                               transforms.Resize(64),
                               transforms.CenterCrop(64),
                               transforms.ToTensor(),
                               transforms.Normalize(
                                   (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
    # Create the dataloader
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Create the generator and the discriminator()
    # Initialize them
    # Send them to your device
    generator = Generator(ngpu).to(device)
    discriminator = Discriminator(ngpu).to(device)

    # Setup optimizers for both G and D and setup criterion at the same time
    optimizer_g = optim.Adam(generator.parameters(), lr=args.lr)
    optimizer_d = optim.Adam(discriminator.parameters(), lr=args.lr)
    criterion = nn.BCELoss()

    # Start training~~

    train(dataloader, generator, discriminator, optimizer_g,
          optimizer_d, criterion, args.num_epochs, logger)


if __name__ == '__main__':
    args = common_arg_parser()
    logger = get_logger()
    main(args, logger)

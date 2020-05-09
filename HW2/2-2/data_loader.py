import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, datasets
from glob import glob


def show_face(image):
    """Show image"""
    plt.imshow(image)
    plt.pause(0.001)  # pause a bit so that plots are updated


class AnimationFaceDataset(Dataset):
    """Animation Face dataset."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.img_path = glob(os.path.join(root_dir, '*.png'))
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_path)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = self.img_path[idx]
        image = io.imread(img_name)
        sample = {'image': image}

        if self.transform:
            sample = self.transform(sample)

        return sample


def _test_img_loading():
    """Print the sizes of first 4 samples and show them."""
    face_dataset = AnimationFaceDataset(root_dir='data/')

    fig = plt.figure()

    for i in range(len(face_dataset)):
        sample = face_dataset[i]

        print(i, sample['image'].shape)

        ax = plt.subplot(1, 4, i + 1)
        plt.tight_layout()
        ax.set_title('Sample #{}'.format(i))
        ax.axis('off')
        show_face(**sample)

        if i == 3:
            plt.show()
            break


class Normalize(object):
    """Normalize ndarrays to 0~1."""

    def __call__(self, sample):
        image = sample['image']
        return {'image': image / 256}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image = sample['image']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image).float()}


def _test_with_transform():
    transformed_dataset = AnimationFaceDataset(root_dir='data/',
                                               transform=transforms.Compose([
                                                   ToTensor()
                                               ]))

    for i in range(len(transformed_dataset)):
        sample = transformed_dataset[i]
        print(sample['image'])
        print(i, sample['image'].size())

        if i == 3:
            break


# Helper function to show a batch
def show_face_batch(sample_batched):
    """Show image for a batch of samples."""
    images_batch = sample_batched['image']
    batch_size = len(images_batch)
    im_size = images_batch.size(2)

    grid = utils.make_grid(images_batch)
    plt.imshow(grid.numpy().transpose((1, 2, 0)))

    plt.title('Batch from dataloader')


def _test_data_loader():
    transformed_dataset = AnimationFaceDataset(root_dir='data/',
                                               transform=transforms.Compose([
                                                   ToTensor()
                                               ]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size())

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_face_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


def _test_data_loader_with_normalize():
    transformed_dataset = AnimationFaceDataset(root_dir='data/',
                                               transform=transforms.Compose([
                                                   Normalize(),
                                                   ToTensor()
                                               ]))

    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size())
        print(sample_batched['image'])

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_face_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break


def _test_image_folder():
    # TODO: still have bug - can't find images
    transformed_dataset = datasets.ImageFolder('./data', transform=transforms.Compose([
        Normalize(),
        ToTensor()
    ]))
    dataloader = DataLoader(transformed_dataset, batch_size=4,
                            shuffle=True, num_workers=4)

    for i_batch, sample_batched in enumerate(dataloader):
        print(i_batch, sample_batched['image'].size())
        print(sample_batched['image'])

        # observe 4th batch and stop.
        if i_batch == 3:
            plt.figure()
            show_face_batch(sample_batched)
            plt.axis('off')
            plt.ioff()
            plt.show()
            break

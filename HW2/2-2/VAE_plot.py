# 1. load model
# 2. find 2 samples
# 3. get 2 samples' z (i.e. z_1 and z_2)
# 4. decode from 1 * z_1 + 0 * z_2, ...., 0 * z_1 + 1 * z_2

import ipdb
from skimage import io
from VAE_construct_animationface import VAE, version_info
from data_loader import AnimationFaceDataset, ToTensor, Normalize
import torch
import torch.nn as nn
from torchvision import transforms
import numpy as np
from torchvision.utils import save_image

OUTPUT_SIZE = 8
EPOCH = 20
# KL = 1.0
# KL = 100.0
KL = 0.0

model = VAE()
model.load_state_dict(torch.load(f"model/model_epoch_{EPOCH}_KL_{KL}.pt"))

face_dataset = AnimationFaceDataset('data', transform=transforms.Compose([
    Normalize(),
    ToTensor()
]))


def use_upsampling():
    data_loader = torch.utils.data.DataLoader(face_dataset, batch_size=2)
    # TODO: currently failed
    for data in data_loader:
        _, mu, logvar = model(data['image'])
        upsampled_recon_x = nn.Upsample(
            size=[8, 20], mode='linear')(mu.unsqueeze(0))
        break


def manual():
    # for i in range(0, len(face_dataset), 2):
    for i in range(0, 10, 2):
        image_0 = face_dataset[i]['image']
        image_1 = face_dataset[i + 1]['image']

        _, mu_0, logvar_0 = model(image_0)
        _, mu_1, logvar_1 = model(image_1)

        outputs = []

        for percent in np.linspace(0, 1, num=OUTPUT_SIZE):
            mu = mu_0 * percent + mu_1 * (1 - percent)
            logvar = logvar_0 * percent + logvar_1 * (1 - percent)
            z = model.reparameterize(mu, logvar)

            output = model.decode(z)
            outputs.append(output.view(3, 64, 64))

        save_image(torch.stack(outputs),
                   f'results/generate_{i}_' + version_info.format(EPOCH, KL) + '.png')


if __name__ == "__main__":
    manual()

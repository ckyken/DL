import ipdb
import argparse
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.utils import save_image
import numpy as np
import os
from data_loader import AnimationFaceDataset, ToTensor, Normalize
import matplotlib.pyplot as plt


MODEL_PATH = 'model'
IMG_HEIGHT = 64
COLOR = 3

parser = argparse.ArgumentParser(description='VAE Animation Face Generation')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 128)')
parser.add_argument('--epochs', type=int, default=20, metavar='N',
                    help='number of epochs to train (default: 20)')
parser.add_argument('--learning-rate', type=float, default=1e-4, metavar='N',
                    help='learning rate')
parser.add_argument('--kl-scale', type=float, default=1.0, metavar='N',
                    help='number of scale of KL term in ELBO')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--test-split-ratio', type=float, default=0.2, metavar='N',
                    help='split ratio for the test set')
parser.add_argument('--model-path', type=str, default=MODEL_PATH, metavar='PATH',
                    help='model save path')
args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

device = torch.device("cuda" if args.cuda else "cpu")

kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}

face_dataset = AnimationFaceDataset('data', transform=transforms.Compose([
    Normalize(),
    ToTensor()
]))

test_size = int(len(face_dataset) * args.test_split_ratio)
lengths = [len(face_dataset) - test_size, test_size]
train_set, test_set = random_split(face_dataset, lengths)

train_loader = torch.utils.data.DataLoader(
    train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    test_set, batch_size=args.batch_size, shuffle=True, **kwargs)


version_info = 'epoch_{}_KL_{}'


class VAE(nn.Module):
    def __init__(self, img_height: int = IMG_HEIGHT, color: int = COLOR):
        super(VAE, self).__init__()

        self.img_height = img_height
        self.img_size = img_height**2 * color
        self.fc1 = nn.Linear(self.img_size, 400)
        self.fc21 = nn.Linear(400, 20)
        self.fc22 = nn.Linear(400, 20)
        self.fc3 = nn.Linear(20, 400)
        self.fc4 = nn.Linear(400, self.img_size)

    # def __init__(self, img_height: int = IMG_HEIGHT, color: int = COLOR):
    #     super(VAE, self).__init__()
    #     self.CNN = nn.Sequential(

    #         Conv2d(3, 3, 3, 1, 1),
    #         BatchNorm2d(4),
    #         ReLU(inplace=True),
    #         MaxPool2d(2, 2),

    #         Conv2d(3, 3, 3, 1, 1),
    #         BatchNorm2d(4),
    #         ReLU(inplace=True),
    #         MaxPool2d(2, 2),
    #     )
    #     self.img_height = img_height
    #     self.img_size = img_height**2 * color
    #     self.fc1 = nn.Linear(self.img_size, 400)
    #     self.fc21 = nn.Linear(400, 20)
    #     self.fc22 = nn.Linear(400, 20)
    #     self.fc3 = nn.Linear(20, 400)
    #     self.fc4 = nn.Linear(400, self.img_size)

    def encode(self, x):
        h1 = F.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        h3 = F.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x.view(-1, self.img_size))
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        return recon_x, mu, logvar


model = VAE().to(device)
optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, logvar):
    """ i.e. -ELBO """
    BCE = F.binary_cross_entropy(
        recon_x, x.view(-1, COLOR * IMG_HEIGHT**2), reduction='sum')

    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())

    return BCE + KLD * args.kl_scale


def train(epoch):
    model.train()
    train_loss = 0
    ELBO = []
    for batch_idx, data in enumerate(train_loader):
        data = data['image']
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item() / len(data)))
            ELBO.append(loss.item() / len(data))
    print('====> Epoch: {} Average loss: {:.4f}'.format(
          epoch, train_loss / len(train_loader.dataset)))
    return ELBO


def test(epoch):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            data = data['image']
            data = data.to(device)
            recon_batch, mu, logvar = model(data)
            test_loss += loss_function(recon_batch, data, mu, logvar).item()
            if i == 0:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        recon_batch.view(args.batch_size, COLOR, IMG_HEIGHT, IMG_HEIGHT)[:n]])
                save_image(comparison.cpu(),
                           'results/reconstruction_' + version_info.format(epoch, args.kl_scale) + '.png', nrow=n)

    test_loss /= len(test_loader.dataset)
    print('====> Test set loss: {:.4f}'.format(test_loss))


if __name__ == "__main__":
    print(np.sum([torch.numel(param) for param in model.parameters()]))
    os.makedirs('results', exist_ok=True)
    os.makedirs(MODEL_PATH, exist_ok=True)
    ELBO = []
    for epoch in range(1, args.epochs + 1):
        ELBO＿part = train(epoch)
        ELBO.extend(ELBO＿part)
        torch.save(model.state_dict(), os.path.join(
            MODEL_PATH, 'model_' + version_info.format(epoch, args.kl_scale) + '.pt'))
        test(epoch)
        with torch.no_grad():
            sample = torch.randn(64, 20).to(device)
            sample = model.decode(sample).cpu()
            save_image(sample.view(64, COLOR, IMG_HEIGHT, IMG_HEIGHT),
                       'results/sample_' + version_info.format(epoch, args.kl_scale) + '.png')

    plt.figure()
    plt.plot(list(range(1, len(ELBO) + 1)), ELBO)
    plt.show()
